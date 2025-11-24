import contextlib
import subprocess
import sys
import threading

import torch
import torch.distributed.rpc as rpc

from recis.data.local_data_sampler import LocalDataSampler
from recis.utils.logger import Logger


logger = Logger(__name__)


class _ReadWriteLock:
    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0
        self._upgrader = False
        self._writer = False

    def acquire_read(self):
        with self._read_ready:
            while self._upgrader or self._writer:
                self._read_ready.wait()
            self._readers += 1

    def release_read(self):
        with self._read_ready:
            self._readers -= 1
            if self._readers == 0:
                self._read_ready.notify_all()

    def acquire_write(self):
        with self._read_ready:
            while self._writer or self._readers > 0 or self._upgrader:
                self._read_ready.wait()
            self._writer = True

    def release_write(self):
        with self._read_ready:
            self._writer = False
            self._read_ready.notify_all()

    def acquire_upgrade(self):
        with self._read_ready:
            while self._upgrader or self._writer:
                self._read_ready.wait()
            self._upgrader = True

    def release_upgrade(self):
        with self._read_ready:
            self._upgrader = False
            self._read_ready.notify_all()

    def acquire_upgrade_to_write(self):
        with self._read_ready:
            assert self._upgrader
            while self._writer or self._readers > 0:
                self._read_ready.wait()
            self._writer = True

    def release_upgrade_to_write(self):
        with self._read_ready:
            self._writer = False
            self._read_ready.notify_all()

    @contextlib.contextmanager
    def write_lock(self):
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()

    @contextlib.contextmanager
    def read_lock(self):
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextlib.contextmanager
    def upgrade_lock(self):
        self.acquire_upgrade()
        try:
            yield
        finally:
            self.release_upgrade()

    @contextlib.contextmanager
    def upgrade_to_write_lock(self):
        self.acquire_upgrade_to_write()
        try:
            yield
        finally:
            self.release_upgrade_to_write()


class _SamplerWrapper:
    """
    Because sampler will be called in multi-thread, the class needs to be thread safe.
    """

    def __init__(self, profiling: bool = False):
        self._sampler = None
        self._rw_lock = _ReadWriteLock()
        self._profiling = profiling

    def reload_sampler_by_batch(
        self,
        batch,
        dedup_tag,
        sample_tag=None,
        weight_tag=None,
        skey_name=None,
        put_back=True,
        use_positive=False,
        is_compressed=False,
        ignore_invalid_dedup_tag=True,
        profiling=False,
    ):
        logger.info("Starting to load data into the sampler.")
        new_sampler = LocalDataSampler(
            sample_tag,
            dedup_tag,
            weight_tag,
            skey_name,
            put_back,
            ignore_invalid_dedup_tag,
            profiling,
        )
        new_sampler.load_by_batch(batch)
        with self._rw_lock.upgrade_lock():
            with self._rw_lock.upgrade_to_write_lock():
                self._sampler = new_sampler
        logger.info("Finished loading data into the sampler.")

    def sample_ids(
        self,
        sample_tag_ragged_tensor,
        dedup_tag_ragged_tensor,
        sample_cnts,
        avoid_conflict=True,
    ):
        with self._rw_lock.read_lock():
            return self._sampler.sample_ids(
                sample_tag_ragged_tensor,
                dedup_tag_ragged_tensor,
                sample_cnts,
                avoid_conflict,
            )

    def pack_feature(
        self, local_data_sample_ids, decorate_skey=False, default_value=-1
    ):
        with self._rw_lock.read_lock():
            return self._sampler.pack_feature(
                local_data_sample_ids,
                decorate_skey=decorate_skey,
                default_value=default_value,
            )

    def valid_sample_ids(self, ids, default_value=-1):
        with self._rw_lock.read_lock():
            return self._sampler.valid_sample_ids(ids, default_value)


_sampler_wrapper = _SamplerWrapper()


def init_sampler_client(
    name: str, world_size: int, rank: int, master_port: int, rpc_timeout: int = 5 * 60
):
    """
    Initialize the client on the worker side.

    Args:
        name: Unique identifier for the worker client. Must not conflict with other instances.
        world_size: Total number of nodes in the communication group. Must match the world_size value set in create_sampler_process.
        rank: Unique identifier of this client within the communication group.
        master_port: Port number used by the rank 0 node in the communication group. Must be consistent with the master_port specified in create_sampler_process.
    """
    rpc.init_rpc(
        name,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=1,
            rpc_timeout=rpc_timeout,
            init_method=f"tcp://localhost:{master_port}",
            _channels=["basic"],
            _transports=["shm"],
        ),
    )


_server_rank = None
_world_size = None


def _sampler_service(
    name: str,
    world_size: int,
    rank: int,
    master_port: int,
    num_worker_threads: int = 16,
    rpc_timeout: int = 5 * 60,
):
    rpc.init_rpc(
        name,
        rank=rank,
        world_size=world_size,
        rpc_backend_options=rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=num_worker_threads,
            rpc_timeout=rpc_timeout,
            init_method=f"tcp://localhost:{master_port}",
            _channels=["basic"],
            _transports=["shm"],
        ),
    )
    _server_rank = rank
    _world_size = world_size
    rpc.shutdown()


def start_sampler_process(
    name: str,
    world_size: int,
    rank: int,
    master_port: int,
    num_worker_threads: int = 16,
    rpc_timeout: int = 5 * 60,
    inter_op_threads: int = 16,
):
    p = subprocess.Popen(
        f"{sys.executable} {__file__} {name} {world_size} {rank} {master_port} {num_worker_threads} {rpc_timeout} {inter_op_threads}",
        shell=True,
    )
    return p


def get_sampler_service():
    return _sampler_wrapper


if __name__ == "__main__":
    name = sys.argv[1]
    world_size = int(sys.argv[2])
    rank = int(sys.argv[3])
    master_port = int(sys.argv[4])
    num_worker_threads = int(sys.argv[5])
    rpc_timeout = int(sys.argv[6])
    inter_op_threads = int(sys.argv[7])
    torch.set_num_interop_threads(inter_op_threads)
    _sampler_service(
        name, world_size, rank, master_port, num_worker_threads, rpc_timeout
    )
