"""Python wrappers for reader LocalDataSampler."""

import time
from enum import Enum
from functools import wraps

import numpy as np
import torch

from recis.data.local_data_resource import LocalDataResource
from recis.ragged.tensor import RaggedTensor
from recis.utils.logger import Logger


class Timer:
    def __init__(self, name=None):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.perf_counter()
        self.elapsed_time = self.end_time - self.start_time
        if self.name:
            print(
                f"[{self.name}] duration: {self.elapsed_time:.6f} seconds", flush=True
            )
        else:
            print(f"duration: {self.elapsed_time:.6f} seconds", flush=True)

    def timeit(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            with Timer(self.name or func.__name__):
                return func(*args, **kwargs)

        return wrapper

    def result(self):
        return self.elapsed_time

    def restart(self):
        self.start_time = time.perf_counter()
        self.end_time = None
        self.elapsed_time = None

    def stop(self):
        if self.start_time is not None:
            self.end_time = time.perf_counter()
            self.elapsed_time = self.end_time - self.start_time
        else:
            raise RuntimeError("Timing not started.")
        if self.name:
            print(
                f"[{self.name}] duration: {self.elapsed_time:.6f} seconds", flush=True
            )
        else:
            print(f"duration: {self.elapsed_time:.6f} seconds", flush=True)


logger = Logger(__name__)


DataType = Enum("DataType", ("VarLen", "FixedLen"))


def string_to_ragged_char(arr):
    """
    Convert string array to ragged char tensor.
    """
    values = np.frombuffer(b"".join(arr), dtype=np.uint8)
    values = torch.from_numpy(values.copy())
    lens = np.char.str_len(arr)
    splits = np.concatenate(([0], np.cumsum(lens)))
    splits = torch.from_numpy(splits)
    return RaggedTensor(values, [splits])


def ragged_char_to_string(ragged_char):
    # TODO(yanzhensong.yzs): write a cpp operator to replace this
    values = ragged_char.values().numpy()
    splits = ragged_char.offsets()[0].numpy()
    segments = np.split(values, splits[1:-1])
    return np.array([segment.tobytes() for segment in segments])


class LocalDataSampler:
    """init LocalDataSampler.
    Args:
    Returns:
        LocalDataSampler
    """

    def __init__(
        self,
        sample_tag,
        dedup_tag,
        weight_tag,
        skey_name,
        put_back=True,
        ignore_invalid_dedup_tag=True,
        profiling=False,
    ):
        self._sample_tag = sample_tag
        self._dedup_tag = dedup_tag
        self._weight_tag = weight_tag
        self._skey_name = skey_name
        self._put_back = put_back
        self._ignore_invalid_dedup_tag = ignore_invalid_dedup_tag
        # only support loading by batch for now
        self._load_by_batch = True
        self._is_compressed = False
        self._initialized = False
        self._table = 0
        self._local_data_resource = LocalDataResource()
        self._names = None
        self._ragged_ranks = None
        self._ragged_ranks_t = None
        self._profiling = profiling

    def unfold(self, batch):
        batch_size_t = None
        values_t = []
        splits_t = []
        for name in sorted(batch):
            if name == "_sample_group_id":
                continue
            data = batch[name]
            if isinstance(data, RaggedTensor):
                values_t.append(data.values())
                splits_t.extend(data.offsets()[::-1])
                if batch_size_t is None:
                    batch_size_t = torch.tensor(
                        data.offsets()[-1].shape[0] - 1, dtype=torch.int32
                    )
                if data.weight() is not None:
                    values_t.append(data.weight())
                    # TODO(yzs): this splits can be optimized
                    splits_t.extend(data.offsets()[::-1])
            else:
                values_t.append(data)
                if batch_size_t is None:
                    batch_size_t = torch.tensor(data.shape[0], dtype=torch.int32)
        assert batch_size_t is not None
        return [batch_size_t] + values_t + splits_t

    def fold(self, args, names, ragged_ranks):
        val_idx = 1  # skip batch size tensor
        split_idx = len(names) + val_idx
        table = {}

        for name, rank in zip(names, ragged_ranks):
            if rank > 0:
                if name in self.np_names:
                    table[name] = ragged_char_to_string(
                        RaggedTensor(
                            args[val_idx],
                            args[split_idx + rank - 1 : split_idx - 1 : -1],
                        )
                    )
                elif name not in table:
                    table[name] = RaggedTensor(
                        args[val_idx], args[split_idx + rank - 1 : split_idx - 1 : -1]
                    )
                else:  # name in table
                    table[name].set_weight(args[val_idx])
            else:
                if name in table:
                    raise RuntimeError(
                        f"Feature {name} with weight should be sparse feature."
                    )
                table[name] = args[val_idx]
            val_idx += 1
            split_idx += rank
        return table

    def init_names_ranks(self, batch):
        names = []
        ragged_ranks = []
        self.np_names = set()
        for name in sorted(batch):
            if name == "_sample_group_id":
                continue
            names.append(name)
            data = batch[name]
            if isinstance(data, list):
                batch[name] = string_to_ragged_char(
                    data[0]
                )  # TODO(yzs): is this right?
                ragged_ranks.append(1)
                self.np_names.add(name)
            elif isinstance(data, RaggedTensor):
                ragged_ranks.append(len(data.offsets()))
                if data.weight() is not None:
                    names.append(name)
                    ragged_ranks.append(len(data.offsets()))
            elif isinstance(data, torch.Tensor):
                ragged_ranks.append(0)
            else:
                raise NotImplementedError

        ragged_ranks_t = torch.tensor(ragged_ranks, dtype=torch.int32)
        return names, ragged_ranks, ragged_ranks_t

    def sample_ids(
        self,
        sample_tag_ragged_tensor,
        dedup_tag_ragged_tensor,
        sample_cnts,
        avoid_conflict=True,
    ):
        sample_tag_tensors = [
            sample_tag_ragged_tensor.values()
        ] + sample_tag_ragged_tensor.offsets()[::-1]
        dedup_tag_tensors = [
            dedup_tag_ragged_tensor.values()
        ] + dedup_tag_ragged_tensor.offsets()[::-1]
        pos_num = dedup_tag_tensors[0].numel()
        if isinstance(sample_cnts, int):
            sample_cnts = torch.tensor([sample_cnts])
        num_sample_cnts = len(sample_cnts)
        if num_sample_cnts != 1 and num_sample_cnts != pos_num:
            raise RuntimeError(
                f"sample_cnts size {num_sample_cnts} should be equal to positive samples size {pos_num}."
            )

        result = self._local_data_resource.sample_ids(
            sample_tag_tensors, dedup_tag_tensors, sample_cnts, avoid_conflict, pos_num
        )
        return result[0]

    def valid_sample_ids(self, sample_ids, default_value):
        return self._local_data_resource.valid_sample_ids(sample_ids, default_value)

    def pack_feature(
        self, local_data_sample_ids, decorate_skey=False, default_value=-1
    ):
        if self._profiling:
            timer = Timer("pack_feature")
            timer.restart()
        if decorate_skey:
            logger.info("Please make sure each sample has the same number of samples!")
        names = self._names
        ragged_ranks = self._ragged_ranks
        ragged_ranks_t = self._ragged_ranks_t

        output_tensor_list = self._local_data_resource.pack_feature(
            local_data_sample_ids,
            decorate_skey=decorate_skey,
            names=names,
            ragged_ranks=ragged_ranks_t,
            default_value=default_value,
        )
        if self._profiling:
            timer.stop()
            timer = Timer("fold")
            timer.restart()

        output_sample_table = self.fold(output_tensor_list, names, ragged_ranks)

        if self._profiling:
            timer.stop()

        return output_sample_table

    def load_by_batch(self, batch):
        sample_table = batch[self._table]
        assert self._dedup_tag in sample_table, self._dedup_tag + " must be in batch"
        assert self._weight_tag in sample_table, self._weight_tag + " must be in batch"
        assert self._skey_name in sample_table, self._skey_name + " must be in batch"
        assert self._sample_tag in sample_table, self._sample_tag + " must be in batch"
        self._names, self._ragged_ranks, self._ragged_ranks_t = self.init_names_ranks(
            sample_table
        )
        input_tensor_list = self.unfold(sample_table)

        self._local_data_resource.load_by_batch(
            input_tensor_list,
            sample_tag=self._sample_tag,
            dedup_tag=self._dedup_tag,
            weight_tag=self._weight_tag,
            skey_name=self._skey_name,
            put_back=self._put_back,
            names=self._names,
            ragged_ranks=self._ragged_ranks_t,
            ignore_invalid_dedup_tag=self._ignore_invalid_dedup_tag,
        )
        self._initialized = True

    def combine_vector_with_sample_counts(
        self, origin_vector, sample_cnts, sampled_vector
    ):
        return torch.ops.recis.combine_vector_with_sample_counts(
            origin_vector, sample_cnts, sampled_vector
        )
