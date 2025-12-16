import os

import numpy as np
import torch
import torch.distributed.rpc as rpc

from recis.data.sampler_service import (
    _SamplerWrapper,
    get_sampler_service,
    init_sampler_client,
    start_sampler_process,
)
from recis.ragged.tensor import RaggedTensor


def call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)


class LocalRpcDataSampler:
    """To avoid distributed sampling while still being able to load the
    full negative sample table into memory, we designed a solution using
    a local negative sampling service.

    On each machine node, only one negative sampling server process is
    launched, which is shared by all training processes on that node.
    During task initialization, the negative sample table is loaded into
    this server. During training, each training process performs negative
    sampling by making RPC requests to the local server process.

    LocalRpcDataSampler is the negative sampling client for each training
    process. This object automatically starts the negative sampling server
    upon initialization, and all sampling tasks can be completed through
    this object.

    Note: Since the full negative sample table needs to be loaded into the
    memory of each physical machine, whole-machine training is recommended
    (e.g., an 8-GPU single machine), so that the entire system memory can
    be utilized.

    Example:

    .. code-block:: python

        from recis.io.odps_dataset import OdpsDataset
        from recis.io.file_sharding import get_table_size
        from recis.data.local_rpc_data_sampler import LocalRpcDataSampler


        # Get the dataset
        def get_dataset(table, batch_size, varlen_features, worker_num, worker_idx):
            dataset = OdpsDataset(
                batch_size, worker_idx=worker_idx, worker_num=worker_num
            )  # Initialize the dataset
            dataset.add_path(table)  # Add ODPS table
            # Add features to the dataset
            for f in varlen_features:
                dataset.varlen_feature(f)


        WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
        RANK = int(os.getenv("RANK", 0))
        # Get the positive sample dataset
        pos_dataset = get_dataset(
            pos_table, pos_batch_size, pos_varlen_features, WORLD_SIZE, RANK
        )

        # Initialize the negative sampler
        neg_sampler = LocalRpcDataSampler(
            sample_tag,  #  category
            dedup_tag,  # item id
            weight_tag,  # sampling weight
            skey_name,
        )

        # Only the training process with local rank = 0 loads the negative sample table
        if int(os.environ.get("LOCAL_PROCESS_RANK", 0)) == 0:
            neg_table_size = get_table_size(
                neg_table
            )  # Get the number of samples in the negative table
            # Get the negative sample dataset
            neg_dataset = get_dataset(
                neg_table,
                neg_table_size,
                neg_varlen_features,
                # Note: For the negative sample dataset, worker_num and worker_idx
                # must be set to 1 and 0 respectively. Otherwise, it will become distributed reading,
                # resulting in incomplete negative sample tables on each machine.
                worker_num=1,
                worker_idx=0,
            )
            neg_data = next(iter(neg_dataset))[1]  # Read all negative samples
            neg_sampler.reload(neg_data)  # Load negative samples into the sampler


        # Negative sampling function
        def neg_sampling_fn(batch):
            # batch contains positive samples
            sample_tag_ragged_tensor = batch[0][sample_tag]
            dedup_tag_ragged_tensor = batch[0][dedup_tag]
            # Sample (sample_size - 1) negative samples for each positive sample.
            # Values in sample_cnts don't need to be equal, enabling dynamic negative sampling.
            sample_cnts = sample_size - torch.ones_like(
                dedup_tag_ragged_tensor.values()
            )
            # Pass sample_tag and dedup_tag to obtain a set of negative sample IDs
            neg_sample_ids = neg_sampler.sample_ids(
                dedup_tag_ragged_tensor=dedup_tag_ragged_tensor,
                sample_cnts=sample_cnts,
                sample_tag_ragged_tensor=sample_tag_ragged_tensor,
            )
            # Handle cases where some positive samples may not exist in the negative sample
            # table by filling with a default value: murmurhash(0).
            # Users can choose other default values.
            pos_ids = neg_sampler.valid_sample_ids(
                dedup_tag_ragged_tensor.values(), default_value=5533571732986600803
            )
            # Concatenate positive and negative sample IDs.
            # For example, if positive IDs are [1,2,3], sample count is 2,
            # and sampled negative IDs are [5,6,7,8,9,10], the concatenated result is [1,5,6,2,7,8,3,9,10]
            pos_neg_ids = neg_sampler.combine_vector_with_sample_counts(
                pos_ids, sample_cnts, neg_sample_ids
            )

            # Query all features from the negative sample table using the concatenated positive-negative IDs
            output_sample_table = neg_sampler.pack_feature(
                pos_neg_ids, default_value=5533571732986600803
            )
            # Combine original positive samples with negative samples
            combined_batch = neg_sampler.combine(
                batch, output_sample_table, sample_cnts
            )
            return combined_batch


        # Add negative sampling transformation to the positive sample dataset
        pos_dataset.transform_ragged_batch(neg_sampling_fn)

        # Read samples from the positive sample dataset
        for batch in pos_dataset:
            model(batch)
    """

    def __init__(
        self,
        sample_tag,
        dedup_tag,
        weight_tag,
        skey_name,
        put_back=True,
        use_positive=False,
        ignore_invalid_dedup_tag=False,
        rpc_server_name="sampler_server",
        rpc_num_clients_per_worker=1,
        rpc_client_idx_in_worker=0,
        rpc_master_port=13254,
        rpc_server_num_threads=16,
        rpc_server_timeout=5 * 60,
        rpc_server_inter_op_threads=16,
        rpc_server_profiling=False,
    ):
        """
        Create a local RPC sampler. This object automatically launches a negative sampling server process.

        Args:
            sample_tag: Column name for the type field in the negative sample table; used to sample from items of the same type.
                This feature must be a sparse feature and represented as array<bigint> in the ODPS table.
                For sampling with replacement, multiple values are allowed. For sampling without replacement, it must be single-valued.
                This column must exist in the ODPS table. If you don't need this column, you can set it to
                0 in the ODPS table.
            dedup_tag: Column name for the item ID in the negative sample table, used for deduplication during sampling. The IDs must be unique,
                but note that this is not the unique identifier of the sample itself.
                This feature must be a single-value sparse feature, represented as array<bigint> in the ODPS table.
                This column must exist in the ODPS table.
            weight_tag: Column name for the item sampling weight in the negative sample table, used for weighted sampling.
                This feature must be a single-value dense feature, represented as float or double in the ODPS table.
                This column must exist in the ODPS table. If you don't need this column, you can set it to
                1.0 in the ODPS table.
            skey_name: Column name for the sample ID in the negative sample table, which serves as the unique identifier of the sample
                (typically a string derived from the ad ID), of type string.
                This column must exist in the ODPS table. If you don't need this column, you can set it to
                empty strings in the ODPS table.
            put_back: Type of sampling. Default is True (sampling with replacement). Set to False for sampling without replacement.
                Sampling with replacement has time complexity O(1); sampling without replacement has time complexity O(log(n)),
                where n is the total number of ads in the same category.
            use_positive: Reserved parameter; Leave it as default.
            ignore_invalid_dedup_tag: Reserved parameter; Leave it as default.
            rpc_server_name: Name of the server process in the communication group; use default value.
            rpc_num_clients_per_worker: Number of negative sampling clients per training process; typically 1.
            rpc_client_idx_in_worker: Index of the negative sampling client within the worker.
            rpc_master_port: Port number for the negative sampling server.
            rpc_server_num_threads: Number of threads for the negative sampling server.
            rpc_server_timeout: Timeout for the negative sampling server.
            rpc_server_inter_op_threads: Number of inter-op threads for the negative sampling server.
            rpc_server_profiling: Whether to enable profiling for the negative sampling server.
        """
        self._sample_tag = sample_tag
        self._dedup_tag = dedup_tag
        self._weight_tag = weight_tag
        self._skey_name = skey_name
        self._put_back = put_back
        self._use_positive = use_positive
        self._ignore_invalid_dedup_tag = ignore_invalid_dedup_tag
        rpc_num_clients_per_pod = (
            int(os.environ.get("worker_per_pod", 1)) * rpc_num_clients_per_worker
        )
        self._rpc_server_name = rpc_server_name
        self._rpc_server_rank = 0
        self._local_rank = int(os.environ.get("LOCAL_PROCESS_RANK", 0))
        self._rpc_server_profiling = rpc_server_profiling
        if self._local_rank == 0:
            self._proc = start_sampler_process(
                rpc_server_name,
                rpc_num_clients_per_pod + 1,
                self._rpc_server_rank,
                rpc_master_port,
                rpc_server_num_threads,
                rpc_server_timeout,
                rpc_server_inter_op_threads,
            )

        init_sampler_client(
            f"worker{self._local_rank}",
            rpc_num_clients_per_pod + 1,
            self._local_rank * rpc_num_clients_per_worker
            + rpc_client_idx_in_worker
            + 1,
            rpc_master_port,
            rpc_timeout=rpc_server_timeout,
        )
        self.sampler_service = rpc.remote(rpc_server_name, get_sampler_service)

    def __del__(self):
        try:
            rpc.shutdown()
            if hasattr(self, "_proc"):
                # self._proc.join()
                self._proc.wait()
        except Exception:
            self._proc.kill()

    def reload(self, batch: list[dict[str, torch.Tensor | RaggedTensor | np.ndarray]]):
        """Reload the negative sample table.

        Args:
            batch: Negative sample table.
        """
        remote_method(
            _SamplerWrapper.reload_sampler_by_batch,
            self.sampler_service,
            batch=batch,
            dedup_tag=self._dedup_tag,
            weight_tag=self._weight_tag,
            skey_name=self._skey_name,
            sample_tag=self._sample_tag,
            put_back=self._put_back,
            use_positive=self._use_positive,
            ignore_invalid_dedup_tag=self._ignore_invalid_dedup_tag,
            profiling=self._rpc_server_profiling,
        )

    def sample_ids(
        self,
        dedup_tag_ragged_tensor: RaggedTensor,
        sample_cnts: torch.Tensor | int,
        sample_tag_ragged_tensor: RaggedTensor = None,
        avoid_conflict: bool = True,
        avoid_conflict_with_all_dedup_tags: bool = False,
    ) -> torch.Tensor:
        """Sample negative item IDs.

        Args:
            dedup_tag_ragged_tensor: Positive item IDs to be sampled.
            sample_cnts: Number of negative samples to be sampled for each positive sample.
            sample_tag_ragged_tensor: Type of items to be sampled. If you don't need this column,
                you can set it all to 0.
                E.g. sample_tag_ragged_tensor = RaggedTensor(values=torch.zeros_like(dedup_tag_ragged_tensor.values()), offsets=dedup_tag_ragged_tensor.offsets())
            avoid_conflict: Whether the sampled negative items IDs should avoid duplication with positive items IDs.
            avoid_conflict_with_all_dedup_tags: If true, the sampled negative IDs will be different from all dedup tags.

        Returns:
            Negative item IDs.
        """
        return remote_method(
            _SamplerWrapper.sample_ids,
            self.sampler_service,
            dedup_tag_ragged_tensor=dedup_tag_ragged_tensor,
            sample_cnts=sample_cnts,
            sample_tag_ragged_tensor=sample_tag_ragged_tensor,
            avoid_conflict=avoid_conflict,
            avoid_conflict_with_all_dedup_tags=avoid_conflict_with_all_dedup_tags,
        )

    def combine_vector_with_sample_counts(
        self,
        origin_vector: torch.Tensor,
        sample_cnts: torch.Tensor,
        sampled_vector: torch.Tensor,
    ) -> torch.Tensor:
        """Concatenate positive and negative item IDs.

        Args:
            origin_vector: Positive item IDs.
            sample_cnts: Number of negative samples to be sampled for each positive sample.
            sampled_vector: Negative item IDs.

        Returns:
            Combined item IDs.
        """
        return torch.ops.recis.combine_vector_with_sample_counts(
            origin_vector, sample_cnts, sampled_vector
        )

    def pack_feature(
        self,
        local_data_sample_ids: torch.Tensor,
        default_value: int = -1,
    ) -> dict[str, torch.Tensor | RaggedTensor | np.ndarray]:
        """Query all features from the negative sample table using the items IDs.

        Args:
            local_data_sample_ids: Item IDs.
            default_value: Default value for features that are not found in the negative sample table.
                This should be equal to the default value in the valid_sample_ids function, and the
                default value must exist in the negative sample table.

        Returns:
            Negative sample table.
        """
        return remote_method(
            _SamplerWrapper.pack_feature,
            self.sampler_service,
            local_data_sample_ids,
            False,
            default_value,
        )

    def combine(
        self,
        batch: list[dict[str, torch.Tensor | RaggedTensor | np.ndarray]],
        output_sample_table: dict[str, torch.Tensor | RaggedTensor | np.ndarray],
        sample_counts: torch.Tensor | int,
    ) -> list[dict[str, torch.Tensor | RaggedTensor | np.ndarray]]:
        """Combine positive and negative samples.

        Args:
            batch: Positive samples.
            output_sample_table: Packed samples.
            sample_counts: Number of negative samples to be sampled for each positive sample.

        Returns:
            Combined samples.
        """
        group_id_t = []
        indicators_t = []
        indicators_name = []
        do_classify = False
        for dic in batch:
            for name in dic:
                data = dic[name]
                if name.startswith("_indicator"):
                    indicators_name.append(name)
                    assert isinstance(data, torch.Tensor)
                    indicators_t.append(data)
                elif name == "_sample_group_id":
                    assert isinstance(data, torch.Tensor)
                    group_id_t.append(data)
                    do_classify = True
        input_group_indicators_t = group_id_t + indicators_t
        if isinstance(sample_counts, int):
            output_group_indicators_t = self.list_tile(
                input_group_indicators_t, sample_counts
            )
        else:
            assert isinstance(sample_counts, torch.Tensor)
            output_group_indicators_t = self.list_tile_with_sample_counts(
                input_group_indicators_t, sample_counts=sample_counts
            )
        idx = 0
        if do_classify:
            output_sample_table["_sample_group_id"] = output_group_indicators_t[idx]
            idx += 1
        for dic in batch[1:]:
            indicator_name = indicators_name.pop(0)
            dic[indicator_name] = output_group_indicators_t[idx]
            idx += 1
        output_batch = [output_sample_table]
        output_batch.extend(batch[1:])
        return output_batch

    def list_tile(self, inputs, sample_cnt):
        outputs = []
        for input in inputs:
            input_r = torch.reshape(input, [-1, 1])
            input_tl = torch.tile(input_r, [1, sample_cnt + 1])
            output = torch.reshape(input_tl, [-1])
            outputs.append(output)
        return outputs

    def list_tile_with_sample_counts(self, inputs, sample_counts):
        outputs = []
        for input in inputs:
            output = torch.ops.recis.tile_with_sample_counts(input, sample_counts)
            outputs.append(output)
        return outputs

    def valid_sample_ids(
        self, ids: torch.Tensor, default_value: int = -1
    ) -> torch.Tensor:
        """Set invalid IDs to the default value.

        Args:
            ids: Item IDs.
            default_value: Default value for features that are not found in the negative sample table.
                This should be equal to the default value in the pack_feature function, and the
                default value must exist in the negative sample table.

        Returns:
            Valid item IDs.
        """
        return remote_method(
            _SamplerWrapper.valid_sample_ids, self.sampler_service, ids, default_value
        )
