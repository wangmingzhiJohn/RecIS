from typing import List, Tuple

import torch


LocalDataResourceImpl = torch.classes.recis.LocalDataResource


@torch.jit.script
class LocalDataResource:
    def __init__(self):
        self._impl = LocalDataResourceImpl()

    def load_by_batch(
        self,
        input_tensors: List[torch.Tensor],
        sample_tag: str,
        dedup_tag: str,
        weight_tag: str,
        skey_name: str,
        put_back: bool,
        names: List[str],
        ragged_ranks: torch.Tensor,
        ignore_invalid_dedup_tag: bool,
    ):
        self._impl.load_by_batch(
            input_tensors,
            sample_tag,
            dedup_tag,
            weight_tag,
            skey_name,
            put_back,
            names,
            ragged_ranks,
            ignore_invalid_dedup_tag,
        )

    def sample_ids(
        self,
        sample_tag_tensors: List[torch.Tensor],
        dedup_tag_tensors: List[torch.Tensor],
        sample_cnts: torch.Tensor,
        avoid_conflict: bool,
        pos_num: int,
        avoid_conflict_with_all_dedup_tags: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._impl.sample_ids(
            sample_tag_tensors,
            dedup_tag_tensors,
            sample_cnts,
            avoid_conflict,
            pos_num,
            avoid_conflict_with_all_dedup_tags,
        )

    def valid_sample_ids(
        self, sample_ids: torch.Tensor, default_value: int
    ) -> torch.Tensor:
        return self._impl.valid_sample_ids(sample_ids, default_value)

    def pack_feature(
        self,
        local_data_sample_ids: torch.Tensor,
        decorate_skey: bool,
        names: List[str],
        ragged_ranks: torch.Tensor,
        default_value: int,
    ) -> List[torch.Tensor]:
        if decorate_skey:
            raise NotImplementedError
        else:
            return self._impl.extract_feature(
                local_data_sample_ids, names, ragged_ranks, default_value
            )
