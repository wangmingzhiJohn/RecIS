import os
import random
import shutil
import unittest
from collections import defaultdict
from typing import List, Optional, Union

import torch
import torch.distributed as dist
import torch.testing._internal.common_utils as common

from recis.nn.functional import fused_ops
from recis.nn.initializers import ConstantInitializer
from recis.nn.modules.embedding import EmbeddingOption
from recis.nn.modules.embedding_engine import EmbeddingEngine
from recis.nn.modules.hashtable import split_sparse_dense_state_dict
from recis.optim import SparseAdamWTF
from recis.ragged.tensor import RaggedTensor
from recis.serialize import Saver as SSaver
from recis.serialize.checkpoint_reader import CheckpointReader
from recis.utils.logger import Logger


logger = Logger(__name__)


def ts_equal(t1: torch.Tensor, t2: torch.Tensor) -> bool:
    t1 = t1.cuda()
    t2 = t2.cuda()
    if t1.shape != t2.shape:
        print(
            f"Tensors have different shapes: t1.shape={t1.shape}, t2.shape={t2.shape}"
        )
        return False
    sorted_t1, _ = torch.sort(t1)
    sorted_t2, _ = torch.sort(t2)
    is_equal = torch.equal(sorted_t1, sorted_t2)
    if not is_equal:
        print(
            f"Tensors do not contain the same set of elements, {sorted_t1}, {sorted_t2}"
        )
        for i in range(sorted_t1.shape[0]):
            if sorted_t1[i] != sorted_t2[i]:
                print(
                    f"  - At index {i}: t1 has value {sorted_t1[i].item()}, but t2 has value {sorted_t2[i].item()}"
                )
    return is_equal


def gen_2d_ragged_tensor(
    batch_size: int,
    max_row_length: int,
    must_have_vals: Union[int, List[int], None] = None,
    must_not_have_vals: Union[int, List[int], None] = None,
    min_val: int = 0,
    max_val: int = 99999999,
    is_ragged: bool = True,
    device: str = "cpu",
    dtype: torch.dtype = torch.int64,
    seed: Optional[int] = None,
):
    table = set()
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    if must_have_vals is not None and not isinstance(must_have_vals, list):
        must_have_vals = [must_have_vals]
    if must_not_have_vals is not None and not isinstance(must_not_have_vals, list):
        must_not_have_vals = [must_not_have_vals]

    offsets = [0]
    all_flat_values = []

    for i in range(batch_size):
        row_len = random.randint(0, max_row_length) if is_ragged else max_row_length
        if row_len > 0:
            vals = []
            while len(vals) < row_len:
                new_val = random.randint(min_val, max_val)
                if must_not_have_vals is None or new_val not in must_not_have_vals:
                    vals.append(new_val)
                    if new_val not in table:
                        table.add(new_val)
            all_flat_values.extend(vals)

        offsets.append(offsets[-1] + row_len)

    if not all_flat_values:
        print("No values were generated. Consider adjusting parameters.")
        values = torch.empty(0, dtype=dtype, device=device)
        offsets_tensor = torch.tensor([0], dtype=torch.int32, device=device)
        return values, offsets_tensor

    if must_have_vals is not None:
        if len(must_have_vals) > len(all_flat_values):
            raise ValueError(
                f"The number of values to include [{len(must_have_vals)}], exceeds the total number of generated elements [{len(all_flat_values)}]."
            )
        for random_index, val_to_add in enumerate(must_have_vals):
            tmp = all_flat_values[random_index]
            table.discard(tmp)
            all_flat_values[random_index] = val_to_add
            table.add(val_to_add)

    values = torch.tensor(all_flat_values, dtype=dtype, device=device)
    offsets_tensor = torch.tensor(offsets, dtype=torch.int32, device=device)

    return RaggedTensor(values, offsets_tensor), table


def contains_all_values(my_tensor: torch.Tensor, values_to_check: torch.Tensor) -> bool:
    unique_main = torch.unique(my_tensor)
    unique_to_check = torch.unique(values_to_check)
    bool_mask = torch.isin(unique_to_check, unique_main)
    return torch.all(bool_mask).item()


def exclude_all_values(my_tensor: torch.Tensor, values_to_check: torch.Tensor) -> bool:
    isin_mask = torch.isin(my_tensor, values_to_check)
    found_any = torch.any(isin_mask)
    return not found_any


def check_rows_only(tensor, indices, x):
    mask = torch.zeros(tensor.size(0), dtype=torch.bool, device=tensor.device)
    mask[torch.as_tensor(indices, device=tensor.device)] = True
    rows = tensor[mask]
    x_tensor = torch.as_tensor(x, device=tensor.device)
    if x_tensor.dim() == 0:
        return torch.all(rows == x_tensor).item()
    else:
        return torch.all(rows == x_tensor.unsqueeze(0)).item()


def filter_child_ids(ht, child_index):
    ids, index = ht.ids_map()
    mask = (ids >> 52) == child_index
    mask_ids = ids[mask]
    mask_index = index[mask]
    return mask_ids, mask_index


def nested_dict():
    return defaultdict(nested_dict)


class HashTableClearTest(unittest.TestCase):
    DEVICE = None
    CKPT_DIR = None

    @classmethod
    def setUpClass(cls):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        dist.init_process_group()
        cls.DEVICE = os.getenv("TEST_DEVICE", "cuda")
        cls.CKPT_DIR = os.getenv("CKPT_DIR", "./clear_ckpt_cuda")
        os.makedirs(cls.CKPT_DIR, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()
        shutil.rmtree(cls.CKPT_DIR)

    def setIds(self):
        self.ids = {
            "step_0": {
                "fea_1": RaggedTensor(
                    torch.tensor(
                        [12, 13, 14, 15, 16], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int64, device="cuda"),
                ),
                "fea_2": RaggedTensor(
                    torch.tensor(
                        [27, 28, 29, 30, 31, 32, 33], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor([0, 1, 2, 3, 4, 7], dtype=torch.int64, device="cuda"),
                ),
                "fea_3": RaggedTensor(
                    torch.tensor([12, 99], dtype=torch.int64, device="cuda"),
                    torch.tensor([0, 1, 2, 2, 2, 2], dtype=torch.int64, device="cuda"),
                ),
            },
            "step_1": {
                "fea_1": RaggedTensor(
                    torch.tensor(
                        [99, 98, 97, 96, 95, 94], dtype=torch.int64, device="cuda"
                    ),
                    torch.tensor(
                        [0, 1, 2, 3, 4, 5, 6], dtype=torch.int64, device="cuda"
                    ),
                ),
                "fea_2": RaggedTensor(
                    torch.tensor([77, 28], dtype=torch.int64, device="cuda"),
                    torch.tensor(
                        [0, 1, 2, 2, 2, 2, 2], dtype=torch.int64, device="cuda"
                    ),
                ),
            },
        }

        for i in range(2, 4):
            rt_1, re_1 = gen_2d_ragged_tensor(
                batch_size=4096,
                max_row_length=20,
                is_ragged=False,
                device="cuda",
                min_val=100,
            )
            rt_2, re_2 = gen_2d_ragged_tensor(
                batch_size=4096,
                max_row_length=20,
                is_ragged=False,
                device="cuda",
                min_val=100,
            )
            rt_3, re_3 = gen_2d_ragged_tensor(
                batch_size=4096,
                max_row_length=20,
                is_ragged=False,
                device="cuda",
                min_val=100,
            )
            self.ids[f"step_{i}"] = {"fea_1": rt_1, "fea_2": rt_2, "fea_3": rt_3}

        fea_2_ids_list = [
            self.ids[f"step_{step}"]["fea_2"].values()
            for step in range(0, 4)
            if "fea_2" in self.ids[f"step_{step}"]
        ]
        fea_1_ids_list = [
            self.ids[f"step_{step}"]["fea_1"].values()
            for step in range(1, 4)
            if "fea_1" in self.ids[f"step_{step}"]
        ]
        fea_3_ids_list = [
            self.ids[f"step_{step}"]["fea_3"].values()
            for step in range(1, 4)
            if "fea_3" in self.ids[f"step_{step}"]
        ]
        ht1_ids_numel = torch.unique(torch.cat(fea_1_ids_list + fea_3_ids_list)).numel()
        ht2_ids_numel = torch.unique(torch.cat(fea_2_ids_list)).numel()
        self.mmax_ids_numel = (
            ht1_ids_numel + ht2_ids_numel
        )  # check index after insert and delete child
        mmax_ids_numel = self.mmax_ids_numel

        for i in range(4, 5):
            rt_1, re_1 = gen_2d_ragged_tensor(
                batch_size=4096,
                max_row_length=20,
                is_ragged=False,
                device="cuda",
            )
            mmax_ids_numel -= torch.unique(rt_1.values()).numel()
            tmp_val = list(range(100000000, 100000000 + mmax_ids_numel + 1))
            rt_4, re_4 = gen_2d_ragged_tensor(
                batch_size=mmax_ids_numel + 1,
                must_have_vals=tmp_val,
                max_row_length=1,
                is_ragged=False,
                device="cuda",
            )
            self.ids[f"step_{i}"] = {"fea_1": rt_1, "fea_4": rt_4}

    def setFeas(self):
        self.group_a = nested_dict()
        self.group_a["ht1"]["emb_opt"] = {
            "fea_1": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht1",
                combiner="sum",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            ),
            "fea_3": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht1",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            ),
        }
        self.group_a["ht2"]["emb_opt"] = {
            "fea_2": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht2",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            )
        }
        self.group_a["ht3"]["emb_opt"] = {
            "fea_4": EmbeddingOption(
                embedding_dim=8,
                shared_name="ht3",
                combiner="mean",
                initializer=ConstantInitializer(init_val=3.0),
                device=torch.device(self.DEVICE),
            )
        }

    def setEmbeddingEngine(self):
        from collections import ChainMap

        ee_init = dict(
            ChainMap(
                self.group_a["ht3"]["emb_opt"],
                self.group_a["ht2"]["emb_opt"],
                self.group_a["ht1"]["emb_opt"],
            )
        )
        self.ee = EmbeddingEngine(ee_init)

    def setHtInfo(self):
        self.tables = nested_dict()
        group_to_fea = {
            "group_a": "fea_1",
        }
        for group_name, fea_name in group_to_fea.items():
            ht_key = self.ee._fea_to_ht[fea_name]
            self.tables[group_name]["ht"] = self.ee._ht[ht_key]._hashtable
        for group_name in group_to_fea:
            ht = self.tables[group_name]["ht"]
            self.tables[group_name]["emb_slot"] = ht.slot_group().slot_by_name(
                "embedding"
            )
            self.tables[group_name]["sparse_adamw_tf_exp_avg_slot"] = (
                ht.slot_group().slot_by_name("sparse_adamw_tf_exp_avg")
            )
            self.tables[group_name]["sparse_adamw_tf_exp_avg_sq_slot"] = (
                ht.slot_group().slot_by_name("sparse_adamw_tf_exp_avg_sq")
            )

    def setEncodeIds(self):
        self.encode_ids = defaultdict(str)
        self.feas = ["fea_1", "fea_2", "fea_3", "fea_4"]
        for fea in self.feas:
            self.encode_ids[fea] = self.ee._fea_to_group[fea].encode_id(fea)

    def setUp(self):
        self.setIds()
        self.setFeas()
        self.setEmbeddingEngine()
        self.sparse_state, self.dense_state = split_sparse_dense_state_dict(
            self.ee.state_dict()
        )
        self.opt = SparseAdamWTF(self.sparse_state)
        self.setHtInfo()
        self.setEncodeIds()

    def checkDeleteSlot(
        self, ids, ids_to_encode, encode_id, group, child_index, init_val
    ):
        if ids is not None:
            pred_ids = fused_ops.fused_ids_encode_gpu([ids_to_encode], [encode_id])
            self.assertTrue(ts_equal(ids, pred_ids))
        self.assertTrue(
            check_rows_only(
                self.tables[group]["emb_slot"].value(), child_index, init_val
            )
        )
        self.assertTrue(
            check_rows_only(
                self.tables[group]["sparse_adamw_tf_exp_avg_slot"].value(),
                child_index,
                0.0,
            )
        )
        self.assertTrue(
            check_rows_only(
                self.tables[group]["sparse_adamw_tf_exp_avg_sq_slot"].value(),
                child_index,
                0.0,
            )
        )

    def testClear(self):
        hta = self.tables["group_a"]["ht"]
        ### init hashtable with some id
        for step in range(1):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()
            ids, mmap = self.tables["group_a"]["ht"].ids_map()
            pred_ids = fused_ops.fused_ids_encode_gpu(
                [
                    step_id["fea_1"].values(),
                    step_id["fea_2"].values(),
                    torch.tensor([99], dtype=torch.int64, device="cuda"),
                ],
                [
                    self.encode_ids["fea_1"],
                    self.encode_ids["fea_2"],
                    self.encode_ids["fea_3"],
                ],
            )
            self.assertTrue(
                ts_equal(
                    ids,
                    pred_ids,
                )
            )
            self.assertTrue(mmap.max().item() == 12)

        child_ids, child_index = filter_child_ids(hta, self.encode_ids["fea_1"])
        hta.clear("ht1")
        ids, _ = self.tables["group_a"]["ht"].ids_map()
        self.checkDeleteSlot(
            ids,
            step_id["fea_2"].values(),
            self.encode_ids["fea_2"],
            "group_a",
            child_index,
            3.0,
        )
        for step in range(1, 2):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()
            # check existing key after insert
            fea_2_ids = torch.unique(
                torch.cat(
                    (step_id["fea_2"].values(), self.ids["step_0"]["fea_2"].values())
                )
            )
            ids, mmap = self.tables["group_a"]["ht"].ids_map()
            pred_ids = fused_ops.fused_ids_encode_gpu(
                [step_id["fea_1"].values(), fea_2_ids],
                [
                    self.encode_ids["fea_1"],
                    self.encode_ids["fea_2"],
                ],
            )
            self.assertTrue(
                ts_equal(
                    ids,
                    pred_ids,
                )
            )
            self.assertTrue(mmap.max().item() == 13)
        fea_2_ids_list = [
            self.ids[f"step_{step}"]["fea_2"].values()
            for step in range(0, 4)
            if "fea_2" in self.ids[f"step_{step}"]
        ]
        fea_2_ids = torch.unique(torch.cat(fea_2_ids_list))
        for step in range(2, 4):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()

        child_ids, child_index = filter_child_ids(hta, self.encode_ids["fea_1"])
        hta.clear("ht1")
        ids, _ = self.tables["group_a"]["ht"].ids_map()
        self.checkDeleteSlot(
            ids, fea_2_ids, self.encode_ids["fea_2"], "group_a", child_index, 3.0
        )
        child_ids, child_index = filter_child_ids(hta, self.encode_ids["fea_2"])
        hta.clear("ht2")
        self.checkDeleteSlot(
            None, None, self.encode_ids["fea_2"], "group_a", child_index, 3.0
        )

        # check empty
        ids, mmap = self.tables["group_a"]["ht"].ids_map()
        self.assertTrue(len(ids) == 0 and len(mmap) == 0)
        # clear empty child. nothing happened
        hta.clear("ht2")
        for step in range(4, 5):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_4"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()

        _, index_map = self.tables["group_a"]["ht"].ids_map()
        self.assertTrue(index_map.max().item() == self.mmax_ids_numel + 1 - 1)
        fea_1_ids = torch.unique(torch.cat([self.ids["step_4"]["fea_1"].values()]))

        child_ids, child_index = filter_child_ids(hta, self.encode_ids["fea_4"])
        hta.clear("ht3")
        ids, _ = self.tables["group_a"]["ht"].ids_map()
        self.checkDeleteSlot(
            ids, fea_1_ids, self.encode_ids["fea_1"], "group_a", child_index, 3.0
        )
        # now, there is only ht1 in the coalseced hashtable

        ### test save after clear ###
        saver = SSaver(
            shard_index=0,
            shard_num=1,
            parallel=1,
            hashtables=self.sparse_state,
            tensors=self.dense_state,
            path=self.CKPT_DIR,
        )
        saver.save()

        cr = CheckpointReader(self.CKPT_DIR)
        for i in range(1, 4):
            ids = cr.read_tensor(f"ht{i}@id")
            emb = cr.read_tensor(f"ht{i}@embedding")
            exp_avg = cr.read_tensor(f"ht{i}@sparse_adamw_tf_exp_avg")
            exp_avg_sq = cr.read_tensor(f"ht{i}@sparse_adamw_tf_exp_avg_sq")
            if i != 1:
                self.assertTrue(
                    len(ids) == 0
                    and len(emb) == 0
                    and len(exp_avg) == 0
                    and len(exp_avg_sq) == 0
                )

        ### test clear all
        hta.clear()
        act_sz, sz = hta.allocator_id_info()
        self.assertTrue(act_sz == 0)
        self.assertTrue(sz != 0)
        raw_emb = hta.raw_embeddings()
        self.assertTrue(torch.all(raw_emb == 3.0).item())

        ### test reset
        for step in range(4):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()
        hta.reset()
        raw_emb = hta.raw_embeddings()
        act_sz, sz = hta.allocator_id_info()
        self.assertTrue(act_sz == 0)
        self.assertTrue(sz == 0)
        self.assertTrue(torch.all(raw_emb == 3.0).item())
        ### test insert after reset
        for step in range(4):
            step_id = self.ids[f"step_{step}"]
            out = self.ee(step_id)
            (out["fea_1"].sum() + out["fea_2"].sum()).backward()
            self.opt.step()
            self.opt.zero_grad()
        hta.reset()
        saver.save()
        ### test save after reset
        cr = CheckpointReader(self.CKPT_DIR)
        for i in range(1, 4):
            ids = cr.read_tensor(f"ht{i}@id")
            emb = cr.read_tensor(f"ht{i}@embedding")
            self.assertTrue(len(emb) == 0 and len(ids) == 0)


if __name__ == "__main__":
    unittest.main()
