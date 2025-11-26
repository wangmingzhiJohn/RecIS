import os
import shutil
import tempfile
import unittest

import torch
import torch.nn as nn

from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions
from recis.framework.filesystem import get_file_system
from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param, gen_slice
from recis.optim.sparse_adamw_tf import SparseAdamWTF
from recis.serialize.checkpoint_reader import CheckpointReader
from recis.utils.logger import Logger


logger = Logger(__name__)


class Model(torch.nn.Module):
    def __init__(self, shard_idx=0, shard_num=1):
        super().__init__()
        self.shard_idx = shard_idx
        self.shard_num = shard_num
        self.table_1 = HashTable(
            [1024], name="table_1", slice=gen_slice(shard_idx, shard_num)
        )
        self.table_2 = HashTable(
            [1024], name="table_2", slice=gen_slice(shard_idx, shard_num)
        )

        self.dense1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )
        self.dense2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        return self.dense1(self.table_1(x) + self.table_2(x)) + self.dense2(
            self.table_2(x)
        )


class TestModelBank(unittest.TestCase):
    def setUp(self):
        # 创建临时目录
        self.tmpdir = tempfile.mkdtemp()

        self.model = Model(shard_idx=0, shard_num=1)
        self.model = self.model.to("cuda")
        self.sparse_param = filter_out_sparse_param(self.model)
        self.sparse_optim = SparseAdamWTF(param_dict=self.sparse_param, lr=0.001)
        self.dense_optim = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        print("in init", self.tmpdir)

        saver_option = SaverOptions(
            self.model,
            self.sparse_optim,
            self.tmpdir,
            None,
            20,
            1,
            None,
        )
        self.epoch = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        self.global_step = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        self.saver = Saver(saver_option)
        self.saver.register_for_checkpointing(
            ExtraFields.recis_dense_optim, self.dense_optim
        )
        self.saver.register_for_checkpointing(ExtraFields.train_epoch, self.epoch)
        self.saver.register_for_checkpointing(ExtraFields.global_step, self.global_step)
        self._save_model()

    def tearDown(self):
        # 清理临时目录
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _save_model(self):
        for _ in range(10):
            self.epoch.add_(1)
            self.global_step.add_(1)

            self.sparse_optim.zero_grad()
            self.dense_optim.zero_grad()
            ids = torch.arange(100)
            emb = self.model(ids)
            loss = torch.sum(emb)
            loss.backward()
            self.sparse_optim.step()
            self.dense_optim.step()
            self.saver.save(ckpt_id=f"ckpt_{self.global_step.item()}")

    def _check_dense_oname(self, path):
        pt_file = os.path.join(path, "model.pt")
        fs = get_file_system(pt_file)
        with fs.open(pt_file, "rb") as f:
            state_dict = torch.load(f, weights_only=False)
        for name, param in self.model.named_parameters():
            if name.startswith("dense"):
                oname = ""
                if "dense1" in name:
                    oname = name.replace("dense1", "dense2")
                else:
                    oname = name.replace("dense2", "dense1")
                self.assertTrue(torch.allclose(state_dict[oname], param))

    def _check_dense_optim(self, path):
        fs = get_file_system(os.path.join(path, "extra.pt"))
        with fs.open(os.path.join(path, "extra.pt"), "rb") as f:
            extra_data = torch.load(f, weights_only=False)
        tmp_state_dict = self.dense_optim.state_dict()

        optim_key = (
            ExtraFields.recis_dense_optim
            if ExtraFields.recis_dense_optim in extra_data
            else ExtraFields.prev_optim
        )

        for cnt, value in enumerate(extra_data[optim_key]["state"].values()):
            for key, val in value.items():
                self.assertTrue(
                    torch.allclose(
                        val,
                        tmp_state_dict["state"][cnt][key],
                    )
                )

        self.assertEqual(
            tmp_state_dict["param_groups"],
            extra_data[optim_key]["param_groups"],
        )

    def _check_sparse_optim(self, path):
        reader = CheckpointReader(path)
        tensor_names = [
            "sparse_adamw_tf_beta2",
            "sparse_adamw_tf_beta1",
            "sparse_adamw_tf_step",
        ]
        for name in tensor_names:
            self.assertEqual(
                reader.read_tensor(name),
                self.sparse_optim.state_dict()[name],
            )

    def _check_sparse_oname(self, path, src_tables=None, dst_tables=None):
        if src_tables is None:
            src_tables = []
        if dst_tables is None:
            dst_tables = []
        reader = CheckpointReader(path)
        for src_table, dst_table in zip(src_tables, dst_tables):
            tmp_model = getattr(self.model, dst_table)
            _, index = tmp_model.ids_map()
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{src_table}@id"),
                    tmp_model.ids(),
                )
            )
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{src_table}@embedding"),
                    tmp_model.embeddings()[index],
                )
            )
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{src_table}@sparse_adamw_tf_exp_avg"),
                    tmp_model._hashtable_impl.slot_group()
                    .slot_by_name("sparse_adamw_tf_exp_avg")
                    .value()[index],
                )
            )
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{src_table}@sparse_adamw_tf_exp_avg_sq"),
                    tmp_model._hashtable_impl.slot_group()
                    .slot_by_name("sparse_adamw_tf_exp_avg_sq")
                    .value()[index],
                )
            )

    def _check_sparse_model(
        self,
        path,
        slot_names=(
            "table_1",
            "table_2",
        ),
        table_names=("table_1", "table_2"),
    ):
        reader = CheckpointReader(path)
        for table_name in table_names:
            tmp_model = getattr(self.model, table_name)
            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{table_name}@id"),
                    tmp_model.ids(),
                )
            )

            _, index = tmp_model.ids_map()

            self.assertTrue(
                torch.allclose(
                    reader.read_tensor(f"{table_name}@embedding"),
                    tmp_model.embeddings()[index],
                )
            )

            for slot_name in slot_names:
                if slot_name != table_name:
                    continue
                self.assertTrue(
                    torch.allclose(
                        reader.read_tensor(f"{slot_name}@sparse_adamw_tf_exp_avg"),
                        tmp_model._hashtable_impl.slot_group()
                        .slot_by_name("sparse_adamw_tf_exp_avg")
                        .value()[index],
                    )
                )
                self.assertTrue(
                    torch.allclose(
                        reader.read_tensor(f"{slot_name}@sparse_adamw_tf_exp_avg_sq"),
                        tmp_model._hashtable_impl.slot_group()
                        .slot_by_name("sparse_adamw_tf_exp_avg_sq")
                        .value()[index],
                    )
                )

    def _check_dense_model(self, path):
        pt_file = os.path.join(path, "model.pt")
        fs = get_file_system(pt_file)
        with fs.open(pt_file, "rb") as f:
            state_dict = torch.load(f, weights_only=False)
        for name, param in self.model.named_parameters():
            if name.startswith("dense"):
                self.assertTrue(torch.allclose(state_dict[name], param))

    def test_model_bank(self):
        # 使用 self.tmpdir 替代硬编码的路径
        ckpt_1 = os.path.join(self.tmpdir, "ckpt_1")
        ckpt_2 = os.path.join(self.tmpdir, "ckpt_2")
        ckpt_3 = os.path.join(self.tmpdir, "ckpt_3")
        ckpt_4 = os.path.join(self.tmpdir, "ckpt_4")
        ckpt_6 = os.path.join(self.tmpdir, "ckpt_6")
        ckpt_10 = os.path.join(self.tmpdir, "ckpt_10")

        self.saver.output_dir = ""

        # Test 1: Load all
        simple_bank = [
            {
                "path": ckpt_1,
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        self.saver._init_model_bank(simple_bank)
        self.saver.restore()
        self._check_sparse_model(ckpt_1)
        self._check_dense_model(ckpt_1)
        self._check_dense_optim(ckpt_1)
        self._check_sparse_optim(ckpt_1)

        # Test 2: Load only sparse
        sparse_bank = [
            {
                "path": ckpt_2,
                "load": ["table*"],
            }
        ]
        self.saver._init_model_bank(sparse_bank)
        self.saver.restore()
        self._check_sparse_model(ckpt_2)

        # Test 3: Load only dense
        dense_bank = [
            {
                "path": ckpt_4,
                "load": ["table_2*"],
            },
            {
                "path": ckpt_3,
                "load": ["dense*"],
            },
            {
                "path": ckpt_4,
                "load": ["table_1*"],
            },
        ]
        self.saver._init_model_bank(dense_bank)
        self.saver.restore()
        self._check_dense_model(ckpt_3)
        self._check_sparse_model(
            ckpt_4,
            slot_names=["table_1", "table_2"],
            table_names=["table_2", "table_1"],
        )

        # Test 5: Split load from different checkpoints
        split_bank = [
            {
                "path": ckpt_3,
                "load": ["*"],
                "exclude": ["table_1@*avg*", "io_state"],
            },
            {
                "path": ckpt_4,
                "load": ["table_1*"],
                "exclude": ["table_1@*avg*", "io_state"],
                "is_dynamic": False,
                "hashtable_clear": True,
            },
            {
                "path": ckpt_4,
                "load": ["table_2*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
                "hashtable_clear": True,
            },
            {
                "path": ckpt_6,
                "load": ["dense*"],
                "exclude": ["io_state"],
            },
        ]
        self.saver._init_model_bank(split_bank)
        self.saver.restore()
        self._check_dense_model(ckpt_6)
        self._check_sparse_model(
            ckpt_4, slot_names=["table_2"], table_names=["table_2"]
        )
        self._check_sparse_model(
            ckpt_4, slot_names=["table_none"], table_names=["table_1"]
        )
        self._check_sparse_optim(ckpt_3)
        self._check_dense_optim(ckpt_3)

        # Test 6: oname test
        oname_bank = [
            {
                "path": ckpt_10,
                "load": ["table_1*", "table_2*", "dense*"],
                "exclude": ["io_state"],
                "oname": [
                    {"table_1*": "table_2*"},
                    {"table_2*": "table_1*"},
                    {"dense1*": "dense2*"},
                    {"dense2*": "dense1*"},
                ],
            }
        ]
        self.saver._init_model_bank(oname_bank)
        self.saver.restore()
        self._check_sparse_oname(
            ckpt_10,
            src_tables=["table_1", "table_2"],
            dst_tables=["table_2", "table_1"],
        )
        self._check_dense_oname(ckpt_10)

        # Test 7: is dynamic test
        dynamic_bank = [
            {
                "path": ckpt_10,
                "load": ["table*"],
                "exclude": ["io_state"],
                "is_dynamic": True,
                "ignore_error": True,
            },
        ]
        self.saver._init_model_bank(dynamic_bank)
        self.saver.restore()
        self._check_sparse_model(ckpt_10)


if __name__ == "__main__":
    unittest.main()
