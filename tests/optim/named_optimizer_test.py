import os
import shutil
import tempfile
import unittest
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from packaging import version
from torch.nn import MSELoss

from recis.framework.checkpoint_manager import ExtraFields, Saver, SaverOptions
from recis.framework.filesystem import get_file_system
from recis.nn.modules.hashtable import HashTable, filter_out_sparse_param, gen_slice
from recis.optim.named_optimizer import wrapped_named_optimizer
from recis.optim.sparse_adamw_tf import SparseAdamWTF


class MyModel1(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 4),
            nn.Linear(4, 2),
        )
        self.classifier1 = nn.Parameter(torch.randn(2, 9), requires_grad=True)
        self.bias1 = nn.Parameter(torch.randn(9), requires_grad=True)
        self.classifier2 = nn.Parameter(torch.randn(9, 1), requires_grad=True)
        self.bias2 = nn.Parameter(torch.randn(1), requires_grad=True)

    def forward(self, x):
        # x: (B, 10)
        feat = self.feature_extractor(x)  # (B, 2)
        hidden = torch.relu(
            feat @ self.classifier1 + self.bias1
        )  # (B, 2) @ (2, 9) -> (B, 9)
        output = hidden @ self.classifier2 + self.bias2
        return output


class MiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2, 9),
            nn.Linear(9, 1),
        )

    def forward(self, x):
        x = x[:, :2]
        return self.classifier(x)


class NogradMiniModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 4),
            nn.Linear(4, 2),
        )  # 特征提取层
        self.classifier = nn.Sequential(
            nn.Linear(2, 9),
            nn.Linear(9, 1),
        )

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return self.classifier(torch.relu(x))


class HugeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor_add = nn.Sequential(
            nn.Linear(2, 4),
            nn.Linear(4, 2),
        )  # 特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 4),
            nn.Linear(4, 2),
        )  # 特征提取层
        self.classifier = nn.Sequential(
            nn.Linear(2, 9),
            nn.Linear(9, 1),
        )
        self.classifier_add = nn.Sequential(
            nn.Linear(2, 1),
        )

    def forward(self, x):
        feat = self.feature_extractor(x)
        feat_add = self.feature_extractor_add(feat)
        return self.classifier(feat) + self.classifier_add(feat_add)


class MyModel2(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(10, 4),
            nn.Linear(4, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(2, 9),
            nn.Linear(9, 1),
        )

    def forward(self, x):
        return self.classifier(torch.relu(self.feature_extractor(x)))


class Trainer(nn.Module):
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


class Test(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self) -> None:
        if hasattr(self, "tmpdir") and os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)

    def _model_run(self, model, optimizer):
        optimizer.zero_grad()
        dummy_input = torch.randn(10, 10)
        loss = MSELoss()(model(dummy_input), torch.randn(10, 1))
        loss.backward()
        optimizer.step()

    def _build_model_name_idx_map(self, model, optimizer):
        name_to_index = {}
        model_dict = {p: n for n, p in model.named_parameters()}
        layer_idx = 0
        for group in optimizer.param_groups:
            for param in group["params"]:
                name_to_index[model_dict[param]] = layer_idx
                layer_idx += 1

        return name_to_index

    def _get_valid_names(self, model1, model2):
        valid_names = set()
        for name, param in model1.named_parameters():
            if name in model2.state_dict().keys():
                if param.shape == model2.state_dict()[name].shape:
                    valid_names.add(name)
        return valid_names

    def _convert_valid_names(self, valid_names, model, optimizer):
        """
        convert valid model names to optimizer param names
        """
        model_dict = dict(model.named_parameters())
        optim_dict = {}
        for group in optimizer.param_groups:
            optim_dict.update(dict(zip(group["params"], group["param_names"])))

        res = set()
        for name in valid_names:
            res.add(optim_dict[model_dict[name]])
        return res

    def _check_optim_by_extra(self, optimizer, path):
        fs = get_file_system(os.path.join(path, "extra.pt"))
        with fs.open(os.path.join(path, "extra.pt"), "rb") as f:
            extra_data = torch.load(f, weights_only=False)
        tmp_state_dict = optimizer.state_dict()

        optim_key = (
            ExtraFields.recis_dense_optim
            if ExtraFields.recis_dense_optim in extra_data
            else ExtraFields.prev_optim
        )

        for top_key, value in extra_data[optim_key]["state"].items():
            for key, val in value.items():
                self.assertTrue(
                    torch.allclose(
                        val,
                        tmp_state_dict["state"][top_key][key],
                    )
                )

        self.assertEqual(
            tmp_state_dict["param_groups"],
            extra_data[optim_key]["param_groups"],
        )

    def _check_optim_basic(
        self,
        test_optimizer,
        path,
        is_wrapped: bool = False,
        ignore_param_names: bool = True,
        load_idx_map: Optional[dict[int, int]] = None,
    ):
        bench_optimizer = torch.load(path, weights_only=True)
        state_dict = test_optimizer.state_dict(origin=is_wrapped)

        # check param groups
        if len(bench_optimizer["param_groups"]) != len(test_optimizer.param_groups):
            return False

        # check super param
        for g1, g2 in zip(bench_optimizer["param_groups"], test_optimizer.param_groups):
            key1 = {k for k in g1.keys() if k != "params"}
            key2 = {k for k in g2.keys() if k != "params"}
            if ignore_param_names:
                key1.discard("param_names")
                key2.discard("param_names")

            if key1 != key2:
                return False
            for k in key1:
                if g1[k] != g2[k]:
                    return False

        # check state
        if set(bench_optimizer["state"].keys()) != set(state_dict["state"].keys()):
            return False

        # check param
        for k in bench_optimizer["state"].keys():
            for key, value in bench_optimizer["state"][k].items():
                if key not in state_dict["state"][k]:
                    return False

                if load_idx_map is None:
                    if not torch.allclose(state_dict["state"][k][key], value):
                        return False
                else:
                    if not torch.allclose(
                        state_dict["state"][load_idx_map[k]][key], value
                    ):
                        return False
        return True

    def _trainer_run(self, trainer, sparse_optim, dense_optim, model_bank):
        saver_option = SaverOptions(
            trainer,
            sparse_optim,
            self.tmpdir,
            None,
            20,
            1,
            None,
        )
        epoch = torch.scalar_tensor(0, dtype=torch.int64).cuda()
        global_step = torch.scalar_tensor(0, dtype=torch.int64).cuda()

        saver = Saver(saver_option)
        saver.register_for_checkpointing(ExtraFields.recis_dense_optim, dense_optim)
        saver.register_for_checkpointing(ExtraFields.train_epoch, epoch)
        saver.register_for_checkpointing(ExtraFields.global_step, global_step)

        def run():
            for _ in range(2):
                epoch.add_(1)
                global_step.add_(1)
                sparse_optim.zero_grad()
                dense_optim.zero_grad()
                ids = torch.arange(100)
                emb = trainer(ids)
                loss = torch.sum(emb)
                loss.backward()
                sparse_optim.step()
                dense_optim.step()
                saver.save(ckpt_id=f"ckpt_{global_step.item()}")

        run()
        saver.output_dir = ""
        ckpt_1 = os.path.join(self.tmpdir, "ckpt_1")
        saver._init_model_bank(model_bank)
        saver.restore()
        self._check_optim_by_extra(dense_optim, ckpt_1)
        saver.output_dir = self.tmpdir
        run()

        for i in range(1, 3):
            if os.path.exists(os.path.join(self.tmpdir, f"ckpt_{i}")):
                shutil.rmtree(os.path.join(self.tmpdir, f"ckpt_{i}"))

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_trainer(self):
        trainer = Trainer(shard_idx=0, shard_num=1).to("cuda")
        sparse_param = filter_out_sparse_param(trainer)
        sparse_optim = SparseAdamWTF(sparse_param, lr=0.001)
        dense_optim = torch.optim.AdamW(trainer.named_parameters(), lr=0.001)
        ckpt_1 = os.path.join(self.tmpdir, "ckpt_1")
        simple_bank = [
            {
                "path": ckpt_1,
                "load": ["*"],
                "exclude": ["io_state"],
                "is_dynamic": False,
            }
        ]
        self._trainer_run(trainer, sparse_optim, dense_optim, simple_bank)

        dense_optim = torch.optim.AdamW(trainer.parameters(), lr=0.001)
        self._trainer_run(trainer, sparse_optim, dense_optim, simple_bank)

        dense_optim = wrapped_named_optimizer(torch.optim.AdamW)(
            trainer.named_parameters(), lr=0.001
        )
        self._trainer_run(trainer, sparse_optim, dense_optim, simple_bank)

    def test_huge_model_no_warp(self):
        model = MyModel2()
        optimizer = optim.AdamW(model.parameters())
        self._model_run(model, optimizer)

        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))
        del optimizer, model
        model = HugeModel()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "feature_extractor.0.weight",
                "feature_extractor.0.bias",
                "feature_extractor.1.weight",
                "feature_extractor.1.bias",
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        with self.assertRaises(RuntimeError) as cm:
            self._model_run(model, new_optimizer)
        self.assertIn("must match the size of tensor", str(cm.exception))

    def test_huge_model(self):
        model = MyModel2()
        optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        self._model_run(model, optimizer)

        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))
        del optimizer, model
        model = HugeModel()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "feature_extractor.0.weight",
                "feature_extractor.0.bias",
                "feature_extractor.1.weight",
                "feature_extractor.1.bias",
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        self._model_run(model, new_optimizer)

    def test_load_nograd_model_no_warp(self):
        model = NogradMiniModel()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=0.0001,
            betas=(0.99999, 0.99999),
            weight_decay=0.0,
            eps=1e-9,
            amsgrad=False,
        )
        self._model_run(model, optimizer)
        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))
        del optimizer, model
        model = MyModel2()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(
            model.named_parameters(), lr=0.1
        )
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "feature_extractor.0.weight",
                "feature_extractor.0.bias",
                "feature_extractor.1.weight",
                "feature_extractor.1.bias",
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        self._model_run(model, new_optimizer)

    def test_load_nograd_model(self):
        model = NogradMiniModel()
        optimizer = wrapped_named_optimizer(optim.AdamW)(
            model.named_parameters(),
            lr=0.0001,
            betas=(0.99999, 0.99999),
            weight_decay=0.0,
            eps=1e-9,
            amsgrad=False,
        )
        self._model_run(model, optimizer)
        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))
        del optimizer, model
        model = MyModel2()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(
            model.named_parameters(), lr=0.1
        )
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "feature_extractor.0.weight",
                "feature_extractor.0.bias",
                "feature_extractor.1.weight",
                "feature_extractor.1.bias",
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        self._model_run(model, new_optimizer)

    def test_nograd_model_load_no_warp(self):
        model = MyModel2()
        optimizer = optim.AdamW(model.parameters())
        self._model_run(model, optimizer)

        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))
        del optimizer, model
        model = NogradMiniModel()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "feature_extractor.0.weight",
                "feature_extractor.0.bias",
                "feature_extractor.1.weight",
                "feature_extractor.1.bias",
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        self._model_run(model, new_optimizer)

    def test_nograd_model_load(self):
        model = MyModel2()
        optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        self._model_run(model, optimizer)

        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))
        del optimizer, model
        model = NogradMiniModel()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "feature_extractor.0.weight",
                "feature_extractor.0.bias",
                "feature_extractor.1.weight",
                "feature_extractor.1.bias",
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        self._model_run(model, new_optimizer)

    def test_mini_model_no_warp(self):
        model = MyModel2()
        optimizer = optim.AdamW(model.parameters())
        self._model_run(model, optimizer)

        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))

        del optimizer, model
        model = MiniModel()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        with self.assertRaises(RuntimeError) as cm:
            self._model_run(model, new_optimizer)
        self.assertIn("must match the size of tensor", str(cm.exception))

    def test_mini_model(self):
        model = MyModel2()
        optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        self._model_run(model, optimizer)

        torch.save(optimizer.state_dict(), os.path.join(self.tmpdir, "optimizer.pth"))

        del optimizer, model
        model = MiniModel()
        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer.pth")),
            valid_names={
                "classifier.0.weight",
                "classifier.0.bias",
                "classifier.1.weight",
                "classifier.1.bias",
            },
        )
        self._model_run(model, new_optimizer)

    def test_wrapped_load_map(self):
        model2 = MyModel2()
        optimizer2 = wrapped_named_optimizer(optim.AdamW)(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        model1 = MyModel1()

        optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())
        valid_names = {"classifier1", "bias1", "classifier2", "bias2"}

        for name in model1.state_dict().keys():
            if name in model2.state_dict().keys():
                valid_names.add(name)

        optimizer1.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer2.pth")),
            valid_names=self._convert_valid_names(valid_names, model1, optimizer1),
            load_map={
                "classifier1": "classifier.0.weight",
                "bias1": "classifier.0.bias",
                "classifier2": "classifier.1.weight",
                "bias2": "classifier.1.bias",
            },
        )

    def test_basic(self):
        model2 = MyModel2()
        optimizer2 = wrapped_named_optimizer(optim.AdamW)(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model2.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True),
            valid_names=self._convert_valid_names(
                model2.state_dict().keys(), model2, new_optimizer
            ),
        )
        self.assertTrue(
            self._check_optim_basic(
                new_optimizer, os.path.join(self.tmpdir, "optimizer2.pth")
            )
        )
        self._model_run(model2, new_optimizer)

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_load_origin_optim_with_named(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model2.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True),
            valid_names=self._convert_valid_names(
                model2.state_dict().keys(), model2, new_optimizer
            ),
        )
        self.assertTrue(
            self._check_optim_basic(
                new_optimizer,
                os.path.join(self.tmpdir, "optimizer2.pth"),
                is_wrapped=True,
                ignore_param_names=True,
            )
        )
        self._model_run(model2, new_optimizer)

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_load_origin_optim_without_named(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        new_optimizer = wrapped_named_optimizer(optim.AdamW)(model2.named_parameters())
        new_optimizer.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True),
            valid_names=self._convert_valid_names(
                model2.state_dict().keys(), model2, new_optimizer
            ),
        )
        self.assertTrue(
            self._check_optim_basic(
                new_optimizer,
                os.path.join(self.tmpdir, "optimizer2.pth"),
                is_wrapped=True,
                ignore_param_names=True,
            )
        )
        self._model_run(model2, new_optimizer)

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_diff_model(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        model1 = MyModel1()
        optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())
        optimizer1.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True),
            valid_names=self._convert_valid_names(
                self._get_valid_names(model1, model2), model1, optimizer1
            ),
        )
        self._model_run(model1, optimizer1)

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_param_model(self):
        model1 = MyModel1()
        optimizer1 = optim.AdamW(model1.named_parameters())
        self._model_run(model1, optimizer1)

        torch.save(optimizer1.state_dict(), os.path.join(self.tmpdir, "optimizer1.pth"))
        # del optimizer1

        optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())
        optimizer1.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer1.pth"), weights_only=True),
            valid_names=self._convert_valid_names(
                model1.state_dict().keys(), model1, optimizer1
            ),
        )
        self.assertTrue(
            self._check_optim_basic(
                optimizer1,
                os.path.join(self.tmpdir, "optimizer1.pth"),
                is_wrapped=True,
                ignore_param_names=True,
            )
        )
        self._model_run(model1, optimizer1)

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_load_empty(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        model1 = MyModel1()
        optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())
        optimizer1.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True),
            valid_names=set(),
        )
        self._model_run(model1, optimizer1)

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_load_map(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        model1 = MyModel1()

        optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())
        valid_names = {"classifier1", "bias1", "classifier2", "bias2"}

        for name in model1.state_dict().keys():
            if name in model2.state_dict().keys():
                valid_names.add(name)

        optimizer1.load_state_dict(
            torch.load(os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True),
            valid_names=self._convert_valid_names(valid_names, model1, optimizer1),
            load_map={
                "classifier1": "classifier.0.weight",
                "bias1": "classifier.0.bias",
                "classifier2": "classifier.1.weight",
                "bias2": "classifier.1.bias",
            },
        )

        self.assertTrue(
            self._check_optim_basic(
                optimizer1,
                os.path.join(self.tmpdir, "optimizer2.pth"),
                is_wrapped=True,
                ignore_param_names=True,
                load_idx_map={
                    0: 4,
                    1: 5,
                    2: 6,
                    3: 7,
                    4: 0,
                    5: 1,
                    6: 2,
                    7: 3,
                },
            )
        )

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_bad_load_map_in(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        with self.assertRaises(RuntimeError) as cm:
            model1 = MyModel1()
            optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())

            optimizer1.load_state_dict(
                torch.load(
                    os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True
                ),
                valid_names=self._convert_valid_names(
                    {"classifier1", "bias1", "classifier2", "bias2"}, model1, optimizer1
                ),
                load_map={
                    "classifier?": "classifier.0.weight",
                    "bias?": "classifier.0.bias",
                    "classifier2": "classifier.1.weight",
                    "bias2": "classifier.1.bias",
                },
            )
        self.assertIn("classifier? not in model.", str(cm.exception))

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_bad_load_map_out(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        with self.assertRaises(RuntimeError) as cm:
            model1 = MyModel1()
            optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())

            optimizer1.load_state_dict(
                torch.load(
                    os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True
                ),
                valid_names=self._convert_valid_names(
                    {"classifier1", "bias1", "classifier2", "bias2"}, model1, optimizer1
                ),
                load_map={
                    "classifier1": "classifier.2.weight",
                    "bias1": "classifier.2.bias",
                    "classifier2": "classifier.1.weight",
                    "bias2": "classifier.1.bias",
                },
            )
        self.assertIn("classifier.2.weight not in ckpt.", str(cm.exception))

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_bad_load_strict_false(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(model2.named_parameters())
        self._model_run(model2, optimizer2)

        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        model1 = MyModel1()
        optimizer1 = wrapped_named_optimizer(optim.AdamW)(model1.named_parameters())

        with self.assertRaises(RuntimeError) as cm:
            optimizer1.load_state_dict(
                torch.load(
                    os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True
                ),
                valid_names=self._convert_valid_names(
                    {"classifier1", "bias1", "classifier2", "bias2"}, model1, optimizer1
                ),
                strict=False,
                load_map={
                    "classifier1": "classifier.0.weight",
                    "bias1": "classifier.0.bias",
                    "classifier2": "classifier.1.weight",
                    "bias2": "classifier.1.bias",
                },
            )
            self._model_run(model1, optimizer1)
        self.assertIn("must match the size of tensor", str(cm.exception))

    @unittest.skipIf(
        version.parse(torch.__version__) < version.parse("2.6.0"),
        "Requires PyTorch >= 2.6.0",
    )
    def test_add_param_group(self):
        model2 = MyModel2()
        optimizer2 = optim.AdamW(
            [
                {
                    "params": model2.classifier.named_parameters(),
                    "lr": 0.01,
                }
            ]
        )
        optimizer2.add_param_group(
            {
                "params": model2.feature_extractor.named_parameters(),
                "lr": 0.15,
            }
        )
        self._model_run(model2, optimizer2)
        torch.save(optimizer2.state_dict(), os.path.join(self.tmpdir, "optimizer2.pth"))
        del optimizer2

        with self.assertRaises(RuntimeError) as cm:
            optimizer2 = wrapped_named_optimizer(optim.AdamW)(
                {
                    "params": model2.classifier.named_parameters(),
                    "lr": 0.01,
                }
            )
            optimizer2.add_param_group(
                {
                    "params": model2.feature_extractor.named_parameters(),
                    "lr": 0.15,
                }
            )
            optimizer2.load_state_dict(
                torch.load(
                    os.path.join(self.tmpdir, "optimizer2.pth"), weights_only=True
                ),
                valid_names=self._convert_valid_names(
                    model2.state_dict().keys(), model2, optimizer2
                ),
            )
        self.assertIn("found same name in param group", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
