import os
import unittest

import torch
import torch.distributed as dist
import torch.testing._internal.common_utils as common
from accelerate import Accelerator
from torch.nn import Linear
from torch.optim import AdamW

from recis.nn import initializers
from recis.nn.modules.embedding import DynamicEmbedding, EmbeddingOption
from recis.nn.modules.hashtable import filter_out_sparse_param
from recis.optim.sparse_adamw_tf import SparseAdamWTF


def sparse_allclose(sparse_A, sparse_B, atol=1e-6):
    """Compare two coalesced sparse tensors."""
    sparse_A = sparse_A.coalesce()
    sparse_B = sparse_B.coalesce()
    if sparse_A.shape != sparse_B.shape:
        return False
    if not torch.equal(sparse_A.indices(), sparse_B.indices()):
        return False
    if not torch.allclose(sparse_A.values(), sparse_B.values(), atol=atol):
        return False
    return True


def sparse_resize(sparse, size=None):
    if size is None:
        return sparse
    return torch.sparse_coo_tensor(sparse.indices(), sparse.values(), size).coalesce()


class Model(torch.nn.Module):
    def __init__(self, emb_dim=16, emb_name="emb") -> None:
        super().__init__()
        option = EmbeddingOption(
            embedding_dim=emb_dim,
            dtype=torch.float32,
            shared_name=emb_name,
            initializer=initializers.TruncNormalInitializer(std=0.02),
            combiner="mean",
            grad_reduce_by="worker",
        )
        self.emb = DynamicEmbedding(option)
        self.linear = Linear(16, 1)

    def forward(self, x):
        embs = self.emb(x)
        return (self.linear(embs) ** 2).sum() * 128.0


class AccGradAccumulatorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    @classmethod
    def tearDownClass(cls):
        dist.destroy_process_group()

    def setUp(self):
        self.emb_dim = 16
        self.ht_sz = 8
        self.ga_step = 2

    def test_grad_accumulator_with_accelerate(self) -> None:
        # grad accumulate by Accelerator
        accelerator = Accelerator(gradient_accumulation_steps=self.ga_step)
        model = Model(self.emb_dim)
        sparse_params = filter_out_sparse_param(model)
        dense_params = list(model.parameters())
        dense_opt = AdamW(dense_params, lr=1e-4)
        sparse_opt = SparseAdamWTF(sparse_params, lr=1e-4)
        model, dense_opt = accelerator.prepare(model, dense_opt)
        batch0 = (
            torch.tensor([1, 2, 3, 4, 5], dtype=torch.long)
            .unsqueeze(1)
            .to(accelerator.device)
        )
        batch1 = (
            torch.tensor([4, 5, 6, 7, 8], dtype=torch.long)
            .unsqueeze(1)
            .to(accelerator.device)
        )
        dense_opt.zero_grad()
        sparse_opt.zero_grad()
        with accelerator.accumulate(model):
            loss0 = model(batch0)
            accelerator.backward(loss0)
        dense_grad_acc = model.linear.weight.grad.detach().clone()
        sparse_grad_acc = model.emb._hashtable._hashtable_impl.grad(1).detach().clone()
        with accelerator.accumulate(model):
            loss1 = model(batch1)
            accelerator.backward(loss1)
        dense_grad_acc = model.linear.weight.grad.detach().clone()
        sparse_grad_acc = model.emb._hashtable._hashtable_impl.grad(1).detach().clone()

        # grad accumulate manually
        model_ref = model
        dense_opt_ref = AdamW(model_ref.parameters(), lr=1e-4)
        sparse_opt_ref = SparseAdamWTF(filter_out_sparse_param(model_ref), lr=1e-4)
        dense_opt_ref.zero_grad()
        sparse_opt_ref.zero_grad()
        loss0_ref = model_ref(batch0)
        loss0_ref.backward()
        dense_grad0 = model_ref.linear.weight.grad.detach().clone()
        sparse_grad0 = (
            model_ref.emb._hashtable._hashtable_impl.grad(1).detach().clone().coalesce()
        )

        dense_opt_ref.zero_grad()
        sparse_opt_ref.zero_grad()
        loss1_ref = model_ref(batch1)
        loss1_ref.backward()
        dense_grad1 = model_ref.linear.weight.grad.detach().clone()
        sparse_grad1 = (
            model_ref.emb._hashtable._hashtable_impl.grad(1).detach().clone().coalesce()
        )

        sparse_grad0 = sparse_resize(sparse_grad0, (self.ht_sz, self.emb_dim))
        sparse_grad1 = sparse_resize(sparse_grad1, (self.ht_sz, self.emb_dim))
        expected_dense_grad = (dense_grad0 + dense_grad1) / self.ga_step
        expected_sparse_grad = (sparse_grad0 + sparse_grad1).coalesce() / self.ga_step

        dense_diff = (dense_grad_acc - expected_dense_grad).abs().max().item()
        self.assertLessEqual(dense_diff, 1e-6)
        self.assertTrue(sparse_allclose(sparse_grad_acc, expected_sparse_grad))


if __name__ == "__main__":
    unittest.main()
