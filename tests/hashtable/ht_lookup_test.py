import os
import unittest

import torch

from recis.nn.hashtable_hook import AdmitHook
from recis.nn.modules.hashtable import HashTable, split_sparse_dense_state_dict
from recis.optim import SparseAdam


class GPUHashtableTest(unittest.TestCase):
    DEVICE = None

    @classmethod
    def setUpClass(cls):
        cls.DEVICE = os.getenv("TEST_DEVICE", "cuda")

    def setUp(self):
        self.ids_num = 2048
        self.emb_dim = 128
        self.block_size = 1024
        self.dtype = torch.float32

    def test_embedding_lookup(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device(self.DEVICE),
            name="gpu_ht",
        )

        # init hashtable
        ids = torch.arange(self.ids_num, device=self.DEVICE)
        emb_r = ht(ids)
        exp_r = torch.zeros_like(emb_r)
        self.assertTrue((exp_r.cuda() == emb_r).all())

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device=self.DEVICE)
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        emb_r = ht(ids)
        exp_r = emb
        self.assertTrue((exp_r.cuda() == emb_r).all())

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device=self.DEVICE
        )
        emb_r = ht(ids)
        exp_r = torch.concat([emb, torch.zeros_like(emb)], 0)
        self.assertTrue((exp_r.cuda() == emb_r).all())

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(ids_beg + self.ids_num, device=self.DEVICE)
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_readonly(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device(self.DEVICE),
            name="gpu_ht_ro",
        )
        sparse_state, _ = split_sparse_dense_state_dict(ht.state_dict())
        opt = SparseAdam(sparse_state)

        ro_hook = AdmitHook("ReadOnly")
        # init hashtable
        ids = torch.arange(self.ids_num, device=self.DEVICE)
        ht(ids, admit_hook=ro_hook)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device=self.DEVICE)
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        emb = ht(ids, admit_hook=ro_hook)
        emb.sum().backward()
        opt.step()
        opt.zero_grad()

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device=self.DEVICE
        )
        emb = ht(ids, admit_hook=ro_hook)
        emb.sum().backward()
        opt.step()
        opt.zero_grad()

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num, device=self.DEVICE)
        self.assertTrue((ids_r == ids_exp).all())

    def test_embedding_lookup_eval(self):
        ht = HashTable(
            embedding_shape=[self.emb_dim],
            block_size=self.block_size,
            dtype=self.dtype,
            device=torch.device(self.DEVICE),
            name="gpu_ht_eval",
        )
        ht.eval()

        # init hashtable
        ids = torch.arange(self.ids_num, device=self.DEVICE)
        ht(ids)

        # insert datas
        ids_beg = self.ids_num
        ids = torch.arange(ids_beg, ids_beg + self.ids_num, device=self.DEVICE)
        emb = torch.tile(ids.reshape([-1, 1]), [1, self.emb_dim])
        ht._hashtable_impl.insert(ids, emb.type(self.dtype))
        ht(ids)

        # missing key
        ids_beg += self.ids_num
        ids = torch.arange(
            ids_beg - self.ids_num, ids_beg + self.ids_num, device=self.DEVICE
        )
        ht(ids)

        ids_r, _ = torch.sort(ht.ids())
        ids_exp = torch.arange(self.ids_num, 2 * self.ids_num, device=self.DEVICE)
        self.assertTrue((ids_r == ids_exp).all())


if __name__ == "__main__":
    unittest.main()
