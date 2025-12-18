import unittest
from typing import List

import numpy as np
import torch

import recis.nn.functional.ragged_ops as ragged


def get_table(rows, cols, random=False):
    matrix = []
    if not random:
        for i in range(rows):
            row = []
            for j in range(cols):
                value = i + j / 100.0
                row.append(value)
            matrix.append(row)
        matrix = np.array(matrix, dtype=np.float32)
    else:
        matrix = np.random.rand(rows, cols).astype(np.float32)
    return matrix


def get_dy(batch, seq, dim, random=False):
    row = np.array(batch) * np.array(seq)
    cum = np.cumsum(row).tolist()
    rows = cum[-1]
    dy = get_table(rows, dim, random)
    return dy


def splite_dy(batch, seq, dy):
    dys = []
    row = np.array(batch) * np.array(seq)
    cum = [0] + np.cumsum(row).tolist()
    for idx in range(len(cum) - 1):
        start = cum[idx]
        end = cum[idx + 1]
        cur_dy = dy[start:end]
        dys.append(cur_dy)
    return dys


def combin_offset(offset):
    out_off = offset[0].copy()
    for idx in range(1, len(offset)):
        compen = out_off[-1]
        cur = offset[idx]
        out = cur[1:]
        out = [x + compen for x in out]
        out_off += out
    return out_off


def splite_off(batch, offset):
    bt = batch.copy()
    bt[0] += 1
    batch_cum = [0] + np.cumsum(bt).tolist()
    out = []
    last_off = 0
    for idx in range(1, len(batch_cum)):
        cur_off = offset[batch_cum[idx - 1] : batch_cum[idx]]
        if idx >= 2:
            cur_off = [last_off] + cur_off
        last_off = cur_off[-1]
        out.append(cur_off)
    return out


def tile_row_cpu(offset, seq, table):
    table_row, dim = table.shape
    out = []
    arr_zero = np.zeros(dim, dtype=table.dtype)  # used to null value to padding
    for idx in range(0, len(offset) - 1):
        row_start = offset[idx]
        row_end = offset[idx + 1]
        row_len = row_end - row_start
        for id in range(seq):
            cur = None
            if id < row_len:
                cur = table[row_start + id, :]
            else:
                cur = arr_zero
            out.append(cur)
    return out


def tile_back_row_cpu(offset, seq, indices, dy, d_table):
    for idx in range(len(offset) - 1):
        start = offset[idx]
        end = offset[idx + 1]
        len_off = end - start
        lens = min(len_off, seq)
        for id in range(lens):
            indice = indices[start + id]
            cur_dy = dy[idx][id]
            d_table[indice, :] += cur_dy


def restore_table(table, value):
    out = []
    row, col = table.shape
    for idx in range(len(value)):
        val = value[idx]
        assert val < row
        obj = table[val, :]
        out.append(obj)
    return np.array(out)


def tile_cpu(
    value: List[int], offset: List[int], seq: List[int], table: np.ndarray
) -> np.ndarray:
    out = []
    new_table = restore_table(table, value)
    for idx in range(len(offset)):
        rt = tile_row_cpu(offset[idx], seq[idx], new_table)
        out = out + rt
    return np.array(out)


def tile_row_torch(offset, seq, table):
    dim = table.shape[1]
    padding_vec = torch.zeros(1, dim, dtype=table.dtype, device=table.device)
    table_with_padding = torch.cat([table, padding_vec], dim=0)
    padding_idx = table.shape[0]

    starts = offset[:-1]
    lengths = offset[1:] - starts

    arange_seq = torch.arange(seq, device=table.device).unsqueeze(0)
    indices_grid = starts.unsqueeze(1) + arange_seq
    mask = arange_seq < lengths.unsqueeze(1)

    final_indices = torch.where(mask, indices_grid, padding_idx)
    output = table_with_padding[final_indices]
    output = output.view(-1, dim)

    return output


def tile_fwd_torch(
    value: torch.Tensor,
    offsets: List[torch.Tensor],
    seqs: List[int],
    table: torch.Tensor,
):
    out = []
    table_src = table[value]
    for idx in range(len(seqs)):
        offset = offsets[idx]
        max_seq = seqs[idx]
        rt = tile_row_torch(offset, max_seq, table_src)
        out.append(rt)
    return torch.cat(out, dim=0)


def tile_backward_cpu(
    value: List[int],
    offset: List[int],
    seq: List[int],
    dy: List[np.ndarray],
    dx_shape: tuple,
) -> np.ndarray:
    dx = np.zeros(dx_shape, dy[0].dtype)
    for idx in range(len(offset)):
        cur_off = offset[idx]
        shape = (len(cur_off) - 1, seq[idx], dx_shape[1])
        cur_dy = dy[idx].reshape(shape)
        tile_back_row_cpu(cur_off, seq[idx], value, cur_dy, dx)
    return dx


def tile_back_row_torch(
    offset: torch.Tensor,
    seq: int,
    indices: torch.Tensor,
    dy: torch.Tensor,
    dx_shape: tuple,
):
    batch_size, max_len, _ = dy.shape
    device = dy.device
    d_table = torch.zeros(dx_shape, dtype=dy.dtype, device=dy.device)
    off_lens = offset[1:] - offset[:-1]
    effect_len = torch.min(off_lens, torch.tensor(seq, device=device))
    arange_t = torch.arange(max_len, device=device)
    mask = arange_t < effect_len.unsqueeze(1)

    src_dy = dy[mask]
    id_tensor = torch.arange(max_len, device=device).unsqueeze(0)
    flat_indices_map = offset[:-1].unsqueeze(1) + id_tensor
    valid_id_loc = flat_indices_map[mask]
    indices_new = indices[valid_id_loc]
    d_table.index_add_(0, indices_new, src_dy)
    return d_table


def tile_backward_torch(
    value: torch.Tensor,
    offset: torch.Tensor,
    seq: List[int],
    dy: List[torch.Tensor],
    dx_shape: tuple,
) -> np.ndarray:
    dx = torch.zeros(dx_shape, dtype=dy[0].dtype, device=dy[0].device)
    for idx in range(len(offset)):
        cur_off = offset[idx]
        shape = (len(cur_off) - 1, seq[idx], dx_shape[1])
        cur_dy = dy[idx].reshape(shape)
        cur_dx = tile_back_row_torch(cur_off, seq[idx], value, cur_dy, dx_shape)
        dx += cur_dx
    return dx


def tile_para():
    batch = [3, 3, 4]
    seq = [3, 4, 5]
    offset1 = [0, 1, 3, 4]
    offset2 = [0, 1, 4, 5]
    offset3 = [0, 1, 5, 6, 7]
    offset = [offset1, offset2, offset3]
    dim = 2
    combin_off = combin_offset(offset)
    value = list(range(max(combin_off)))
    val_max = max(value)
    table = get_table(val_max + 1, dim)
    dy = get_dy(batch, seq, dim)
    out = {
        "batch": batch,
        "seq": seq,
        "offset": combin_off,
        "value": value,
        "table": table,
        "dy": dy,
    }
    return out


def tile_para_random(
    batch: tuple = (10, 20),
    seq: tuple = (16, 32),
    tensor_num: int = 10,
    value_len: int = 128,
    value_max: int = 100,
    dim: int = 32,
):
    batch_min, batch_max = batch[0], batch[1]
    seq_min, seq_max = seq[0], seq[1]

    batch = np.random.randint(
        batch_min, batch_max, size=tensor_num, dtype=np.int32
    ).tolist()
    seq = np.random.randint(seq_min, seq_max, size=tensor_num, dtype=np.int32).tolist()
    batch_lens = np.sum(batch).item()
    offset = np.random.randint(0, value_len, size=batch_lens + 1, dtype=np.int32)
    offset = np.sort(offset).tolist()
    offset[0] = 0
    value = np.random.randint(0, value_max, size=max(offset), dtype=np.int32)
    value = np.sort(value).tolist()
    value_max = max(value)
    table = get_table(value_max + 1, dim, True)
    dy = get_dy(batch, seq, dim, True)
    out = {
        "batch": batch,
        "seq": seq,
        "offset": offset,
        "value": value,
        "table": table,
        "dy": dy,
    }
    return out


def para_impro(para):
    para_cpu = para.copy()
    para_gpu = para.copy()
    para_torch = para.copy()
    # cpu
    para_cpu["offset"] = splite_off(para_cpu["batch"], para_cpu["offset"])
    para_cpu["dy"] = splite_dy(para_cpu["batch"], para_cpu["seq"], para_cpu["dy"])
    # torch
    para_torch["offset"] = [torch.tensor(x) for x in para_cpu["offset"].copy()]
    para_torch["dy"] = [torch.tensor(x) for x in para_cpu["dy"].copy()]
    para_torch["value"] = torch.tensor(para_torch["value"])
    para_torch["table"] = torch.tensor(para_torch["table"])
    # gpu
    device = "cuda"
    para_gpu["offset"] = torch.tensor(para_gpu["offset"], device=device)
    para_gpu["value"] = torch.tensor(para_gpu["value"], device=device)
    para_gpu["table"] = torch.tensor(para_gpu["table"], device=device)
    para_gpu["dy"] = torch.tensor(para_gpu["dy"], device=device)
    return (para_cpu, para_gpu, para_torch)


class TestRaggedTileOp(unittest.TestCase):
    def _test_fwd(self, para):
        tile_func = tile_cpu
        para_cpu, para_gpu, para_torch = para
        x_torch = tile_fwd_torch(
            para_torch["value"],
            para_torch["offset"],
            para_torch["seq"],
            para_torch["table"],
        )
        x_cpu = tile_func(
            para_cpu["value"], para_cpu["offset"], para_cpu["seq"], para_cpu["table"]
        )
        x_gpu = ragged.ragged_tile(
            para_gpu["batch"],
            para_gpu["seq"],
            para_gpu["value"],
            para_gpu["offset"],
            para_gpu["table"],
        )
        self.assertTrue(torch.allclose(torch.tensor(x_cpu), x_gpu.cpu(), atol=1e-6))
        self.assertTrue(torch.allclose(x_torch.cpu(), x_gpu.cpu(), atol=1e-6))

    def _test_bwd(self, para):
        para_cpu, para_gpu, para_torch = para
        para_gpu["table"].requires_grad_(True)
        dx_cpu = tile_backward_cpu(
            para_cpu["value"],
            para_cpu["offset"],
            para_cpu["seq"],
            para_cpu["dy"],
            para_cpu["table"].shape,
        )
        dx_torch = tile_backward_torch(
            para_torch["value"],
            para_torch["offset"],
            para_torch["seq"],
            para_torch["dy"],
            para_torch["table"].shape,
        )
        y_gpu = ragged.ragged_tile(
            para_gpu["batch"],
            para_gpu["seq"],
            para_gpu["value"],
            para_gpu["offset"],
            para_gpu["table"],
        )
        y_gpu.backward(para_gpu["dy"])
        dx_gpu = para_gpu["table"].grad
        self.assertTrue(torch.allclose(torch.tensor(dx_cpu), dx_gpu.cpu(), atol=1e-6))
        self.assertTrue(torch.allclose(dx_torch.cpu(), dx_gpu.cpu(), atol=1e-6))

    def test_fwd_and_bwd(self):
        configurations = [
            {
                "batch": (10, 20),
                "seq": (8, 24),
                "tensor_num": 10,
                "value_len": 32,
                "value_max": 60,
                "dim": 32,
            },
            {
                "batch": (30, 41),
                "seq": (11, 19),
                "tensor_num": 7,
                "value_len": 71,
                "value_max": 83,
                "dim": 7,
            },
            {
                "batch": (1, 7),
                "seq": (10, 13),
                "tensor_num": 8,
                "value_len": 50,
                "value_max": 81,
                "dim": 29,
            },
        ]
        for params in configurations:
            with self.subTest(msg="Testing with params", **params):
                para = tile_para_random(
                    params["batch"],
                    params["seq"],
                    params["tensor_num"],
                    params["value_len"],
                    params["value_max"],
                    params["dim"],
                )
                para = para_impro(para)
                self._test_fwd(para)
                self._test_bwd(para)

    # sample data for debug
    def test_simple_data(self):
        para = tile_para()
        para = para_impro(para)
        self._test_fwd(para)
        self._test_bwd(para)


if __name__ == "__main__":
    unittest.main()
