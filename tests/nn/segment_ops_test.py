import unittest

import torch

from recis.nn.functional.embedding_ops import segment_sum_sparse, weight_norm_sparse


def get_src_tensor(src, indices, num_segments):
    dim = src.shape[1]
    indices_row = indices.shape[0]
    out_shape = (indices_row, dim)
    out_tensor = torch.zeros(out_shape, dtype=src.dtype)
    for idx in range(len(indices)):
        val = indices[idx].item()
        out_tensor[idx] = src[val]
    return out_tensor


def tensor_weight(sum_tensor, weight_sum):
    for idx in range(len(weight_sum)):
        val = weight_sum[idx].item()
        if val != 0:
            sum_tensor[idx] = sum_tensor[idx] / weight_sum[idx]
    return sum_tensor


def segment_weight_mean_cpu(data, weight, indices, segment_ids, num_segments):
    assert data.dim() == 2, "dim must be 2"
    dim = data.shape[1]
    out_shape = (num_segments, dim)
    src_tensor = get_src_tensor(data, indices, num_segments)
    out_tensor = torch.zeros(out_shape, dtype=src_tensor.dtype)
    weight_sum = torch.zeros(num_segments, dtype=weight.dtype)
    for idx in range(segment_ids.numel()):
        seg_id = segment_ids[idx].item()
        w_val = weight[idx]
        src_val = src_tensor[idx]
        out_tensor[seg_id] += src_val * w_val
        weight_sum[seg_id] += weight[idx] * 1
    out = tensor_weight(out_tensor, weight_sum)
    return out


def segment_weight_sum_cpu(data, weight, indices, segment_ids, num_segments):
    assert data.dim() == 2, "dim must be 2"
    dim = data.shape[1]
    out_shape = (num_segments, dim)
    src_tensor = get_src_tensor(data, indices, num_segments)
    out_tensor = torch.zeros(out_shape, dtype=src_tensor.dtype)
    lens = min(segment_ids.numel(), src_tensor.numel())
    for idx in range(lens):
        seg_id = segment_ids[idx].item()
        val = src_tensor[idx] * weight[idx]
        out_tensor[seg_id] += val
    return out_tensor


def segment_weight_sum_torch(data, weight, indices, segment_ids, num_segments):
    assert data.dim() == 2, "dim must be 2"
    D = data.shape[1]
    data_src = data[indices]  # torch.index_select(data, 0, indices)
    data_w = data_src * weight.unsqueeze(-1)
    segment_ids2 = segment_ids.view(-1, 1).expand_as(data_w)
    output = torch.zeros((num_segments, D), dtype=data.dtype, device=data.device)
    output.scatter_add_(0, segment_ids2, data_w)
    return output


def segment_weight_mean_torch(data, weight, indices, segment_ids, num_segments):
    assert data.dim() == 2, "dim must be 2"
    D = data.shape[1]
    data_src = data[indices]
    data_w = data_src * weight.unsqueeze(-1)
    numerator = torch.zeros((num_segments, D), dtype=data.dtype, device=data.device)
    segment_ids2 = segment_ids.view(-1, 1).expand_as(data_w)
    numerator.scatter_add_(0, segment_ids2, data_w)
    denominator = torch.zeros(num_segments, dtype=weight.dtype, device=data.device)
    denominator.scatter_add_(0, segment_ids, weight)
    output = numerator / denominator.unsqueeze(-1)
    return output


def get_rand_tensor(shape, dtype=torch.float32, min=0.0, max=10.0):
    random_tensor = (max - min) * torch.rand(shape, dtype=torch.float32) + min
    random_tensor = random_tensor.to(dtype)
    return random_tensor


def get_input_tensor(data_rows, dim, indices_rows, num_segments):
    data = get_rand_tensor((data_rows, dim), torch.float32, min=-1, max=1)
    weight = get_rand_tensor((indices_rows), torch.float32, min=0, max=1)
    indices = get_rand_tensor((indices_rows), torch.int64, min=0, max=data_rows)
    segment_ids = get_rand_tensor((indices_rows), torch.int64, min=0, max=num_segments)
    # Validate inputs; invalid cases will be caught by assertions.
    segment_check(data, indices, segment_ids, num_segments, weight)
    data_g = data.cuda()
    weight_g = weight.cuda()
    indices_g = indices.cuda()
    segment_ids_g = segment_ids.cuda()
    gpu_out = (data_g, indices_g, segment_ids_g, weight_g)
    cpu_out = (data, indices, segment_ids, weight)
    return gpu_out, cpu_out


def segment_check(data, indices, segment_ids, num_segments, weight=None):
    if (data.numel() == 0) and (indices.numel() == 0) and (segment_ids.numel() == 0):
        # In business scenarios, `data`,`indices`, segment_ids can be empty  simultaneously.
        return
    assert data.dim() == 2, "segment data shape must be 2 dim"
    assert indices.max() < data.shape[0], (
        "segment indices max value must be < data rows"
    )
    assert indices.min() >= 0, "segment indices must >= 0!"
    assert segment_ids.max() < num_segments, (
        "segment segment_ids max value must < num_segments"
    )
    assert segment_ids.min() >= 0, "segment segment_ids must >= 0"
    assert len(indices) == len(segment_ids)
    assert num_segments >= 0
    if weight is not None:
        assert len(weight) == len(segment_ids)


class TestSegmentOps(unittest.TestCase):
    def test_segment_weight_sum_rand(self):
        configurations = [
            {"dim": 27, "num_segments": 15, "data_rows": 19, "indices_rows": 7},
            {"dim": 128, "num_segments": 10, "data_rows": 23, "indices_rows": 34},
            {"dim": 10, "num_segments": 4, "data_rows": 5, "indices_rows": 17},
            {
                "dim": 10,
                "num_segments": 4,
                "data_rows": 0,
                "indices_rows": 0,
            },  # empty case
        ]
        for params in configurations:
            with self.subTest(msg="Testing with params", **params):
                dim = params["dim"]
                num_segments = params["num_segments"]
                data_rows = params["data_rows"]
                indices_rows = params["indices_rows"]
                rt = get_input_tensor(data_rows, dim, indices_rows, num_segments)
                data_g, indices_g, segment_ids_g, weight_g = rt[0]
                data, indices, segment_ids, weight = rt[1]

                gpu_out = segment_sum_sparse(
                    data_g, weight_g, indices_g, segment_ids_g, num_segments
                )
                cpu_out = segment_weight_sum_cpu(
                    data, weight, indices, segment_ids, num_segments
                )
                torch_out = segment_weight_sum_torch(
                    data, weight, indices, segment_ids, num_segments
                )

                self.assertTrue(torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-7))
                self.assertTrue(
                    torch.allclose(torch_out.cpu(), gpu_out.cpu(), atol=1e-7)
                )

    def test_segment_weight_mean_rand(self):
        configurations = [
            {"dim": 27, "num_segments": 11, "data_rows": 19, "indices_rows": 70},
            {"dim": 128, "num_segments": 10, "data_rows": 23, "indices_rows": 34},
            {"dim": 10, "num_segments": 4, "data_rows": 5, "indices_rows": 17},
            {
                "dim": 10,
                "num_segments": 4,
                "data_rows": 0,
                "indices_rows": 0,
            },  # empty case
        ]
        for params in configurations:
            with self.subTest(msg="Testing with params", **params):
                dim = params["dim"]
                num_segments = params["num_segments"]
                data_rows = params["data_rows"]
                indices_rows = params["indices_rows"]
                rt = get_input_tensor(data_rows, dim, indices_rows, num_segments)
                data_g, indices_g, segment_ids_g, weight_g = rt[0]
                data, indices, segment_ids, weight = rt[1]
                gpu_weight_norm = weight_norm_sparse(
                    data_g, weight_g, segment_ids_g, num_segments
                )
                gpu_out = segment_sum_sparse(
                    data_g, gpu_weight_norm, indices_g, segment_ids_g, num_segments
                )
                cpu_out = segment_weight_mean_cpu(
                    data, weight, indices, segment_ids, num_segments
                )
                torch_out = segment_weight_mean_cpu(
                    data, weight, indices, segment_ids, num_segments
                )

                self.assertTrue(torch.allclose(gpu_out.cpu(), cpu_out, atol=1e-7))
                self.assertTrue(
                    torch.allclose(gpu_out.cpu(), torch_out.cpu(), atol=1e-7)
                )


if __name__ == "__main__":
    unittest.main()
