import unittest

import numpy as np
import torch

from recis.utils.logger import Logger


logger = Logger(__name__)


def make_data(N, shape, device="cuda"):
    np.random.seed(0)
    params = [
        torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device=device)
        for _ in range(N)
    ]
    grads = [
        torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device=device)
        for _ in range(N)
    ]
    avg = [
        torch.from_numpy(np.random.randn(*shape).astype(np.float32)).to(device=device)
        for _ in range(N)
    ]
    avg_sq = [
        torch.from_numpy(np.abs(np.random.randn(*shape).astype(np.float32))).to(
            device=device
        )
        for _ in range(N)
    ]
    state_steps = [torch.scalar_tensor(i) for i in range(N)]
    return params, grads, avg, avg_sq, state_steps


class FusedAdamWTest(unittest.TestCase):
    def test_adamw_tf_cuda(self):
        for device in ["cpu", "cuda"]:
            logger.info(f"Testing fused_adamw_tf_apply on {device}")
            N = 10
            shape = (10,)
            params, grads, avg, avg_sq, state_steps = make_data(N, shape, device=device)
            lr_scalar = 0.001
            beta1 = 0.9
            beta2 = 0.999
            eps = 1e-8
            weight_decay = 0.02

            for i in range(N):
                param, grad, exp_avg, exp_avg_sq = (
                    params[i],
                    grads[i],
                    avg[i],
                    avg_sq[i],
                )
                param.mul_(1.0 - weight_decay)
                step = state_steps[i] + 1
                torch.ops.recis.adam_tf_apply(
                    param,
                    grad,
                    exp_avg,
                    exp_avg_sq,
                    step.item(),
                    lr_scalar,
                    beta1,
                    beta2,
                    eps,
                )

            params_2, grads_2, avg_2, avg_sq_2, state_steps_2 = make_data(
                N, shape, device=device
            )
            torch.ops.recis.fused_adamw_tf_apply(
                params_2,
                grads_2,
                avg_2,
                avg_sq_2,
                state_steps_2,
                weight_decay,
                lr_scalar,
                beta1,
                beta2,
                eps,
            )

            for i in range(N):
                param, param_2 = params[i], params_2[i]
                self.assertTrue(torch.allclose(param, param_2))

            for i in range(N):
                exp_avg, exp_avg_2 = avg[i], avg_2[i]
                self.assertTrue(torch.allclose(exp_avg, exp_avg_2))

            for i in range(N):
                exp_avg_sq, exp_avg_sq_2 = avg_sq[i], avg_sq_2[i]
                self.assertTrue(torch.allclose(exp_avg_sq, exp_avg_sq_2))


if __name__ == "__main__":
    unittest.main()
