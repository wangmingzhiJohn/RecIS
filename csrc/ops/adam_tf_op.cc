#include "adam_tf_op.h"

#include <cmath>

namespace recis {
namespace functional {

template <typename scalar_t>
void adam_tf_apply_cpu_kernel(scalar_t* param, scalar_t* grad, scalar_t* avg,
                              scalar_t* avg_sq, float step, float lr, float b1,
                              float b2, float eps, const int64_t param_size) {
  float b1_power = std::pow(b1, step);
  float b2_power = std::pow(b2, step);
  float alpha = -lr / (1. - b1_power) * std::sqrt(1. - b2_power);
  for (int64_t i = 0; i < param_size; ++i) {
    avg[i] = avg[i] * b1 + grad[i] * (1. - b1);
    avg_sq[i] = avg_sq[i] * b2 + (1. - b2) * grad[i] * grad[i];
    param[i] = param[i] + alpha * (avg[i] / (std::sqrt(avg_sq[i]) + eps));
  }
}

void adam_tf_apply(torch::Tensor param, torch::Tensor grad, torch::Tensor avg,
                   torch::Tensor avg_sq, torch::Scalar step, torch::Scalar lr,
                   torch::Scalar beta1, torch::Scalar beta2,
                   torch::Scalar eps) {
  auto param_size = param.numel();
  float step_v = step.to<float>();
  float lr_v = lr.to<float>();
  float b1_v = beta1.to<float>();
  float b2_v = beta2.to<float>();
  float eps_v = eps.to<float>();
  if (param_size == 0) {
    return;
  }

  if (param.device().is_cuda()) {
    adam_tf_apply_cuda(param, grad, avg, avg_sq, step_v, lr_v, b1_v, b2_v,
                       eps_v, param_size);
  } else {
    AT_DISPATCH_FLOATING_TYPES(
        param.scalar_type(), "adam_tf_apply_cpu", ([&] {
          adam_tf_apply_cpu_kernel<scalar_t>(
              param.data_ptr<scalar_t>(), grad.data_ptr<scalar_t>(),
              avg.data_ptr<scalar_t>(), avg_sq.data_ptr<scalar_t>(), step_v,
              lr_v, b1_v, b2_v, eps_v, param_size);
        }));
  }
}

void fused_adamw_tf_apply_cpu(std::vector<torch::Tensor> params,
                              std::vector<torch::Tensor> grads,
                              std::vector<torch::Tensor> avgs,
                              std::vector<torch::Tensor> avg_sqs,
                              std::vector<torch::Tensor> state_steps,
                              float weight_decay, float lr, float beta1,
                              float beta2, float eps) {
  int64_t num_tensors = params.size();
  for (int64_t i = 0; i < num_tensors; i++) {
    torch::parallel_for(
        0, params[i].size(0), 1, [&](int64_t begin, int64_t end) {
          auto step = state_steps[i].item<float>();

          float b1_power = std::pow(beta1, step);
          float b2_power = std::pow(beta2, step);
          float alpha = -lr / (1. - b1_power) * std::sqrt(1. - b2_power);

          for (int64_t j = begin; j < end; j++) {
            auto param = params[i].data_ptr<float>();
            auto grad = grads[i].data_ptr<float>();
            auto avg = avgs[i].data_ptr<float>();
            auto avg_sq = avg_sqs[i].data_ptr<float>();
            avg[j] = avg[j] * beta1 + grad[j] * (1. - beta1);
            avg_sq[j] = avg_sq[j] * beta2 + (1. - beta2) * grad[j] * grad[j];
            param[j] = param[j] * (1. - weight_decay) +
                       alpha * (avg[j] / (std::sqrt(avg_sq[j]) + eps));
          }
        });
  }
}

bool check_inputs_all_cuda(std::vector<torch::Tensor> params,
                           std::vector<torch::Tensor> grads,
                           std::vector<torch::Tensor> avg,
                           std::vector<torch::Tensor> avg_sq) {
  for (auto& tensor : params) {
    if (!tensor.device().is_cuda()) {
      return false;
    }
  }
  for (auto& tensor : grads) {
    if (!tensor.device().is_cuda()) {
      return false;
    }
  }
  for (auto& tensor : avg) {
    if (!tensor.device().is_cuda()) {
      return false;
    }
  }
  for (auto& tensor : avg_sq) {
    if (!tensor.device().is_cuda()) {
      return false;
    }
  }
  return true;
}

void fused_adamw_tf_apply(std::vector<torch::Tensor> params,
                          std::vector<torch::Tensor> grads,
                          std::vector<torch::Tensor> avg,
                          std::vector<torch::Tensor> avg_sq,
                          std::vector<torch::Tensor> state_steps,
                          torch::Scalar weight_decay, torch::Scalar lr,
                          torch::Scalar beta1, torch::Scalar beta2,
                          torch::Scalar eps) {
  float weight_decay_v = weight_decay.to<float>();
  float lr_v = lr.to<float>();
  float b1_v = beta1.to<float>();
  float b2_v = beta2.to<float>();
  float eps_v = eps.to<float>();
  for (int64_t i = 0; i < state_steps.size(); ++i) {
    state_steps[i].add_(1);
  }
  if (params.size() == 0) {
    return;
  }
  if (params[0].device().is_cuda()) {
    if (!check_inputs_all_cuda(params, grads, avg, avg_sq)) {
      throw std::runtime_error("All inputs must be on the same device.");
    }
    fused_adamw_tf_apply_cuda(params, grads, avg, avg_sq, state_steps,
                              weight_decay_v, lr_v, b1_v, b2_v, eps_v);
  } else {
    fused_adamw_tf_apply_cpu(params, grads, avg, avg_sq, state_steps,
                             weight_decay_v, lr_v, b1_v, b2_v, eps_v);
  }
}

}  // namespace functional
}  // namespace recis
