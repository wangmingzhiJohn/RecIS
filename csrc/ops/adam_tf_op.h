#include <torch/extension.h>

namespace recis {
namespace functional {
void adam_tf_apply(torch::Tensor param, torch::Tensor grad, torch::Tensor avg,
                   torch::Tensor avg_sq, torch::Scalar step, torch::Scalar lr,
                   torch::Scalar beta1, torch::Scalar beta2, torch::Scalar eps);
void adam_tf_apply_cuda(torch::Tensor param, torch::Tensor grad,
                        torch::Tensor avg, torch::Tensor avg_sq, float step,
                        float lr, float beta1, float beta2, float eps,
                        int64_t param_size);

void fused_adamw_tf_apply(std::vector<torch::Tensor> params,
                          std::vector<torch::Tensor> grads,
                          std::vector<torch::Tensor> avg,
                          std::vector<torch::Tensor> avg_sq,
                          std::vector<torch::Tensor> state_steps,
                          torch::Scalar weight_decay, torch::Scalar lr,
                          torch::Scalar beta1, torch::Scalar beta2,
                          torch::Scalar eps);

void fused_adamw_tf_apply_cuda(std::vector<torch::Tensor> params,
                               std::vector<torch::Tensor> grads,
                               std::vector<torch::Tensor> avg,
                               std::vector<torch::Tensor> avg_sq,
                               std::vector<torch::Tensor> state_steps,
                               float weight_decay, float lr, float beta1,
                               float beta2, float eps);

void fused_adamw_tf_apply_cpu(std::vector<torch::Tensor> params,
                              std::vector<torch::Tensor> grads,
                              std::vector<torch::Tensor> avg,
                              std::vector<torch::Tensor> avg_sq,
                              std::vector<torch::Tensor> state_steps,
                              float weight_decay, float lr, float beta1,
                              float beta2, float eps);

}  // namespace functional
}  // namespace recis
