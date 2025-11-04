#pragma once

#include "ATen/core/TensorBody.h"

namespace recis {
namespace data {

using Tensor = at::Tensor;

Tensor TileWithSampleCounts(Tensor input_tensor, Tensor sample_counts);

Tensor CombineVectorWithSampleCounts(Tensor origin_vector, Tensor sample_counts,
                                     Tensor sampled_vector);

}  // namespace data
}  // namespace recis
