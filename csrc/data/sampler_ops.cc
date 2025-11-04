#include "data/sampler_ops.h"

#include <ATen/Dispatch.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>

namespace recis {
namespace data {

Tensor TileWithSampleCounts(Tensor input_tensor, Tensor sample_counts) {
  TORCH_CHECK(input_tensor.sizes()[0] == sample_counts.numel(),
              "shape(input_tensor)[0] should == size(sample_counts)");
  return AT_DISPATCH_INDEX_TYPES(
      sample_counts.scalar_type(), "TileWithSampleCounts0", [&]() {
        auto sample_counts_vec = sample_counts.const_data_ptr<index_t>();
        auto output_size = input_tensor.sizes()[0];
        for (size_t i = 0; i < sample_counts.numel(); i++) {
          output_size += sample_counts_vec[i];
          TORCH_CHECK(sample_counts_vec[i] >= 0,
                      "values in sample counts must be >= 0");
        }
        size_t dimension = 1;
        std::vector<int64_t> outshape;
        outshape.reserve(input_tensor.ndimension());
        outshape.push_back(output_size);
        for (size_t i = 1; i < input_tensor.ndimension(); i++) {
          dimension *= input_tensor.sizes()[i];
          outshape.push_back(input_tensor.sizes()[i]);
        }

        Tensor output = at::empty(outshape, input_tensor.scalar_type());

        AT_DISPATCH_ALL_TYPES(
            input_tensor.scalar_type(), "TileWithSampleCounts1", [&]() {
              auto input_tensor_vec = input_tensor.const_data_ptr<scalar_t>();
              auto output_vec = output.mutable_data_ptr<scalar_t>();
              size_t output_idx = 0;
              for (size_t i = 0; i < input_tensor.sizes()[0]; i++) {
                for (size_t k = 0; k < dimension; k++) {
                  output_vec[output_idx * dimension + k] =
                      input_tensor_vec[i * dimension + k];
                }
                output_idx++;
                for (size_t j = 0; j < sample_counts_vec[i]; j++) {
                  for (size_t k = 0; k < dimension; k++) {
                    output_vec[output_idx * dimension + k] =
                        input_tensor_vec[i * dimension + k];
                  }
                  output_idx++;
                }
              }
            });
        return output;
      });
}

Tensor CombineVectorWithSampleCounts(Tensor origin_vector, Tensor sample_counts,
                                     Tensor sampled_vector) {
  // get input tensor
  TORCH_CHECK(origin_vector.sizes()[0] == sample_counts.numel(),
              "size(origin_vector) ", origin_vector.sizes()[0],
              " should == size(sample_counts) ", sample_counts.numel());
  return AT_DISPATCH_ALL_TYPES(
      sample_counts.scalar_type(), "CombineVectorWithSampleCounts0", [&]() {
        auto sample_counts_vec = sample_counts.const_data_ptr<scalar_t>();
        auto output_size = origin_vector.sizes()[0];
        for (size_t i = 0; i < sample_counts.numel(); i++) {
          output_size += sample_counts_vec[i];
          TORCH_CHECK(sample_counts_vec[i] >= 0,
                      "values in sample counts must be >= 0");
        }

        TORCH_CHECK(
            output_size - origin_vector.sizes()[0] == sampled_vector.sizes()[0],
            "sum(sample_counts) must == size(sampled_vector)");
        TORCH_CHECK(
            origin_vector.numel() / origin_vector.sizes()[0] ==
                sampled_vector.numel() / sampled_vector.sizes()[0],
            "subdims of origin vector and sampled vector must be equal");

        std::vector<int64_t> outshape;
        outshape.reserve(origin_vector.ndimension());
        outshape.push_back(output_size);
        size_t dimension = 1;
        for (size_t i = 1; i < origin_vector.ndimension(); i++) {
          outshape.push_back(origin_vector.sizes()[i]);
          dimension *= origin_vector.sizes()[i];
        }
        Tensor output = at::empty(outshape, origin_vector.scalar_type());

        AT_DISPATCH_ALL_TYPES(
            output.scalar_type(), "CombineVectorWithSampleCounts1", [&]() {
              auto origin_vector_vec = origin_vector.const_data_ptr<scalar_t>();
              auto sampled_vector_vec =
                  sampled_vector.const_data_ptr<scalar_t>();
              auto output_vec = output.mutable_data_ptr<scalar_t>();
              size_t output_idx = 0;
              size_t sampled_vec_ids = 0;
              for (size_t i = 0; i < origin_vector.sizes()[0]; i++) {
                for (size_t k = 0; k < dimension; k++) {
                  output_vec[output_idx * dimension + k] =
                      origin_vector_vec[i * dimension + k];
                }
                output_idx++;
                for (size_t j = 0; j < sample_counts_vec[i]; j++) {
                  for (size_t k = 0; k < dimension; k++) {
                    output_vec[output_idx * dimension + k] =
                        sampled_vector_vec[sampled_vec_ids * dimension + k];
                  }
                  output_idx++;
                  sampled_vec_ids++;
                }
              }
            });
        return output;
      });
}

}  // namespace data
}  // namespace recis
