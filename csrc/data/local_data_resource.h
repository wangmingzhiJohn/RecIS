#pragma once

#include <ATen/Dispatch.h>
#include <ATen/core/TensorBody.h>
#include <torch/custom_class.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

#include "data/fast_weighted_collection.h"
#include "data/weighted_collection_factory.h"

namespace recis {
namespace data {

using Tensor = at::Tensor;

// The macro CASES() expands to a switch statement conditioned on
// TYPE_ENUM. Each case expands the STMTS after a typedef for T.
#define SINGLE_ARG(...) __VA_ARGS__
#define CASE(TYPE, STMTS)               \
  case ScalarTypeToEnum<TYPE>::value: { \
    typedef TYPE T;                     \
    STMTS;                              \
    break;                              \
  }
#define CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, INVALID, DEFAULT) \
  switch (TYPE_ENUM) {                                         \
    CASE(float, SINGLE_ARG(STMTS))                             \
    CASE(double, SINGLE_ARG(STMTS))                            \
    CASE(int32, SINGLE_ARG(STMTS))                             \
    CASE(uint8, SINGLE_ARG(STMTS))                             \
    CASE(uint16, SINGLE_ARG(STMTS))                            \
    CASE(uint32, SINGLE_ARG(STMTS))                            \
    CASE(uint64, SINGLE_ARG(STMTS))                            \
    CASE(int16, SINGLE_ARG(STMTS))                             \
    CASE(int8, SINGLE_ARG(STMTS))                              \
    CASE(int64, SINGLE_ARG(STMTS))                             \
    CASE(bool, SINGLE_ARG(STMTS))                              \
    CASE(qint32, SINGLE_ARG(STMTS))                            \
    CASE(quint8, SINGLE_ARG(STMTS))                            \
    CASE(qint8, SINGLE_ARG(STMTS))                             \
    CASE(quint16, SINGLE_ARG(STMTS))                           \
    CASE(qint16, SINGLE_ARG(STMTS))                            \
    case DT_INVALID:                                           \
      INVALID;                                                 \
      break;                                                   \
    default:                                                   \
      DEFAULT;                                                 \
      break;                                                   \
  }

#define CASES(TYPE_ENUM, STMTS)                                      \
  CASES_WITH_DEFAULT(TYPE_ENUM, STMTS, LOG(FATAL) << "Type not set"; \
                     , LOG(FATAL) << "Unexpected type: " << TYPE_ENUM;)

class SparseIndexer {
 public:
  SparseIndexer(std::vector<Tensor> feature) { feature_ = feature; }

  std::vector<size_t> Find(size_t index) {
    size_t begin = index, end = index + 1;
    for (int k = feature_.size() - 1; k > 0; --k) {
      AT_DISPATCH_INDEX_TYPES(
          feature_[k].scalar_type(), "SparseIndexerFindSplits", [&]() {
            auto splits = feature_[k].mutable_data_ptr<index_t>();
            begin = splits[begin];
            end = splits[end];
          });
    }

    auto scalar_type = feature_[0].scalar_type();
    std::vector<size_t> res;
    res.reserve(end - begin);
    AT_DISPATCH_ALL_TYPES(scalar_type, "SparseIndexerFindValues", [&]() {
      auto values = feature_[0].mutable_data_ptr<scalar_t>();
      for (size_t i = begin; i < end; ++i) {
        res.push_back(values[i]);
      }
    });
    // CASES(scalar_type, do {
    //   auto values = feature_[0].mutable_data_ptr<T>();
    //   for (size_t i = begin; i < end; ++i) {
    //     res.push_back(values[i]);
    //   }
    // } while(0));
    return res;
  }

 private:
  std::vector<Tensor> feature_;
};

class DenseIndexer {
 public:
  DenseIndexer(std::vector<Tensor> feature) {
    feature_ = feature;
    auto shape = feature_[0].sizes();
    for (int i = 1; i < shape.size(); ++i) {
      dense_dim_ *= shape[i];
    }
  }

  std::vector<double> Find(size_t index) {
    auto scalar_type = feature_[0].scalar_type();
    std::vector<double> res;
    res.reserve(dense_dim_);
    AT_DISPATCH_ALL_TYPES(scalar_type, "DenseIndexerFind", [&]() {
      auto values = feature_[0].mutable_data_ptr<scalar_t>();
      for (size_t i = index * dense_dim_; i < (index + 1) * dense_dim_; ++i) {
        res.push_back(values[i]);
      }
    });
    // CASES(scalar_type, do {
    //   auto values = feature_[0].mutable_data_ptr<T>();
    //   for (size_t i = index*dense_dim_; i < (index+1)*dense_dim_; ++i) {
    //     res.push_back(values[i]);
    //   }
    // } while(0));
    return res;
  }

 private:
  std::vector<Tensor> feature_;
  size_t dense_dim_ = 1;
};

class LocalDataResource : public torch::CustomClassHolder {
 public:
  std::tuple<Tensor, Tensor, Tensor> SampleIds(
      std::vector<Tensor> sample_tag_tensors,
      std::vector<Tensor> dedup_tag_tensors, Tensor sample_cnts,
      bool avoid_conflict, int64_t pos_num,
      bool avoid_conflict_with_all_dedup_tags) const;
  void SampleIdsWithSampleCounts(std::vector<Tensor> &sample_tag_tensors,
                                 std::vector<Tensor> &dedup_tag_tensors,
                                 int64_t sample_cnt, const Tensor *sample_cnts,
                                 bool avoid_conflict, size_t pos_num,
                                 Tensor *id_tensor_t, Tensor *weight_tensor_t,
                                 Tensor *cate_tensor_t);
  Tensor ValidSampleIds(Tensor sample_ids_tensor, int64_t default_value) const;
  void PackFeature(std::vector<Tensor> &input_tensors,
                   const Tensor &sample_ids_tensor,
                   std::vector<std::string> &names,
                   std::vector<int> &ragged_ranks, int64_t default_value,
                   std::vector<Tensor> &out_tensors);
  void PackFeatureWithSampleCounts(std::vector<Tensor> &input_tensors,
                                   const Tensor &sample_ids_tensor,
                                   const Tensor *sample_cnts,
                                   std::vector<std::string> &names,
                                   std::vector<int> &ragged_ranks,
                                   int64_t default_value,
                                   std::vector<Tensor> &out_tensors);
  std::vector<Tensor> ExtractFeature(Tensor sample_ids_tensor,
                                     std::vector<std::string> names,
                                     Tensor ragged_ranks,
                                     int64_t default_value) const;
  void ExtractFeatureWithSampleCounts(const Tensor *sample_cnts_tensor,
                                      const Tensor *origin_ids_tensor,
                                      const Tensor &sample_ids_tensor,
                                      std::vector<std::string> &names,
                                      std::vector<int> &ragged_ranks,
                                      int64_t default_value,
                                      std::vector<Tensor> &out_tensors);
  void DecorateSkeyWithSampleCounts(Tensor &skey_tensor, size_t pos_num,
                                    const Tensor *sample_cnts);
  void LoadByBatch(std::vector<Tensor> input_tensors, std::string sample_tag,
                   std::string dedup_tag, std::string weight_tag,
                   std::string skey_name, bool put_back,
                   std::vector<std::string> names, Tensor ragged_ranks,
                   bool ignore_invalid_dedup_tag);
  void GetIdsWeights(std::vector<Tensor> dedup_tag_tensors,
                     const Tensor &default_value, Tensor *output);
  void Clear();
  bool Initialized();

  Tensor CombineVectorWithSampleCounts(Tensor origin_vector,
                                       Tensor sample_counts,
                                       Tensor sampled_vector);

 private:
  void ClearMembers();
  void InitSampler();
  void NoPutBackSample(std::vector<size_t> &cates,
                       std::set<size_t> &dedup_id_set, size_t sample_cnt,
                       std::vector<size_t> &out_ids,
                       std::vector<size_t> &out_cates,
                       std::vector<double> &out_weights) const;
  void PutBackSample(std::vector<size_t> &cates, std::set<size_t> &dedup_id_set,
                     size_t sample_cnt, std::vector<size_t> &out_ids,
                     std::vector<size_t> &out_cates,
                     std::vector<double> &out_weights) const;
  void FlatLocalData(std::vector<std::string> &names, Tensor &ragged_ranks,
                     std::vector<Tensor> &flat_tensors) const;
  void LoadByFlatTensors(std::vector<Tensor> &flat_tensors,
                         std::vector<std::string> &names, Tensor &ragged_ranks);
  void Pack(std::vector<std::vector<Tensor>> &tables, Tensor &ragged_ranks_t,
            std::vector<std::pair<size_t, size_t>> &all_table_ranges,
            std::vector<Tensor> &out_tensors) const;
  void DecorateSkey(Tensor &skey_tensor, size_t pos_num, size_t sample_cnt);

  std::vector<std::string> paths_;
  std::vector<std::string> selected_columns_;
  std::vector<std::string> hash_features_;
  std::vector<std::string> dense_features_;
  std::vector<Tensor> dense_defaults_;
  bool is_compressed_;
  int32_t batch_size_;
  std::vector<c10::ScalarType> output_types_;
  std::string sample_tag_, dedup_tag_, weight_tag_, skey_name_;
  bool put_back_ = true;
  mutable std::mutex mu_;
  mutable std::condition_variable cond_var_;
  // bool loading_ = false;
  std::atomic_bool initialized_{false};
  mutable std::atomic_int reader_count_{0};
  std::atomic_bool sampler_initialized_{false};
  std::vector<std::string> selected_table_columns_;

  std::map<std::string, std::vector<std::vector<Tensor>>> map_feature_tensors_;
  std::map<size_t, std::vector<std::pair<size_t, double>>>
      map_cate_id_weight_pairs_;
  std::map<size_t, double> map_cate_weight_;
  std::map<size_t, size_t> map_id_index_;
  std::map<size_t, std::shared_ptr<WeightedCollection<size_t>>>
      map_cate_sampler_;
  // std::atomic<bool> initialized_(false);
  // std::atomic<bool> sampler_initialized_(false);

  const int kPackTensorBlock_ = 16;
  bool ignore_invalid_dedup_tag_;
};

#undef CASES
#undef CASE
}  // namespace data
}  // namespace recis
