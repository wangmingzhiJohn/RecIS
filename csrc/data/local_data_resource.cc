#include "data/local_data_resource.h"

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <c10/util/Exception.h>
#include <c10/util/Logging.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>

#include "data/weighted_collection.h"

namespace recis {
namespace data {

namespace {
class CounterGuard {
 public:
  CounterGuard(std::atomic_int* counter_ptr) : counter_ptr_(counter_ptr) {
    counter_ptr_->fetch_add(1);
  }
  ~CounterGuard() { counter_ptr_->fetch_sub(1); }

 private:
  std::atomic_int* counter_ptr_;
};

class BlockingCounter {
 public:
  explicit BlockingCounter(int initial_count)
      : count_(initial_count), done_(false) {}

  void DecrementCount() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (count_ > 0) {
      --count_;
    }
    if (count_ == 0) {
      done_ = true;
      cv_.notify_all();
    }
  }

  void Wait() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this]() { return done_; });
  }

 private:
  int count_;
  bool done_;
  std::mutex mutex_;
  std::condition_variable cv_;
};
}  // namespace

// This macro used for add read lock for counter;
#define CounterReadLock                                    \
  std::shared_ptr<CounterGuard> counter_guard;             \
  {                                                        \
    std::unique_lock l(mu_);                               \
    if (initialized_.load() == false) {                    \
      LOG(INFO) << "Wait for initialization";              \
      while (initialized_.load() == false) {               \
        cond_var_.wait_for(l, std::chrono::seconds(1));    \
      }                                                    \
      LOG(INFO) << "Initialize done, start to read.";      \
    }                                                      \
    counter_guard.reset(new CounterGuard(&reader_count_)); \
  }

void LocalDataResource::LoadByBatch(
    std::vector<Tensor> input_tensors, std::string sample_tag,
    std::string dedup_tag, std::string weight_tag, std::string skey_name,
    bool put_back, std::vector<std::string> names, Tensor ragged_ranks,
    bool ignore_invalid_dedup_tag) {
  LOG(INFO) << "local data initialize begin!";
  if (!initialized_) {
    std::lock_guard l(mu_);
    if (!initialized_) {
      sample_tag_ = sample_tag;
      dedup_tag_ = dedup_tag;
      weight_tag_ = weight_tag;
      skey_name_ = skey_name;
      put_back_ = put_back;
      ignore_invalid_dedup_tag_ = ignore_invalid_dedup_tag;
      LoadByFlatTensors(input_tensors, names, ragged_ranks);
      InitSampler();
      initialized_.store(true);
      cond_var_.notify_all();
      LOG(INFO) << "local data initialize finished!";
    }
  }
}

void LocalDataResource::LoadByFlatTensors(std::vector<Tensor>& flat_tensors,
                                          std::vector<std::string>& names,
                                          Tensor& ragged_ranks) {
  auto flat_tensor_it = flat_tensors.begin();
  batch_size_ = flat_tensor_it->const_data_ptr<int32_t>()[0];
  flat_tensor_it++;
  auto nit = names.begin();
  for (; nit != names.end(); ++nit) {
    std::vector<Tensor> tuple_of_tensor;
    tuple_of_tensor.push_back(*flat_tensor_it);
    flat_tensor_it++;
    if (map_feature_tensors_.count(*nit) > 0) {
      map_feature_tensors_[*nit].push_back(tuple_of_tensor);
    } else {
      std::vector<std::vector<Tensor>> tuple_of_feature;
      tuple_of_feature.push_back(tuple_of_tensor);
      map_feature_tensors_[*nit] = tuple_of_feature;
    }
  }
  nit = names.begin();
  auto rit = ragged_ranks.const_data_ptr<int32_t>();
  auto ragged_ranks_end = rit + ragged_ranks.numel();
  std::map<std::string, size_t> feature_valcnt;
  for (; nit != names.end() && rit != ragged_ranks_end; ++nit, ++rit) {
    if (feature_valcnt.count(*nit) <= 0) feature_valcnt[*nit] = 0;
    for (int i = 0; i < *rit; ++i) {
      map_feature_tensors_[*nit][feature_valcnt[*nit]].push_back(
          *flat_tensor_it);
      flat_tensor_it++;
    }
    ++feature_valcnt[*nit];
  }
}

void LocalDataResource::InitSampler() {
  LOG(INFO) << "sampler initialize begin!";
  auto iter = map_feature_tensors_.find(sample_tag_);
  TORCH_CHECK(iter != map_feature_tensors_.end(),
              "local data table does not have " + sample_tag_);
  std::vector<std::vector<Tensor>> sample_tag_tup_tensors = iter->second;
  TORCH_CHECK(sample_tag_tup_tensors.size() == 1 &&
                  sample_tag_tup_tensors[0].size() == 2,
              sample_tag_ + "'s type must be sparse");
  std::vector<Tensor> sample_tag_tensors = sample_tag_tup_tensors[0];
  TORCH_CHECK(sample_tag_tensors[1].sizes()[0] == batch_size_ + 1,
              sample_tag_ + "'s size must be ", batch_size_,
              ", but now it's size is ", sample_tag_tensors[1].sizes()[0]);
  iter = map_feature_tensors_.find(dedup_tag_);
  TORCH_CHECK(iter != map_feature_tensors_.end(),
              "local data table does not have " + dedup_tag_);
  std::vector<std::vector<Tensor>> dedup_tag_tup_tensors = iter->second;
  TORCH_CHECK(
      dedup_tag_tup_tensors.size() == 1 && dedup_tag_tup_tensors[0].size() == 2,
      dedup_tag_ + "'s type must be sparse");
  std::vector<Tensor> dedup_tag_tensors = dedup_tag_tup_tensors[0];
  TORCH_CHECK(dedup_tag_tensors[1].sizes()[0] == batch_size_ + 1,
              dedup_tag_ + "'s size must be ", batch_size_,
              ", but now it's size is ", dedup_tag_tensors[1].sizes()[0]);
  iter = map_feature_tensors_.find(weight_tag_);
  TORCH_CHECK(iter != map_feature_tensors_.end(),
              "local data table does not have " + weight_tag_);
  std::vector<std::vector<Tensor>> weight_tag_tup_tensors = iter->second;
  TORCH_CHECK(weight_tag_tup_tensors.size() == 1 &&
                  weight_tag_tup_tensors[0].size() == 1,
              weight_tag_ + "'s type must be dense");
  std::vector<Tensor> weight_tag_tensors = weight_tag_tup_tensors[0];
  TORCH_CHECK(weight_tag_tensors[0].sizes()[0] == batch_size_,
              weight_tag_ + "'s size must be ", batch_size_);

  SparseIndexer sample_tag_indexer(sample_tag_tensors);
  SparseIndexer dedup_tag_indexer(dedup_tag_tensors);
  DenseIndexer weight_tag_indexer(weight_tag_tensors);

  // static cate, ids, weight
  LOG(INFO) << "The total number of negative samples is: " << batch_size_;
  for (int i = 0; i < batch_size_; i++) {
    std::vector<size_t> cates = sample_tag_indexer.Find(i);
    std::vector<size_t> ids = dedup_tag_indexer.Find(i);
    std::vector<double> weights = weight_tag_indexer.Find(i);
    if (ids.size() != 1) {
      TORCH_CHECK(ignore_invalid_dedup_tag_,
                  dedup_tag_ + " must be single value sparse!")
      LOG(WARNING) << dedup_tag_ << " is not single value, the size is "
                   << ids.size() << ", ignore it.";
      continue;
    }
    TORCH_CHECK(weights.size() >= cates.size(), weight_tag_,
                "'s size must be greater than or equal to ", sample_tag_,
                "'s.");
    size_t id = ids[0];
    TORCH_CHECK(map_id_index_.find(id) == map_id_index_.end(),
                dedup_tag_ + "'s id must be globally unique!");
    map_id_index_[id] = i;
    for (int j = 0; j < cates.size(); ++j) {
      size_t cate = cates[j];
      double weight = weights[j];
      if (map_cate_id_weight_pairs_.find(cate) ==
          map_cate_id_weight_pairs_.end()) {
        std::vector<std::pair<size_t, double>> tmp;
        map_cate_id_weight_pairs_[cate] = tmp;
        map_cate_weight_[cate] = 0.0;
      }
      map_cate_id_weight_pairs_[cate].push_back(
          std::pair<size_t, double>(id, weight));
      map_cate_weight_[cate] += weight;
    }
  }

  // init cate sampler
  for (auto mciwp : map_cate_id_weight_pairs_) {
    size_t cate = mciwp.first;
    std::vector<std::pair<size_t, double>> id_weight_pairs = mciwp.second;
    std::shared_ptr<WeightedCollection<size_t>> sampler(
        WeightedCollectionFactory::Create<size_t>(put_back_));
    sampler->Init(id_weight_pairs);
    map_cate_sampler_[cate] = sampler;
  }
  sampler_initialized_ = true;
  LOG(INFO) << "sampler initialize finished!";
}

std::tuple<Tensor, Tensor, Tensor> LocalDataResource::SampleIds(
    std::vector<Tensor> sample_tag_tensors,
    std::vector<Tensor> dedup_tag_tensors, Tensor sample_cnts,
    bool avoid_conflict, int64_t pos_num) const {
  CounterReadLock;
  TORCH_CHECK(sampler_initialized_, "please init sampler!");
  // auto output = output_t->vec<int64>();
  SparseIndexer sample_tag_indexer(sample_tag_tensors);
  SparseIndexer dedup_tag_indexer(dedup_tag_tensors);
  std::vector<size_t> out_ids;
  std::vector<size_t> out_cates;
  std::vector<double> out_weights;
  bool multi_cnts;
  int64_t total_sample_cnt = 0;
  return AT_DISPATCH_INDEX_TYPES(sample_cnts.scalar_type(), "Sample", [&]() {
    auto sample_cnts_vec = sample_cnts.const_data_ptr<index_t>();
    if (sample_cnts.numel() > 1) {
      multi_cnts = true;
      TORCH_CHECK(sample_cnts.numel() == dedup_tag_tensors[0].numel(),
                  "sample_cnts size should be equal to dedup tag size.");
      for (int i = 0; i < sample_cnts.numel(); i++) {
        total_sample_cnt += sample_cnts_vec[i];
      }
    } else {
      multi_cnts = false;
      total_sample_cnt = sample_cnts_vec[0] * pos_num;
    }
    for (int i = 0; i < pos_num; i++) {
      // check dedup_ids
      std::vector<size_t> dedup_ids = dedup_tag_indexer.Find(i);
      TORCH_CHECK(dedup_ids.size() >= 1,
                  "dedup_tag_tensors of positive samples can't be empty");
      std::set<size_t> dedup_id_set(dedup_ids.begin(), dedup_ids.end());
      std::vector<size_t> cates = sample_tag_indexer.Find(i);
      if (put_back_) {
        PutBackSample(cates, dedup_id_set,
                      multi_cnts ? sample_cnts_vec[i] : sample_cnts_vec[0],
                      out_ids, out_cates, out_weights);
      } else {
        NoPutBackSample(cates, dedup_id_set,
                        multi_cnts ? sample_cnts_vec[i] : sample_cnts_vec[0],
                        out_ids, out_cates, out_weights);
      }
    }

    TORCH_CHECK(out_ids.size() == total_sample_cnt, "out_ids.size() ",
                out_ids.size(), " != total_sample_cnt ", total_sample_cnt);
    TORCH_CHECK(out_cates.size() == total_sample_cnt, "out_cates.size() ",
                out_cates.size(), " != total_sample_cnt ", total_sample_cnt);
    TORCH_CHECK(out_weights.size() == total_sample_cnt, "out_weights.size() ",
                out_weights.size(), " != total_sample_cnt ", total_sample_cnt);
    Tensor id_t = at::empty(total_sample_cnt, at::kLong);
    Tensor weight_t = at::empty(total_sample_cnt, at::kDouble);
    Tensor cate_t = at::empty(total_sample_cnt, at::kLong);
    memcpy(id_t.mutable_data_ptr(), out_ids.data(),
           total_sample_cnt * sizeof(int64_t));
    memcpy(weight_t.mutable_data_ptr(), out_weights.data(),
           total_sample_cnt * sizeof(double));
    memcpy(cate_t.mutable_data_ptr(), out_cates.data(),
           total_sample_cnt * sizeof(int64_t));
    return std::make_tuple(id_t, weight_t, cate_t);
  });
}

void LocalDataResource::NoPutBackSample(
    std::vector<size_t>& cates, std::set<size_t>& dedup_id_set,
    size_t sample_cnt, std::vector<size_t>& out_ids,
    std::vector<size_t>& out_cates, std::vector<double>& out_weights) const {
  TORCH_CHECK(
      cates.size() == 1,
      "sampling without replacement don't support mulit value sample_tag!");
  size_t cate = cates.front();
  auto map_cate_sampler_it = map_cate_sampler_.find(cate);
  TORCH_CHECK(map_cate_sampler_it != map_cate_sampler_.end(),
              cate + " not in local data sample_tag");

  auto id_weight_pairs =
      map_cate_sampler_it->second->NSample(sample_cnt + dedup_id_set.size());
  size_t cnt = 0;
  for (auto id_weight_pair : id_weight_pairs) {
    if (dedup_id_set.count(id_weight_pair.first)) continue;
    out_ids.push_back(id_weight_pair.first);
    out_cates.push_back(cate);
    out_weights.push_back(id_weight_pair.second);
    if (++cnt == sample_cnt) break;
  }
  TORCH_CHECK(cnt == sample_cnt, "cnt != sample_cnt, cnt = ", cnt,
              ", sample_cnt = ", sample_cnt,
              " dedup_id_set.size() = ", dedup_id_set.size());
}

void LocalDataResource::PutBackSample(std::vector<size_t>& cates,
                                      std::set<size_t>& dedup_id_set,
                                      size_t sample_cnt,
                                      std::vector<size_t>& out_ids,
                                      std::vector<size_t>& out_cates,
                                      std::vector<double>& out_weights) const {
  // init cate sampler
  std::vector<std::pair<size_t, double>> cate_weights;
  size_t total_set_size = 0;
  for (size_t cate : cates) {
    auto map_cate_weight_it = map_cate_weight_.find(cate);
    TORCH_CHECK(map_cate_weight_it != map_cate_weight_.end(), cate,
                " not in local data sample_tag");
    double weight = map_cate_weight_it->second;
    cate_weights.push_back(std::pair<size_t, double>(cate, weight));
    total_set_size += map_cate_id_weight_pairs_.find(cate)->second.size();
  }
  TORCH_CHECK(total_set_size >= sample_cnt + 1,
              "sample_cnt can't be greater than sample set size!");
  std::shared_ptr<WeightedCollection<size_t>> cate_sampler(
      WeightedCollectionFactory::Create<size_t>(put_back_));
  cate_sampler->Init(cate_weights);

  // sample
  size_t cnt = 0;
  while (cnt < sample_cnt) {
    size_t cate = cate_sampler->Sample().first;
    std::pair<size_t, double> id_weight =
        map_cate_sampler_.find(cate)->second->Sample();
    size_t id = id_weight.first;
    double weight = id_weight.second;
    if (dedup_id_set.count(id)) continue;
    out_ids.push_back(id);
    out_cates.push_back(cate);
    out_weights.push_back(weight);
    cnt++;
  }
}

Tensor LocalDataResource::ValidSampleIds(Tensor sample_ids_tensor,
                                         int64_t default_value) const {
  CounterReadLock;
  TORCH_CHECK(sampler_initialized_, "please init sampler!");
  Tensor output_t = at::empty(sample_ids_tensor.sizes(), at::kLong);
  auto input = sample_ids_tensor.const_data_ptr<int64_t>();
  auto output = output_t.mutable_data_ptr<int64_t>();
  const int N = sample_ids_tensor.numel();
  int miss_cnt = 0;
  for (int i = 0; i < N; ++i) {
    int64_t sample_id = input[i];
    auto id_index = map_id_index_.find(sample_id);
    if (id_index == map_id_index_.end()) {
      output[i] = default_value;
      ++miss_cnt;
    } else {
      output[i] = sample_id;
    }
  }
  VLOG(2) << "the count of miss sample_id: " << miss_cnt;
  return output_t;
}

std::vector<Tensor> LocalDataResource::ExtractFeature(
    Tensor sample_ids_tensor, std::vector<std::string> names,
    Tensor ragged_ranks, int64_t default_value) const {
  CounterReadLock;
  std::vector<Tensor> flat_tensors;
  FlatLocalData(names, ragged_ranks, flat_tensors);
  // generate line index refer to local line
  std::vector<std::pair<size_t, size_t>> all_table_ranges;
  size_t total_sample_cnt = sample_ids_tensor.numel();

  auto sample_ids = sample_ids_tensor.const_data_ptr<int64_t>();
  auto default_index = map_id_index_.find(default_value);
  for (size_t i = 0; i < total_sample_cnt; ++i) {
    size_t sample_id = sample_ids[i];
    auto id_index = map_id_index_.find(sample_id);
    if (id_index == map_id_index_.end()) {
      TORCH_CHECK(default_index != map_id_index_.end(), "sample_id ", sample_id,
                  " and default_value ", default_value,
                  " not in local data, local data size ", map_id_index_.size());
      id_index = default_index;
    }
    all_table_ranges.push_back(std::pair<size_t, size_t>(0, id_index->second));
  }

  std::vector<std::vector<Tensor>> tables;
  tables.push_back(flat_tensors);

  std::vector<Tensor> out_tensors;
  Pack(tables, ragged_ranks, all_table_ranges, out_tensors);
  return out_tensors;
}

void LocalDataResource::FlatLocalData(std::vector<std::string>& names,
                                      Tensor& ragged_ranks,
                                      std::vector<Tensor>& flat_tensors) const {
  std::vector<Tensor> values_t;
  std::vector<Tensor> splits_t;
  std::set<std::string> flat_names;
  for (size_t i = 0; i < names.size(); ++i) {
    std::string name = names[i];
    if (flat_names.count(name) != 0)
      continue;
    else
      flat_names.insert(name);
    auto map_feature_tensors_it = map_feature_tensors_.find(name);
    TORCH_CHECK(map_feature_tensors_it != map_feature_tensors_.end(),
                "LocalData hasn't feature: " + name);
    std::vector<std::vector<Tensor>> tuples = map_feature_tensors_it->second;
    for (size_t j = 0; j < tuples.size(); j++) {
      std::vector<Tensor> tensors = tuples[j];
      TORCH_CHECK(
          tensors.size() == ragged_ranks.const_data_ptr<int32_t>()[i + j] + 1,
          "innest tuple size not equal in feature: " + name);
      for (size_t k = 0; k < tensors.size(); k++) {
        if (k == 0)
          values_t.push_back(tensors[k]);
        else
          splits_t.push_back(tensors[k]);
      }
    }
  }
  Tensor batch_size_t = at::tensor(batch_size_, at::kInt);
  flat_tensors.push_back(batch_size_t);
  flat_tensors.insert(flat_tensors.end(), values_t.begin(), values_t.end());
  flat_tensors.insert(flat_tensors.end(), splits_t.begin(), splits_t.end());
}

void LocalDataResource::Pack(
    std::vector<std::vector<Tensor>>& batch_elements, Tensor& ragged_ranks_t,
    std::vector<std::pair<size_t, size_t>>& all_table_ranges,
    std::vector<Tensor>& out_tensors) const {
  auto ragged_ranks = ragged_ranks_t.const_data_ptr<int32_t>();
  std::vector<int> splits_sub_idx(ragged_ranks_t.numel());
  for (size_t i = 1; i < ragged_ranks_t.numel(); ++i) {
    splits_sub_idx[i] = splits_sub_idx[i - 1] + ragged_ranks[i - 1];
  }
  int num_tensors = ragged_ranks_t.numel();
  int num_splits =
      splits_sub_idx.back() + ragged_ranks[ragged_ranks_t.numel() - 1];
  int values_idx = 1;
  int splits_idx = values_idx + num_tensors;

  Tensor batch_size_t = at::empty({}, at::kInt);
  std::vector<Tensor> values_t(num_tensors);
  std::vector<Tensor> splits_t(num_splits);
  int total_batch_size = all_table_ranges.size();

  std::function<void(int)> pack_tensor = [&](int j) {
    int sub_idx = splits_sub_idx[j];
    // Do statistic
    std::vector<int> ragged_size(ragged_ranks[j] + 1, 0);
    for (auto table_index_pair : all_table_ranges) {
      std::vector<Tensor>& batch_element =
          batch_elements[table_index_pair.first];
      int begin = table_index_pair.second, end = begin + 1;
      for (int k = ragged_ranks[j] - 1; k >= 0; --k) {
        AT_DISPATCH_INDEX_TYPES(
            batch_element[splits_idx + sub_idx + k].scalar_type(),
            "LocalDataResourcePack1", [&]() {
              ragged_size[k + 1] += end - begin;
              auto current_splits = batch_element[splits_idx + sub_idx + k]
                                        .const_data_ptr<index_t>();
              begin = current_splits[begin];
              end = current_splits[end];
            });
      }
      ragged_size[0] += end - begin;
    }

    // Allocate tensors.
    for (int k = 0; k < ragged_ranks[j]; ++k) {
      auto scalar_type =
          batch_elements[0][splits_idx + sub_idx + k].scalar_type();
      AT_DISPATCH_INDEX_TYPES(scalar_type, "LocalDataResourcePack2", [&]() {
        splits_t[sub_idx + k] =
            at::empty({ragged_size[k + 1] + 1}, scalar_type);
        splits_t[sub_idx + k].mutable_data_ptr<index_t>()[0] = 0;
      });
    }

    auto scalar_type = batch_elements[0][values_idx + j].scalar_type();
    auto shape = batch_elements[0][values_idx + j].sizes().vec();
    shape[0] = ragged_size[0];
    values_t[j] = at::empty(shape, scalar_type);
    int dense_dim = 1;
    for (int i = 1; i < shape.size(); ++i) {
      dense_dim *= shape[i];
    }

    // Copy results.

    int next_offset = 0;
    for (auto table_index_pair : all_table_ranges) {
      std::vector<Tensor>& batch_element =
          batch_elements[table_index_pair.first];
      int begin = table_index_pair.second, end = begin + 1;
      int offset = next_offset;
      next_offset += end - begin;

      for (int k = ragged_ranks[j] - 1; k >= 0; --k) {
        AT_DISPATCH_INDEX_TYPES(
            batch_element[splits_idx + sub_idx + k].scalar_type(),
            "LocalDataResourcePack3", [&]() {
              auto splits = splits_t[sub_idx + k].mutable_data_ptr<index_t>();
              auto current_splits = batch_element[splits_idx + sub_idx + k]
                                        .const_data_ptr<index_t>();
              int splits_offset = splits[offset];
              int splits_begin = current_splits[begin];
              int splits_end = current_splits[end];
              for (int l = 1; l <= end - begin; ++l) {
                splits[offset + l] =
                    current_splits[begin + l] + splits_offset - splits_begin;
              }
              begin = splits_begin;
              end = splits_end;
              offset = splits_offset;
            });
      }

      AT_DISPATCH_ALL_TYPES(scalar_type, "LocalDataResourcePack4", [&]() {
        auto values = values_t[j].mutable_data_ptr<scalar_t>();
        auto current_values =
            batch_element[values_idx + j].mutable_data_ptr<scalar_t>();
        std::memcpy(&values[offset * dense_dim],
                    &current_values[begin * dense_dim],
                    sizeof(scalar_t) * (end - begin) * dense_dim);
      });
    }
  };

  BlockingCounter counter((num_tensors - 1) / kPackTensorBlock_ + 1);
  for (int job_begin = 0; job_begin < num_tensors;
       job_begin += kPackTensorBlock_) {
    int job_end = std::min(job_begin + kPackTensorBlock_, num_tensors);
    at::launch([job_begin, job_end, &counter, &pack_tensor]() {
      for (int j = job_begin; j < job_end; ++j) {
        pack_tensor(j);
      }
      counter.DecrementCount();
    });
  }

  batch_size_t.mutable_data_ptr<int32_t>()[0] = total_batch_size;
  counter.Wait();

  out_tensors.emplace_back(std::move(batch_size_t));
  std::move(values_t.begin(), values_t.end(), std::back_inserter(out_tensors));
  std::move(splits_t.begin(), splits_t.end(), std::back_inserter(out_tensors));
}

#undef CounterReadLock

}  // namespace data
}  // namespace recis
