#pragma once
#include <c10/util/Logging.h>

#include <utility>
#include <vector>

#include "data/aexpj_method.h"
#include "data/weighted_collection.h"

namespace recis {
namespace data {

template <class T>
class ReservoirWeightedCollection : public WeightedCollection<T> {
 public:
  bool Init(const std::vector<T>& ids,
            const std::vector<double>& weights) override;

  bool Init(const std::vector<std::pair<T, double>>& id_weight_pairs) override;

  std::pair<T, double> Sample() const override;

  std::vector<std::pair<T, double>> NSample(size_t n) const override;

  size_t GetSize() const override;

  std::pair<T, double> Get(size_t idx) const override;

  double GetSumWeight() const override;

  const std::vector<T>& GetIds() const;

  const std::vector<double>& GetWeights() const;

 private:
  std::vector<T> ids_;
  std::vector<double> weights_;
  AexpjMethod aexpj_;
  double sum_weight_;
};

template <class T>
bool ReservoirWeightedCollection<T>::Init(const std::vector<T>& ids,
                                          const std::vector<double>& weights) {
  if (ids.size() != weights.size()) {
    return false;
  }
  ids_.resize(ids.size());
  weights_.resize(weights.size());
  sum_weight_ = 0.0;
  for (size_t i = 0; i < weights.size(); i++) {
    sum_weight_ += weights[i];
    ids_[i] = ids[i];
    weights_[i] = weights[i];
  }
  std::vector<double> norm_weights(weights);
  // for (size_t i = 0; i < norm_weights.size(); i++) {
  //   norm_weights[i] /= sum_weight_;
  // }
  aexpj_.Init(norm_weights);
  return true;
}

template <class T>
bool ReservoirWeightedCollection<T>::Init(
    const std::vector<std::pair<T, double>>& id_weight_pairs) {
  ids_.resize(id_weight_pairs.size());
  weights_.resize(id_weight_pairs.size());
  sum_weight_ = 0.0;
  for (size_t i = 0; i < id_weight_pairs.size(); i++) {
    sum_weight_ += id_weight_pairs[i].second;
    ids_[i] = id_weight_pairs[i].first;
    weights_[i] = id_weight_pairs[i].second;
  }
  std::vector<double> norm_weights(weights_);
  // for (size_t i = 0; i < norm_weights.size(); i++) {
  //   norm_weights[i] /= sum_weight_;
  // }
  aexpj_.Init(norm_weights);
  return true;
}

template <class T>
std::pair<T, double> ReservoirWeightedCollection<T>::Sample() const {
  int64_t column = aexpj_.NSample(1).front();
  std::pair<T, double> id_weight_pair(ids_[column], weights_[column]);
  return id_weight_pair;
}

template <class T>
std::vector<std::pair<T, double>> ReservoirWeightedCollection<T>::NSample(
    size_t n) const {
  if (ids_.size() < n) {
    LOG(ERROR)
        << "NSample requires ids_ size greater than or equal sample_cnt: " << n;
  }
  std::vector<size_t> columns = aexpj_.NSample(n);
  std::vector<std::pair<T, double>> res;
  for (auto column : columns) {
    res.push_back(std::pair<T, double>(ids_[column], weights_[column]));
  }
  return res;
}

template <class T>
size_t ReservoirWeightedCollection<T>::GetSize() const {
  return ids_.size();
}

template <class T>
std::pair<T, double> ReservoirWeightedCollection<T>::Get(size_t idx) const {
  if (idx < ids_.size()) {
    std::pair<T, double> id_weight_pair(ids_[idx], weights_[idx]);
    return id_weight_pair;
  } else {
    // LOG(ERROR) << "idx out of boundary";
    return std::pair<T, double>();
  }
}

template <class T>
double ReservoirWeightedCollection<T>::GetSumWeight() const {
  return sum_weight_;
}

template <class T>
const std::vector<T>& ReservoirWeightedCollection<T>::GetIds() const {
  return ids_;
}

template <class T>
const std::vector<double>& ReservoirWeightedCollection<T>::GetWeights() const {
  return weights_;
}

}  // namespace data
}  // namespace recis
