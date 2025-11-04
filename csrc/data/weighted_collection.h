#pragma once

#include <stddef.h>

#include <utility>
#include <vector>

namespace recis {
namespace data {

template <class T>
class WeightedCollection {
 public:
  virtual ~WeightedCollection() {}
  virtual bool Init(const std::vector<T>& ids,
                    const std::vector<double>& weights) = 0;
  virtual bool Init(
      const std::vector<std::pair<T, double>>& id_weight_pairs) = 0;
  virtual std::pair<T, double> Sample() const = 0;
  virtual std::vector<std::pair<T, double>> NSample(size_t n) const = 0;
  virtual size_t GetSize() const = 0;
  virtual std::pair<T, double> Get(size_t idx) const = 0;
  virtual double GetSumWeight() const = 0;
};

}  // namespace data
}  // namespace recis
