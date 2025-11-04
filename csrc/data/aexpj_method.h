#pragma once

#include <map>
#include <string>
#include <vector>

#include "data/random.h"

namespace recis {
namespace data {

class AexpjMethod {
 public:
  struct compare {
    bool operator()(std::pair<double, size_t> a, std::pair<double, size_t> b) {
      return a.first > b.first;
    }
  };
  void Init(const std::vector<double>& weights);

  std::vector<size_t> NSample(size_t n) const;

  size_t GetSize() const;

  std::string ShowData() const;

 private:
  std::vector<double> weights_;
  std::vector<double> sum_weights_;
  double RandomUniform(double dMin, double dMax) const;
};

}  // namespace data
}  // namespace recis
