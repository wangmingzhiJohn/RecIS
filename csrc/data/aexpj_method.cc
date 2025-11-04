#include "data/aexpj_method.h"

#include <stdint.h>

#include <cmath>
#include <queue>

#include "c10/util/Logging.h"

namespace recis {
namespace data {

void AexpjMethod::Init(const std::vector<double>& weights) {
  weights_.resize(weights.size());
  sum_weights_.resize(weights.size());
  for (size_t i = 0; i < weights_.size(); ++i) {
    if (weights[i] == 0.0) {
      LOG(ERROR) << "weight can't equal 0!";
    }
    weights_[i] = weights[i];
    if (i == 0) {
      sum_weights_[i] = weights_[i];
    } else {
      sum_weights_[i] = sum_weights_[i - 1] + weights_[i];
    }
  }
}  // Init

std::vector<size_t> AexpjMethod::NSample(size_t n) const {
  std::priority_queue<std::pair<double, size_t>,
                      std::vector<std::pair<double, size_t>>, compare>
      heap;
  double Xw = 0, Tw = 0, w_acc = 0;
  size_t i = 0;
  while (i < weights_.size()) {
    if (heap.size() < n) {
      double wi = weights_[i];
      double ui = RandomUniform(0, 1);
      double ki = pow(ui, 1 / wi);
      heap.push(std::pair<double, size_t>(ki, i));
      i++;
      if (heap.size() == n) w_acc = sum_weights_[i - 1];
      continue;
    }
    Tw = heap.top().first;
    double rd = RandomUniform(0, 1);
    Xw = log(rd) / log(Tw);
    i = std::lower_bound(sum_weights_.begin() + i, sum_weights_.end(),
                         Xw + w_acc) -
        sum_weights_.begin();
    if (i >= weights_.size()) break;
    w_acc = sum_weights_[i];
    double tw = pow(Tw, weights_[i]);
    double r2 = RandomUniform(tw, 1);
    double ki = pow(r2, 1 / weights_[i]);
    heap.pop();
    heap.push(std::pair<double, size_t>(ki, i));
  }

  std::vector<size_t> res;
  while (!heap.empty()) {
    res.push_back(heap.top().second);
    heap.pop();
  }
  return res;
}

size_t AexpjMethod::GetSize() const { return weights_.size(); }

double AexpjMethod::RandomUniform(double dMin, double dMax) const {
  double pRandom = ThreadLocalRandom();
  pRandom = pRandom * (dMax - dMin) + dMin;
  return pRandom;
}

std::string AexpjMethod::ShowData() const {
  std::string result = "weights_: {\n";
  for (auto& weight : weights_) {
    result += std::to_string(weight);
    result += "\n";
  }
  result += "}\n";

  result += "sum_weights_: {\n";
  for (auto& sum_weight : sum_weights_) {
    result += std::to_string(sum_weight);
    result += "\n";
  }
  result += "}\n";

  return result;
}

}  // namespace data
}  // namespace recis
