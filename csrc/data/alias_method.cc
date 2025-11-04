#include "data/alias_method.h"

#include <stdint.h>

namespace recis {
namespace data {

void AliasMethod::Init(const std::vector<double>& weights) {
  prob_.resize(weights.size());
  alias_.resize(weights.size());
  std::vector<int64_t> small, large;
  std::vector<double> weights_(weights);
  double avg = 1 / static_cast<double>(weights_.size());
  for (size_t i = 0; i < weights_.size(); i++) {
    if (weights_[i] > avg) {
      large.push_back(i);
    } else {
      small.push_back(i);
    }
  }

  int64_t less, more;
  while (large.size() > 0 && small.size() > 0) {
    less = small.back();
    small.pop_back();
    more = large.back();
    large.pop_back();
    prob_[less] = weights_[less] * weights_.size();
    alias_[less] = more;
    weights_[more] = weights_[more] + weights_[less] - avg;
    if (weights_[more] > avg) {
      large.push_back(more);
    } else {
      small.push_back(more);
    }

  }  // while (large.size() > 0 && small.size() > 0)
  while (small.size() > 0) {
    less = small.back();
    small.pop_back();
    prob_[less] = 1.0;
  }

  while (large.size() > 0) {
    more = large.back();
    large.pop_back();
    prob_[more] = 1.0;
  }
}  // Init

int64_t AliasMethod::Next() const {
  int64_t column = NextLong(prob_.size());
  bool coinToss = ThreadLocalRandom() < prob_[column];
  return coinToss ? column : alias_[column];
}

size_t AliasMethod::GetSize() const { return prob_.size(); }

int64_t AliasMethod::NextLong(int64_t n) const {
  return floor(n * ThreadLocalRandom());
}

std::string AliasMethod::ShowData() const {
  std::string result = "prob: {\n";
  for (auto& prob : prob_) {
    result += std::to_string(prob);
    result += "\n";
  }
  result += "}\n";

  result += "alias: {\n";
  for (auto& alias : alias_) {
    result += std::to_string(alias);
    result += "\n";
  }
  result += "}\n";

  return result;
}

}  // namespace data
}  // namespace recis
