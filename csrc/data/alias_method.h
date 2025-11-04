#pragma once

#include <map>
#include <string>
#include <vector>

#include "data/random.h"

namespace recis {
namespace data {

class AliasMethod {
 public:
  void Init(const std::vector<double>& weights);

  int64_t Next() const;

  size_t GetSize() const;

  std::string ShowData() const;

 private:
  std::vector<double> prob_;
  std::vector<int64_t> alias_;
  int64_t NextLong(int64_t n) const;
};

}  // namespace data
}  // namespace recis
