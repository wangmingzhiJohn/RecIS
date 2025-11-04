#pragma once
#include <stddef.h>

#include <utility>
#include <vector>

#include "data/fast_weighted_collection.h"
#include "data/reservoir_weighted_collection.h"
#include "data/weighted_collection.h"

namespace recis {
namespace data {

class WeightedCollectionFactory {
 public:
  template <typename T>
  static WeightedCollection<T>* Create(bool put_back) {
    if (put_back) {
      return new FastWeightedCollection<T>();
    } else {
      return new ReservoirWeightedCollection<T>();
    }
  }
};

}  // namespace data
}  // namespace recis
