#pragma once
#include <atomic>

namespace recis {
namespace serialize {

class SizeCounter {
 public:
  std::atomic<int64_t> size_;
  SizeCounter() : size_(0) {};
  void InitCounter();
  void AddSize(const int64_t size);
  int64_t GetSize();
};

}  // namespace serialize
}  // namespace recis