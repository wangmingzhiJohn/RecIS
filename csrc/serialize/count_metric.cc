#include "serialize/count_metric.h"

namespace recis {
namespace serialize {
void SizeCounter::AddSize(const int64_t size) { size_.fetch_add(size); }
int64_t SizeCounter::GetSize() { return size_.load(); }
void SizeCounter::InitCounter() { size_.store(0); }
}  // namespace serialize
}  // namespace recis