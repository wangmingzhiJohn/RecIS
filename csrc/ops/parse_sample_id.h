#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <string>
#include <vector>

namespace recis {
namespace functional {
int64_t parse_sample_id(const std::vector<std::string>& raw_sample_ids,
                        bool is_max = true);
}
}  // namespace recis