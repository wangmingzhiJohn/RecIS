#include <torch/extension.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <vector>

namespace recis {
namespace functional {

int64_t parse_sample_id(const std::vector<std::string>& raw_sample_ids,
                        bool is_max = true) {
  std::vector<int64_t> timestamps;
  timestamps.reserve(raw_sample_ids.size());

  for (const auto& id_str : raw_sample_ids) {
    size_t last_pos = id_str.find_last_of('^');
    std::string last_token =
        (last_pos == std::string::npos) ? id_str : id_str.substr(last_pos + 1);
    int64_t ts = std::stoll(last_token);
    timestamps.push_back(ts);
  }
  if (is_max) {
    return *std::max_element(timestamps.begin(), timestamps.end());
  } else {
    return *std::min_element(timestamps.begin(), timestamps.end());
  }
}

}  // namespace functional
}  // namespace recis
