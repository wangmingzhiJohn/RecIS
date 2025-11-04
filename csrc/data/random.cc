#include "data/random.h"

namespace recis {
namespace data {

namespace {
static thread_local std::default_random_engine e(time(0));
static thread_local std::uniform_real_distribution<double> u(0., 1.);
}  // namespace

double ThreadLocalRandom() { return u(e); }

}  // namespace data
}  // namespace recis
