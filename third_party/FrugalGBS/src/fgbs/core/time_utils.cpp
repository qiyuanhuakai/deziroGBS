#include "fgbs/core/time_utils.hpp"

#include <chrono>

namespace fgbs {

Clock::time_point time_now() {
    return Clock::now();
}

real time_interval(Clock::time_point start, Clock::time_point end) {
    return static_cast<real>(std::chrono::duration_cast<TimeUnit>(end - start).count()) * TimeUnit::period::num / TimeUnit::period::den;
}

real time_since(Clock::time_point start) {
    return time_interval(start, time_now());
}

}
