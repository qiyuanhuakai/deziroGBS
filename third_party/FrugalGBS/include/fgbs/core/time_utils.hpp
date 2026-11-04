#ifndef FGBS_CORE_TIME_UTILS_HPP
#define FGBS_CORE_TIME_UTILS_HPP

#include "fgbs/core/type_utils.hpp"

#include <chrono>

namespace fgbs {

using Clock = std::chrono::steady_clock;
using TimeUnit = std::chrono::nanoseconds;

/**
 * \brief Get current clock time.
 **/
Clock::time_point time_now();

/**
 * \brief Get time between start and end in seconds.
 **/
real time_interval(Clock::time_point start, Clock::time_point end);

/**
 * \brief Get time since start in seconds.
 **/
real time_since(Clock::time_point start);

// FUNCTION = void()
template<typename FUNCTION>
real time_of(FUNCTION&& f) {
    const auto start_time = time_now();
    f();
    return time_since(start_time);
}

}

#endif
