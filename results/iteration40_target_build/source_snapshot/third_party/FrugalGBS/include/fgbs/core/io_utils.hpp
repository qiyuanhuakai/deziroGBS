#ifndef FGBS_CORE_IO_UTILS_HPP
#define FGBS_CORE_IO_UTILS_HPP

#include "fgbs/core/type_utils.hpp"

#include <iomanip>
#include <limits>

namespace fgbs {

template<typename T>
void set_real_precision(T& ostream) {
    ostream << std::setprecision(std::numeric_limits<real>::max_digits10);
}

}

#endif
