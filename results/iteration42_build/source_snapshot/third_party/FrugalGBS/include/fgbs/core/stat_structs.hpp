#ifndef FGBS_CORE_STAT_STRUCTS_HPP
#define FGBS_CORE_STAT_STRUCTS_HPP

#include "fgbs/core/type_utils.hpp"

namespace fgbs {

struct LinregressInfo {
    f64 correlation, correlation_error;
    f64 gradient, gradient_error;
    f64 intercept, intercept_error;
};

}

#endif
