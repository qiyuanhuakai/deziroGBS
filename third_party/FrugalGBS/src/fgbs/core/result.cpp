#include "fgbs/core/result.hpp"

#include <cstdlib>
#include <iostream>

namespace fgbs {

[[noreturn]] void panic(char const* message) {
    std::cerr << message << '\n';
    std::abort();
}

}
