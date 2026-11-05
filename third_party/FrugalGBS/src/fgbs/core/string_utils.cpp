#include "fgbs/core/string_utils.hpp"

#include <string>

namespace fgbs {

std::string to_str(char const* str) {
    return std::string(str);
}

std::string to_str(std::string const& str) {
    return str;
}

}
