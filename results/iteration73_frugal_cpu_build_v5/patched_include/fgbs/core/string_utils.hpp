#ifndef FGBS_CORE_STRING_UTILS_HPP
#define FGBS_CORE_STRING_UTILS_HPP

#include "fgbs/core/io_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <ios>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fgbs {

std::string to_str(char const* str);

std::string to_str(std::string const& str);

template<typename... T>
std::string build_string(T&&... args) {
    std::stringstream ss;
    set_real_precision(ss);
    auto append = [&ss](auto&& arg) {
        using Arg = remove_cvref_t<decltype(arg)>;
        if constexpr (std::is_same_v<Arg, real>) {
            real integer;
            if (std::modf(arg, &integer) == real{0.0}) {
                ss << std::setprecision(1) << std::fixed << integer << std::defaultfloat;
                set_real_precision(ss);
            }
            else {
                ss << arg;
            }
        }
        else {
            ss << std::forward<decltype(arg)>(arg);
        }
    };
    (append(std::forward<T>(args)), ...);
    return ss.str();
}

template<typename STR>
usize str_to_usize(STR&& str) {
    return static_cast<usize>(std::stoull(to_str(str)));
}

template<typename STR>
isize str_to_isize(STR&& str) {
    return static_cast<isize>(std::stoll(to_str(str)));
}

template<typename STR>
real str_to_real(STR&& str) {
    #if FGBS_REAL_PRECISION < 2
    return std::stof(to_str(str));
    #else
    return std::stod(to_str(str));
    #endif
}

template<typename BEGIN, typename END, typename PREFIX, typename SUFFIX, typename DELIMITER>
std::string join_string(BEGIN begin, END end, PREFIX&& prefix, SUFFIX&& suffix, DELIMITER&& delimiter) {
    std::stringstream ss;
    ss << prefix;

    bool start = true;
    for (; begin != end; ++begin) {
        if (start) {
            start = false;
        }
        else {
            ss << delimiter;
        }
        ss << *begin;
    }

    ss << suffix;
    return ss.str();
}

/**
 * \brief Split string by specific character.
 **/
template<typename STR>
std::vector<std::string> split(STR&& str, char separator) {
    std::vector<std::string> results;
    auto start = str.begin();
    auto end = str.end();

    auto next = std::find(start, end, separator);
    while (next != end) {
        results.push_back(std::string(start, next));
        start = next + 1;
        next = std::find(start, end, separator);
    }
    results.push_back(std::string(start, next));

    return results;
}

template<typename STR>
std::string lowercase(STR&& str) {
    std::string lower = to_str(std::forward<STR>(str));
    std::transform(lower.begin(), lower.end(), lower.begin(), [](unsigned char c) { return std::tolower(c); });
    return lower;
}

template<typename STR>
std::string pluralize(STR&& str, usize count) {
    return count == 1 ? str : (to_str(str) + 's');
}

}

#endif
