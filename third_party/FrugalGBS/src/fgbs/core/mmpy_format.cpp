#include "fgbs/core/io_utils.hpp"
#include "fgbs/core/macro_utils.hpp"
#include "fgbs/core/mmpy_format.hpp"
#include "fgbs/core/tuple_utils.hpp"

#include <array>
#include <list>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fgbs {

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE0(MMPY, T, REF, VALUE) \
    std::string MMPY##_format(T REF v) {                              \
    return VALUE;                                                     \
}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE1(MMPY, T, LV, RV) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE0(MMPY, T, const&, LV)     \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE0(MMPY, T, &&, RV)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE(T, LV, RV) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE1(mm, T, LV, RV)    \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE1(py, T, LV, RV)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SCALAR(T, VALUE) FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE(T, VALUE, VALUE)

// Modified from https://stackoverflow.com/a/16606128/16920912
namespace {

template<typename T>
std::string py_precision_string(T&& v) {
    std::stringstream ss;
    set_real_precision(ss);
    ss << v;
    return ss.str();
}

template<typename T>
std::string mm_precision_string(T&& v) {
    auto str = py_precision_string(std::forward<T>(v));
    auto e_index = str.find('e');
    if (e_index == std::string::npos) {
        return str;
    }
    std::stringstream ss;
    ss << str.substr(0, e_index) << "*10^" << std::stoi(str.substr(e_index + 1, str.size()));
    return ss.str();
}

}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(T) FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SCALAR(T, std::to_string(v))

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(short)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(unsigned short)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(int)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(unsigned)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(long)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(unsigned long)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(long long)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT(unsigned long long)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT0(MMPY, T, VALUE)  \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE0(MMPY, T, const&, VALUE) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE0(MMPY, T, &&, VALUE)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT(T)                      \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT0(mm, T, mm_precision_string(v)) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT0(py, T, py_precision_string(v))

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT(float)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT(double)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT(long double)

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SCALAR(bool, v ? "True" : "False")

// Modified from https://stackoverflow.com/a/7369771/16920912
namespace {

std::string unescape_char(char ch) {
    if (isprint(ch) && ch != '\'' && ch != '\"' && ch != '\\' && ch != '\?') {
        return std::string{ch};
    }

    switch (ch) {
    case '\a':
        return "\\a";
    case '\b':
        return "\\b";
    case '\f':
        return "\\f";
    case '\n':
        return "\\n";
    case '\r':
        return "\\r";
    case '\t':
        return "\\t";
    case '\v':
        return "\\v";
    case '\\':
        return "\\\\";
    case '\'':
        return "\\'";
    case '\"':
        return "\\\"";
    case '\?':
        return "\\\?";
    default:
        return std::string{ch};
    }
}

template<typename STRING>
std::string unescape(STRING&& str) {
    std::string result;
    for (char ch : str) {
        result += unescape_char(ch);
    }
    return result;
}

}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING0(MMPY, T, REF, DELIM, VALUE) \
    std::string MMPY##_format(T REF v) {                                     \
    std::string str = DELIM;                                                 \
    str += unescape(VALUE);                                                  \
    str += DELIM;                                                            \
    return str;                                                              \
}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING1(MMPY, T, DELIM, LV, RV) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING0(MMPY, T, const&, DELIM, LV)     \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING0(MMPY, T, &&, DELIM, RV)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING(T, LV, RV)    \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING1(mm, T, "\"", LV, RV) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING1(py, T, "'", LV, RV)

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING(char, std::string{v}, std::string{v})
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING(char*, std::string(v), std::string(v))
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING(char const*, std::string(v), std::string(v))
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING(std::string, v, std::move(v))

#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_FLOAT0
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_INT
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SCALAR
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE0
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SIMPLE1
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING0
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_STRING1

}
