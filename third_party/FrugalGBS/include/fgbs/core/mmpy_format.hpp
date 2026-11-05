#ifndef FGBS_CORE_MMPY_FORMAT_HPP
#define FGBS_CORE_MMPY_FORMAT_HPP

#include "fgbs/core/io_utils.hpp"
#include "fgbs/core/macro_utils.hpp"
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

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR0(MMPY, PTR) \
template<typename T>                                     \
std::string MMPY##_format(T PTR v) {                     \
    return v == nullptr ? "null" : *v;                   \
}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR(PTR) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR0(mm, PTR)    \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR0(py, PTR)

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR(*)
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR(const*)

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE0(MMPY, T, REF, VALUE) std::string MMPY##_format(T REF v);

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE1(MMPY, T, LV, RV) \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE0(MMPY, T, const&, LV)     \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE0(MMPY, T, &&, RV)

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE(T, LV, RV) \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE1(mm, T, LV, RV)    \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE1(py, T, LV, RV)

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SCALAR(T, VALUE) FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE(T, VALUE, VALUE)

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(T) FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SCALAR(T, std::to_string(v))

FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(short)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(unsigned short)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(int)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(unsigned)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(long)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(unsigned long)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(long long)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT(unsigned long long)

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT0(MMPY, T, VALUE)  \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE0(MMPY, T, const&, VALUE) \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE0(MMPY, T, &&, VALUE)

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT(T)                      \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT0(mm, T, mm_precision_string(v)) \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT0(py, T, py_precision_string(v))

FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT(float)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT(double)
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT(long double)

FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SCALAR(bool, v ? "True" : "False")

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING0(MMPY, T, REF, DELIM, VALUE) std::string MMPY##_format(T REF v);

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING1(MMPY, T, DELIM, LV, RV) \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING0(MMPY, T, const&, DELIM, LV)     \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING0(MMPY, T, &&, DELIM, RV)

#define FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING(T, LV, RV)    \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING1(mm, T, "\"", LV, RV) \
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING1(py, T, "'", LV, RV)

FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING(char, std::string{v}, std::string{v})
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING(char*, std::string(v), std::string(v))
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING(char const*, std::string(v), std::string(v))
FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING(std::string, v, std::move(v))

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE0(MMPY, TEMPLATE, TYPE, REF, PRE, START, DELIM, POST, END) \
template<TEMPLATE>                                                                                          \
std::string MMPY##_format(TYPE REF v) {                                                                     \
    PRE;                                                                                                    \
    std::string str = START;                                                                                \
    bool begin = true;                                                                                      \
    for (auto const& e : v) {                                                                               \
        if (begin) {                                                                                        \
            begin = false;                                                                                  \
        }                                                                                                   \
        else {                                                                                              \
            str += DELIM;                                                                                   \
        }                                                                                                   \
        str += MMPY##_format(e);                                                                            \
    }                                                                                                       \
    POST;                                                                                                   \
    str += END;                                                                                             \
    return str;                                                                                             \
}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE1(MMPY, TEMPLATE, TYPE, PRE, START, DELIM, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE0(MMPY, TEMPLATE, TYPE, const&, PRE, START, DELIM, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE0(MMPY, TEMPLATE, TYPE, &&, PRE, START, DELIM, POST, END)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE2(MMPY, TYPE, PRE, START, DELIM, POST, END) FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE1(MMPY, typename T, TYPE<T>, PRE, START, DELIM, POST, END)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ARRAY0(MMPY, REF, PRE, START, DELIM, POST, END) \
template<typename T, size_t N>                                                           \
std::string MMPY##_format(std::array<T, N> REF v) {                                      \
    PRE;                                                                                 \
    std::string str = START;                                                             \
    bool begin = true;                                                                   \
    for (auto const& e : v) {                                                            \
        if (begin) {                                                                     \
            begin = false;                                                               \
        }                                                                                \
        else {                                                                           \
            str += DELIM;                                                                \
        }                                                                                \
        str += MMPY##_format(e);                                                         \
    }                                                                                    \
    POST;                                                                                \
    str += END;                                                                          \
    return str;                                                                          \
}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ARRAY1(MMPY, PRE, START, DELIM, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ARRAY0(MMPY, const&, PRE, START, DELIM, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ARRAY0(MMPY, &&, PRE, START, DELIM, POST, END)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_LIST(MMPY, PRE, START, DELIM, POST, END)           \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ARRAY1(MMPY, PRE, START, DELIM, POST, END)                 \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE2(MMPY, std::vector, PRE, START, DELIM, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE2(MMPY, std::list, PRE, START, DELIM, POST, END)

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_LIST(mm, {}, "{", ",", {}, "}")
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_LIST(py, {}, "(", ", ", if (v.size() == 1) str += ",", ")")

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SET(MMPY, PRE, START, DELIM, POST, END)         \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE2(MMPY, std::set, PRE, START, DELIM, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE2(MMPY, std::unordered_set, PRE, START, DELIM, POST, END)

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SET(mm, {}, "{", ",", {}, "}")
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SET(py, if (v.empty()) return "set()", "{", ", ", {}, "}")

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP0(MMPY, TYPE, REF, PRE, START, DELIM, SEP, POST, END) \
template<typename K, typename V>                                                                  \
std::string MMPY##_format(TYPE<K, V> REF v) {                                                     \
    std::string str = START;                                                                      \
    bool begin = true;                                                                            \
    for (auto const& e : v) {                                                                     \
        if (begin) {                                                                              \
            begin = false;                                                                        \
        }                                                                                         \
        else {                                                                                    \
            str += DELIM;                                                                         \
        }                                                                                         \
        str += MMPY##_format(e.first);                                                            \
        str += SEP;                                                                               \
        str += MMPY##_format(e.second);                                                           \
    }                                                                                             \
    str += END;                                                                                   \
    return str;                                                                                   \
}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP1(MMPY, TYPE, PRE, START, DELIM, SEP, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP0(MMPY, TYPE, const&, PRE, START, DELIM, SEP, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP0(MMPY, TYPE, &&, PRE, START, DELIM, SEP, POST, END)

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP(MMPY, PRE, START, DELIM, SEP, POST, END)    \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP1(MMPY, std::map, PRE, START, DELIM, SEP, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP1(MMPY, std::unordered_map, PRE, START, DELIM, SEP, POST, END)

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP(mm, {}, "<|", ",", "->", {}, "|>")
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP(py, {}, "{", ", ", ": ", {}, "}")

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE0(MMPY, REF, PRE, START, DELIM, POST, END) \
template<typename... T>                                                                  \
std::string MMPY##_format(std::tuple<T...> REF v) {                                      \
    PRE;                                                                                 \
    std::string str = START;                                                             \
    bool begin = true;                                                                   \
    auto f = [&](auto const& e) {                                                        \
        if (begin) {                                                                     \
            begin = false;                                                               \
        }                                                                                \
        else {                                                                           \
            str += DELIM;                                                                \
        }                                                                                \
        str += MMPY##_format(e);                                                         \
    };                                                                                   \
    apply_ordered(f, v);                                                                 \
    POST;                                                                                \
    str += END;                                                                          \
    return str;                                                                          \
}

#define FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE(MMPY, PRE, START, DELIM, POST, END)  \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE0(MMPY, const&, PRE, START, DELIM, POST, END) \
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE0(MMPY, &&, PRE, START, DELIM, POST, END)

FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE(mm, {}, "{", ",", {}, "}")
FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE(py, {}, "(", ", ", if (std::tuple_size<std::tuple<T...>>::value == 1) str += ",", ")")

}

#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_FLOAT0
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_INT
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SCALAR
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE0
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_SIMPLE1
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING0
#undef FGBS_INTERNAL_DECLARE_MMPY_FORMAT_STRING1
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE0
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE1
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_ITERABLE2
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_LIST
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP0
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_MAP1
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_PTR0
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_SET
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE
#undef FGBS_INTERNAL_DEFINE_MMPY_FORMAT_TUPLE0

#endif
