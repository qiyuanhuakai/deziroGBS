#ifndef FGBS_CORE_MACRO_UTILS_HPP
#define FGBS_CORE_MACRO_UTILS_HPP

namespace fgbs {

#define FGBS_INTERNAL_STRINGIFY(X) #X

#define FGBS_INTERNAL_L_PAR (
#define FGBS_INTERNAL_R_PAR )

#define FGBS_INTERNAL_EXPAND(...) __VA_ARGS__

#define FGBS_INTERNAL_PTR_CAST(T, X) reinterpret_cast<T>(X)

}

#endif
