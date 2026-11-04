#ifndef FGBS_CORE_SLICE_HPP
#define FGBS_CORE_SLICE_HPP

#include "fgbs/core/type_utils.hpp"

namespace fgbs {

template<typename T>
struct Slice {
    using value_type = T;

    T const* const ptr;
    const usize len;

    Slice() = delete;

    Slice(T const* ptr, usize len) : ptr{ptr}, len{len} {}

    T const* data() const {
        return ptr;
    }

    usize size() const {
        return len;
    }

    T const& operator[](usize index) const {
        return ptr[index];
    }
};

template<typename T>
struct SliceMut {
    using value_type = T;

    T* const ptr;
    const usize len;

    SliceMut() = delete;

    SliceMut(T* ptr, usize len) : ptr{ptr}, len{len} {}

    T* data() {
        return ptr;
    }

    T const* data() const {
        return ptr;
    }

    usize size() const {
        return len;
    }

    T& operator[](usize index) {
        return ptr[index];
    }

    T const& operator[](usize index) const {
        return ptr[index];
    }
};

}

#endif
