#ifndef FGBS_GPU_COMMON_IMPL_CUH
#define FGBS_GPU_COMMON_IMPL_CUH

#include "fgbs/gpu/gpu_structs.cuh"
#include "fgbs/gpu/gpu_utils.cuh"

#include "fgbs/core/structs.hpp"
#include "fgbs/core/validation.hpp"
#include "fgbs/core/macro_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/thread_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <string>
#include <vector>

namespace fgbs {

namespace gpu {

__device__ constexpr usize device_bitstring_size(usize n) {
    return (n + usize{31}) / usize{32};
}

template<usize N>
__forceinline__ __device__ void device_flip_bit(Bitstring<N>* __restrict__ bitstring, usize index) {
    FGBS_INTERNAL_PTR_CAST(u8*, bitstring)[index / 8] ^= u8{1} << (index % 8);
}

template<usize N>
__forceinline__ __device__ i32 device_characteristic(Bitstring<N> const* __restrict__ x, Bitstring<N> const* __restrict__ y) {
    constexpr auto BITSTRING_SIZE = device_bitstring_size(N);
    i32 weight = 0;
    for (usize i = 0; i < BITSTRING_SIZE; ++i) {
        weight += __popc(FGBS_INTERNAL_PTR_CAST(u32 const*, x)[i] & FGBS_INTERNAL_PTR_CAST(u32 const*, y)[i]);
    }
    return 1 - 2 * (weight & 1);
}

__forceinline__ __device__ real device_clamped_prob(real prob) {
    return prob < 0.0 ? 0.0 : (prob > 1.0 ? 1.0 : prob);
}

__forceinline__ __device__ real device_parity_factorial(usize n) {
    isize result = 1;
    for (isize i = 1, m = static_cast<isize>(n); i <= m; ++i) {
        result *= -i;
    }
    return static_cast<real>(result);
}

__forceinline__ __device__ void device_kahan_add(real& sum, real& c, real value) {
    const auto y = value - c;
    const auto t = sum + y;
    c = (t - sum) - y;
    sum = t;
}

template<typename... DATA>
Result<void> validate_device_data(DATA const&... data) {
    std::string error;
    auto update_error = [&](auto const& e) {
        auto status_result = e.status();
        if (error.empty() && !status_result.ok()) {
            error = status_result.error();
        }
    };
    (update_error(data), ...);

    if (!error.empty()) {
        return Err{error};
    }
    return Ok<void>{};
}

}

}

#endif
