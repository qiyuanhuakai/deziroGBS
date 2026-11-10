#ifndef FGBS_GPU_STRUCTS_CUH
#define FGBS_GPU_STRUCTS_CUH

#include "fgbs/core/result.hpp"
#include "fgbs/core/slice.hpp"
#include "fgbs/core/string_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <cuda_runtime.h>

#include <array>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fgbs {

namespace gpu {

inline void set_cuda_status(Result<void>& status, char const* operation, cudaError_t code) {
    if (status.ok() && code != cudaSuccess) {
        (void)(status = Result<void>{Err{build_string(operation, " failed: ", cudaGetErrorString(code))}});
    }
}

inline void set_status_from(Result<void>& status, Result<void> const& result) {
    if (status.ok() && !result.ok()) {
        (void)(status = Result<void>{Err{result.error()}});
    }
}

template<typename T>
struct DeviceData {
public:
    typedef std::vector<T> Host;

    DeviceData() = delete;

    DeviceData(usize size) : _ptr{nullptr}, _size{size}, _status{Ok<void>{}} {
        auto const code = cudaMalloc(&_ptr, bytes());
        if (code != cudaSuccess) {
            _ptr = nullptr;
            _size = 0;
            set_cuda_status(_status, "cudaMalloc", code);
        }
    }

    DeviceData(usize size, u8 value) : DeviceData(size) {
        set_status_from(_status, memset(value));
    }

    DeviceData(T const* h_ptr, usize size) : DeviceData(size) {
        set_status_from(_status, copy_from_host(h_ptr));
    }

    DeviceData(Slice<T> slice) : DeviceData(slice.ptr, slice.len) {}

    DeviceData(SliceMut<T> slice) : DeviceData(slice.ptr, slice.len) {}

    template<typename VECTOR>
    DeviceData(VECTOR&& vec) : DeviceData(vec.data(), vec.size()) {}

    DeviceData(DeviceData const& other) = delete;

    DeviceData(DeviceData&& other) : _ptr{std::exchange(other._ptr, nullptr)}, _size{std::exchange(other._size, 0)}, _status{std::exchange(other._status, Result<void>{Ok<void>{}})} {}

    ~DeviceData() {
        if (_ptr != nullptr) {
            cudaFree(_ptr);
        }
    }

    DeviceData& operator=(DeviceData const& other) = delete;

    DeviceData& operator=(DeviceData&& other) {
        if (this != &other) {
            if (_ptr != nullptr) {
                cudaFree(_ptr);
            }
            _ptr = std::exchange(other._ptr, nullptr);
            _size = std::exchange(other._size, 0);
            (void)(_status = std::exchange(other._status, Result<void>{Ok<void>{}}));
        }
        return *this;
    }

    T* ptr() const {
        return _ptr;
    }

    usize size() const {
        return _size;
    }

    usize bytes() const {
        return sizeof(T) * _size;
    }

    Result<void> memset(u8 value) {
        return run_cuda("cudaMemset", cudaMemset(_ptr, value, bytes()));
    }

    Result<void> copy_from_host(T const* h_ptr) {
        return run_cuda("cudaMemcpy(HostToDevice)", cudaMemcpy(_ptr, h_ptr, bytes(), cudaMemcpyHostToDevice));
    }

    template<typename VECTOR>
    Result<void> copy_from_host(VECTOR&& vec) {
        return copy_from_host(vec.data());
    }

    Result<void> copy_to_host(T* h_ptr) {
        return run_cuda("cudaMemcpy(DeviceToHost)", cudaMemcpy(h_ptr, _ptr, bytes(), cudaMemcpyDeviceToHost));
    }

    Result<void> copy_to_host(Host& vec) {
        if (vec.size() != _size) {
            vec.resize(_size);
        }
        return copy_to_host(vec.data());
    }

    Result<Host> copy_to_host() {
        Host vec(_size);
        auto copy_result = copy_to_host(vec);
        if (!copy_result.ok()) {
            return Err{copy_result.error()};
        }
        return Ok<Host>{std::move(vec)};
    }

    Result<void> status() const {
        return _status;
    }

private:
    Result<void> ready_status() const {
        if (!_status.ok() || _ptr == nullptr) {
            return status();
        }
        return Ok<void>{};
    }

    Result<void> run_cuda(char const* operation, cudaError_t code) {
        auto ready_result = ready_status();
        if (!ready_result.ok()) {
            return ready_result;
        }
        set_cuda_status(_status, operation, code);
        return status();
    }

    T* _ptr;
    usize _size;
    Result<void> _status;
};

}

}

#endif
