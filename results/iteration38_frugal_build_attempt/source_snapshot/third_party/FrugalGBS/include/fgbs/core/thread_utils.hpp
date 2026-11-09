#ifndef FGBS_CORE_THREAD_UTILS_HPP
#define FGBS_CORE_THREAD_UTILS_HPP

#include "fgbs/core/result.hpp"
#include "fgbs/core/string_utils.hpp"

#include <atomic>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fgbs {

struct ThreadResultWrapper {
public:
    bool ok() const {
        return _ok.load(std::memory_order_acquire);
    }

    Result<void> get() const {
        if (ok()) {
            return Ok<void>{};
        }
        std::lock_guard<std::mutex> lock(_mutex);
        if (_error.empty()) {
            return Err{"Thread operation failed."};
        }
        return Err{_error};
    }

    template<typename RESULT>
    void set(RESULT const& result) {
        if (result.ok() || !ok()) {
            return;
        }

        std::lock_guard<std::mutex> lock(_mutex);
        if (!_ok.load(std::memory_order_relaxed)) {
            return;
        }
        _error = result.error();
        _ok.store(false, std::memory_order_release);
    }

private:
    std::atomic<bool> _ok{true};
    mutable std::mutex _mutex;
    std::string _error;
};

#define FGBS_THREAD_VALIDATE_OR_RETURN(THREAD_STATE, RESULT) \
do {                                                         \
    auto&& _fgbs_result = (RESULT);                          \
    if (!_fgbs_result.ok()) {                                \
        (THREAD_STATE).set(_fgbs_result);                    \
        return;                                              \
    }                                                        \
}                                                            \
while (false)

#define FGBS_THREAD_VALIDATE_OR_RETURN_CTX(THREAD_STATE, RESULT, PREFIX)                                       \
do {                                                                                                           \
    auto&& _fgbs_result = (RESULT);                                                                            \
    if (!_fgbs_result.ok()) {                                                                                  \
        (THREAD_STATE).set(fgbs::Result<void>{fgbs::Err{fgbs::build_string((PREFIX), _fgbs_result.error())}}); \
        return;                                                                                                \
    }                                                                                                          \
}                                                                                                              \
while (false)

#define FGBS_THREAD_VALIDATE_OR_BREAK(THREAD_STATE, RESULT) \
do {                                                        \
    auto&& _fgbs_result = (RESULT);                         \
    if (!_fgbs_result.ok()) {                               \
        (THREAD_STATE).set(_fgbs_result);                   \
        break;                                              \
    }                                                       \
}                                                           \
while (false)

#define FGBS_THREAD_VALIDATE_OR_BREAK_CTX(THREAD_STATE, RESULT, PREFIX)                                        \
do {                                                                                                           \
    auto&& _fgbs_result = (RESULT);                                                                            \
    if (!_fgbs_result.ok()) {                                                                                  \
        (THREAD_STATE).set(fgbs::Result<void>{fgbs::Err{fgbs::build_string((PREFIX), _fgbs_result.error())}}); \
        break;                                                                                                 \
    }                                                                                                          \
}                                                                                                              \
while (false)

void join_threads(std::vector<std::thread>& threads);

}

#endif
