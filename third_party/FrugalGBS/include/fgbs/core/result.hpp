#ifndef FGBS_CORE_RESULT_HPP
#define FGBS_CORE_RESULT_HPP

#include <string>
#include <utility>
#include <variant>

namespace fgbs {

[[noreturn]] void panic(char const* message);

template<typename T>
struct Ok {
    T value;
};

template<>
struct Ok<void> {};

struct Err {
    std::string message;
};

template<typename T>
struct [[nodiscard]] Result {
public:
    Result(Ok<T> ok) : _storage(std::in_place_index<0>, std::move(ok.value)) {}

    Result(Err err) : _storage(std::in_place_index<1>, std::move(err.message)) {}

    bool ok() const noexcept {
        return std::holds_alternative<T>(_storage);
    }

    T& value() {
        if (!ok()) {
            panic("Attempted to read Ok value from Err result.");
        }
        return std::get<T>(_storage);
    }

    T const& value() const {
        if (!ok()) {
            panic("Attempted to read Ok value from Err result.");
        }
        return std::get<T>(_storage);
    }

    std::string const& error() const {
        if (ok()) {
            panic("Attempted to read Err message from Ok result.");
        }
        return std::get<std::string>(_storage);
    }

private:
    std::variant<T, std::string> _storage;
};

template<>
struct [[nodiscard]] Result<void> {
public:
    Result(Ok<void>) : _ok(true) {}

    Result(Err err) : _ok(false), _error(std::move(err.message)) {}

    bool ok() const noexcept {
        return _ok;
    }

    std::string const& error() const {
        if (_ok) {
            panic("Attempted to read Err message from Ok result.");
        }
        return _error;
    }

private:
    bool _ok = false;
    std::string _error;
};

}

#endif
