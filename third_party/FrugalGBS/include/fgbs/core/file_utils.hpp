#ifndef FGBS_CORE_FILE_UTILS_HPP
#define FGBS_CORE_FILE_UTILS_HPP

#include "fgbs/core/io_utils.hpp"
#include "fgbs/core/result.hpp"
#include "fgbs/core/slice.hpp"
#include "fgbs/core/string_utils.hpp"
#include "fgbs/core/type_utils.hpp"

#include <filesystem>
#include <fstream>
#include <ios>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace fgbs {

Result<void> create_directory(std::filesystem::path const& directory_path);

// FUNCTION = void(std::string const&)
template<typename FILENAME, typename FUNCTION>
Result<void> for_each_file_line(FILENAME&& file_name, FUNCTION&& f) {
    std::ifstream file(std::forward<FILENAME>(file_name));
    if (!file) {
        return Err{build_string("Could not open file \"", file_name, "\".")};
    }

    std::string line;
    while (getline(file, line)) {
        std::forward<FUNCTION>(f)(line);
    }

    if (file.bad()) {
        return Err{build_string("Failed while reading file \"", file_name, "\".")};
    }

    return Ok<void>{};
}

template<typename FILENAME>
Result<std::vector<std::string>> get_file_lines(FILENAME&& file_name) {
    std::vector<std::string> lines;
    auto read_result = for_each_file_line(std::forward<FILENAME>(file_name), [&](std::string const& line) { lines.push_back(line); });
    if (!read_result.ok()) {
        return Err{read_result.error()};
    }
    return Ok<std::vector<std::string>>{std::move(lines)};
}

// FUNCTION = void(std::ofstream& | std::stringstream& | std::string&)
template<typename FILENAME, typename FUNCTION>
Result<void> write_file(FILENAME&& file_name, std::ios::openmode mode, FUNCTION&& f) {
    static_assert(std::is_invocable_v<FUNCTION&, std::ofstream&> || std::is_invocable_v<FUNCTION&, std::stringstream&> || std::is_invocable_v<FUNCTION&, std::string&>, "write_file callback must accept std::ofstream&, std::stringstream&, or std::string&.");

    if constexpr (std::is_invocable_v<FUNCTION&, std::ofstream&>) {
        std::ofstream file(file_name, mode);
        if (!file) {
            return Err{build_string("Could not open file \"", file_name, "\" for writing.")};
        }

        set_real_precision(file);
        std::forward<FUNCTION>(f)(file);

        if (!file) {
            return Err{build_string("Failed while writing file \"", file_name, "\".")};
        }

        return Ok<void>{};
    }
    else if constexpr (std::is_invocable_v<FUNCTION&, std::stringstream&>) {
        std::stringstream ss;
        set_real_precision(ss);
        std::forward<FUNCTION>(f)(ss);

        std::ofstream file(file_name, mode);
        if (!file) {
            return Err{build_string("Could not open file \"", file_name, "\" for writing.")};
        }

        file << ss.str();
        if (!file) {
            return Err{build_string("Failed while writing file \"", file_name, "\".")};
        }

        return Ok<void>{};
    }
    else if constexpr (std::is_invocable_v<FUNCTION&, std::string&>) {
        std::string str;
        std::forward<FUNCTION>(f)(str);

        std::ofstream file(file_name, mode);
        if (!file) {
            return Err{build_string("Could not open file \"", file_name, "\" for writing.")};
        }

        set_real_precision(file);
        file << str;

        if (!file) {
            return Err{build_string("Failed while writing file \"", file_name, "\".")};
        }

        return Ok<void>{};
    }
}

template<typename T>
void write_data(std::ofstream& file, Slice<T> data) {
    static_assert(std::is_trivially_copyable_v<T>, "Binary serialization requires trivially copyable element types.");
    file.write(reinterpret_cast<char const*>(data.data()), sizeof(T) * data.size());
}

template<typename T>
void serialize_internal(std::ofstream& file, std::vector<T> const& vec) {
    static_assert(std::is_trivially_copyable_v<T>, "serialize<T> requires T to be trivially copyable.");
    write_data(file, Slice<T>{vec.data(), vec.size()});
}

template<typename T, typename FILENAME>
Result<void> serialize(FILENAME&& file_name, T&& t) {
    return write_file(std::forward<FILENAME>(file_name), std::ios::out | std::ios::binary, [&t](std::ofstream& file) { serialize_internal(file, t); });
}

template<typename T, typename FILENAME>
Result<std::vector<T>> deserialize(FILENAME&& file_name) {
    static_assert(std::is_trivially_copyable_v<T>, "deserialize<T> requires T to be trivially copyable.");

    std::ifstream file(file_name, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file) {
        return Err{build_string("Could not open binary file \"", file_name, "\".")};
    }

    const auto byte_count = file.tellg();
    if (byte_count < 0) {
        return Err{build_string("Could not determine size of binary file \"", file_name, "\".")};
    }

    const auto byte_count_usize = static_cast<usize>(byte_count);
    if (sizeof(T) > 0 && byte_count_usize % sizeof(T) != 0) {
        return Err{build_string("Binary file \"", file_name, "\" size (", byte_count_usize, ") is not aligned to element size (", sizeof(T), ").")};
    }

    std::vector<T> values(byte_count_usize / sizeof(T));
    if (values.empty()) {
        return Ok<std::vector<T>>{std::move(values)};
    }

    file.seekg(0, std::ios::beg);
    file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(byte_count_usize));
    if (!file) {
        return Err{build_string("Failed to read binary data from file \"", file_name, "\".")};
    }

    return Ok<std::vector<T>>{std::move(values)};
}

}

#endif
