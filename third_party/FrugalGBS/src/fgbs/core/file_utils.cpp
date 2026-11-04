#include "fgbs/core/file_utils.hpp"

#include <filesystem>

namespace fgbs {

Result<void> create_directory(std::filesystem::path const& directory_path) {
    std::error_code ec;
    std::filesystem::create_directories(directory_path, ec);
    if (ec) {
        return Err{build_string("Could not create directory \"", directory_path.string(), "\": ", ec.message())};
    }
    return Ok<void>{};
}

}
