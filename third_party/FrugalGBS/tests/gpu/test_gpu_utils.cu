#include <catch2/catch_test_macros.hpp>

#include "fgbs/gpu/gpu_utils.cuh"

#include <algorithm>

using namespace fgbs;

TEST_CASE("GPU utility helpers compute grid sizes", "[gpu][utils]") {
    REQUIRE(gpu::get_grid_size(256, 0) == 0);
    REQUIRE(gpu::get_grid_size(256, 1) == 1);
    REQUIRE(gpu::get_grid_size(256, 256) == 1);
    REQUIRE(gpu::get_grid_size(256, 257) == 2);
    REQUIRE(gpu::get_grid_size(128, 1024) == 8);
}

TEST_CASE("GPU runtime queries are valid when a GPU is present", "[gpu][utils]") {
    const auto default_ids_result = gpu::set_default_device_ids();
    if (!default_ids_result.ok()) {
        REQUIRE(gpu::get_device_ids().size() == 0);
        SKIP("No GPU available in runtime environment.");
    }
    REQUIRE(default_ids_result.ok());

    auto const& selected_ids = gpu::get_device_ids();
    const auto selected_count = selected_ids.size();

    REQUIRE(selected_count > 0);

    const auto current_id_result = gpu::get_current_device_id();
    REQUIRE(current_id_result.ok());
    const auto current_id = current_id_result.value();
    REQUIRE(std::find(selected_ids.begin(), selected_ids.end(), current_id) != selected_ids.end());

    const auto free_memory_result = gpu::get_free_memory();
    REQUIRE(free_memory_result.ok());
    REQUIRE(free_memory_result.value() > 0);

    const auto max_threads_result = gpu::get_max_threads();
    REQUIRE(max_threads_result.ok());
    REQUIRE(max_threads_result.value() > 0);
}

TEST_CASE("GPU device selection rejects duplicate IDs", "[gpu][utils]") {
    const auto default_ids_result = gpu::set_default_device_ids();
    if (!default_ids_result.ok()) {
        REQUIRE(gpu::get_device_ids().size() == 0);
        SKIP("No GPU available in runtime environment.");
    }

    auto const& selected_ids = gpu::get_device_ids();
    REQUIRE(!selected_ids.empty());

    const std::vector<int> duplicate_ids{selected_ids[0], selected_ids[0]};
    const auto ids_result = gpu::set_device_ids(duplicate_ids);
    REQUIRE(!ids_result.ok());
    REQUIRE(ids_result.error().find("duplicated") != std::string::npos);

    auto const& ids_after_failure = gpu::get_device_ids();
    REQUIRE(ids_after_failure == selected_ids);
}
