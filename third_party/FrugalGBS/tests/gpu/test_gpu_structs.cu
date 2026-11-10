#include <catch2/catch_test_macros.hpp>

#include "fgbs/gpu/gpu_structs.cuh"
#include "fgbs/gpu/gpu_utils.cuh"

#include <vector>

using namespace fgbs;

TEST_CASE("DeviceData<T> supports host-device roundtrips", "[gpu][structs]") {
    const auto default_ids_result = gpu::set_default_device_ids();
    if (!default_ids_result.ok()) {
        SKIP("No GPU available in runtime environment.");
    }
    REQUIRE(default_ids_result.ok());

    const std::vector<int> data{1, 2, 3, 4};
    gpu::DeviceData<int> d_data(data);
    REQUIRE(d_data.status().ok());

    const auto copy_result = d_data.copy_to_host();
    REQUIRE(copy_result.ok());
    REQUIRE(copy_result.value() == data);

    const auto memset_result = d_data.memset(0);
    REQUIRE(memset_result.ok());
    const auto memset_copy_result = d_data.copy_to_host();
    REQUIRE(memset_copy_result.ok());
    const std::vector<int> zeros{0, 0, 0, 0};
    REQUIRE(memset_copy_result.value() == zeros);
}
