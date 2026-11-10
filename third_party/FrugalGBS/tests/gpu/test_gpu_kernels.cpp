#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include "fgbs/data.hpp"
#include "fgbs/gpu/gpu_cumulants.cuh"
#include "fgbs/gpu/gpu_sampler.cuh"
#include "fgbs/gpu/gpu_stats.cuh"
#include "fgbs/gpu/gpu_utils.cuh"

#include <vector>

using namespace fgbs;

TEST_CASE("GPU kernel entry points run for a simple vacuum setup", "[gpu][kernels]") {
    const auto default_ids_result = gpu::set_default_device_ids();
    if (!default_ids_result.ok()) {
        SKIP("No GPU available in runtime environment.");
    }
    REQUIRE(default_ids_result.ok());

    constexpr usize N = 100;
    constexpr usize CUTOFF_ORDER = 1;
    constexpr usize SAMPLE_COUNT = 8;

    const auto cov = identity_matrix<f64>(2 * N);
    auto bitstring_layout_result = generate_bitstring_layout<N>(CUTOFF_ORDER);
    REQUIRE(bitstring_layout_result.ok());
    auto bitstring_layout = std::move(bitstring_layout_result.value());

    auto cumulant_layout_result = generate_cumulant_layout<N>(CUTOFF_ORDER);
    REQUIRE(cumulant_layout_result.ok());
    auto cumulant_layout = std::move(cumulant_layout_result.value());

    auto sublist_layout_result = generate_sublist_layout<N>(bitstring_layout, 1);
    REQUIRE(sublist_layout_result.ok());
    auto sublist_layout = std::move(sublist_layout_result.value());

    auto sampling_target_result = generate_sampling_target<N>(bitstring_layout, cumulant_layout, cov);
    REQUIRE(sampling_target_result.ok());
    auto sampling_target = std::move(sampling_target_result.value());

    auto correlator_vec_result = generate_correlator_vec<N>(bitstring_layout, cov);
    REQUIRE(correlator_vec_result.ok());
    auto correlator_vec = std::move(correlator_vec_result.value());

    const auto samples_result = gpu::generate_samples<N>(bitstring_layout, sampling_target, SAMPLE_COUNT, 4598u);
    REQUIRE(samples_result.ok());
    auto const& samples = samples_result.value();
    REQUIRE(samples.size() == SAMPLE_COUNT);

    const Bitstring<N> zero_bitstring{};
    for (auto const& sample : samples) {
        REQUIRE(sample == zero_bitstring);
    }

    const auto sample_correlator_sublists_result = gpu::estimate_sample_correlator_sublists<N>(sublist_layout, samples);
    REQUIRE(sample_correlator_sublists_result.ok());
    auto const& sample_correlator_sublists = sample_correlator_sublists_result.value();
    REQUIRE(sample_correlator_sublists.size() == sublist_layout.sublist_offset_vec[2]);
    REQUIRE(sublist_layout.get_sublist(sample_correlator_sublists, 0).size() == sublist_layout.sublist_size_vec[0]);
    REQUIRE(sublist_layout.get_sublist(sample_correlator_sublists, 0)[0] == Catch::Approx(1.0));

    const auto cumulants_result = gpu::compute_cumulants<N>(bitstring_layout, cumulant_layout, correlator_vec, 0, false);
    REQUIRE(cumulants_result.ok());
    auto const& cumulants = cumulants_result.value();
    REQUIRE(cumulants.size() == bitstring_layout.len);
    REQUIRE(cumulants[0] == Catch::Approx(sampling_target.correlator_cumulant_vec[0]));
}
