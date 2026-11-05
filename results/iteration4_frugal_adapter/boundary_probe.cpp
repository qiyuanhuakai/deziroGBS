// Directly compare the pinned upstream branch operator and trace density
// at threshold boundaries. This probe never changes the official sampler.
#include "frugal_log_trace.hpp"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

void require(bool condition, char const* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    using namespace fgbs;
    using namespace fgbs::cpu;
    const std::vector<double> thresholds = {-1.0, 0.0, std::nextafter(0.0, 1.0),
        .25, .5, std::nextafter(1.0, 0.0), 1.0, std::nextafter(1.0, 2.0), 2.0,
        std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()};
    std::size_t forced_checks = 0, upstream_checks = 0;
    for (double threshold : thresholds) {
        UniformRandom<ProbSimd> expected_rng(90121), official_rng(90121);
        const auto uniforms = expected_rng();
        SampleResult<8, ProbSimd> sample{};
        ProbSimd previous(1.0), parity[8]{};
        update_result<8, ProbSimd>(official_rng, sample, previous, parity, 0, ProbSimd(threshold));
        for (std::size_t lane = 0; lane < ProbSimd::size; ++lane) {
            require(get_bit<8>(sample[lane], 0) == (uniforms.get(lane) > threshold), "upstream threshold decision mismatch");
            ++upstream_checks;
        }
        const double p0 = std::isnan(threshold) ? 1.0 : std::max(0.0, std::min(1.0, threshold));
        for (bool one : {false, true}) {
            TraceLogPathResult trace;
            const bool reachable = record_forced_threshold(trace, threshold, one);
            const long double expected = one ? 1.0L - p0 : p0;
            require(reachable == (expected > 0), "reachable branch mismatch");
            require(trace.probability == expected, "clipped branch mass mismatch");
            require(trace.nonfinite_threshold_path_steps == std::uint64_t(!std::isfinite(threshold)), "nonfinite threshold hidden");
            require(trace.saturated_threshold_path_steps == std::uint64_t(threshold < 0 || threshold > 1), "saturation counter mismatch");
            if (expected == 0) require(std::isinf(trace.log_probability) && trace.log_probability < 0, "zero branch lacks minus infinity");
            else require(std::fabs(std::exp(trace.log_probability) - expected) < 2e-18L, "logged branch mass mismatch");
            ++forced_checks;
        }
    }
    TraceLogPathResult tiny;
    for (int i = 0; i < 6000; ++i)
        require(record_forced_threshold(tiny, .01, false), "product underflow terminated positive log-density path");
    require(tiny.probability == 0 && std::isfinite(tiny.log_probability), "underflow test did not retain finite log density");
    require(std::fabs(tiny.log_probability - 6000 * std::log(.01L)) < 1e-10L, "long log accumulation mismatch");
    std::cout << "{\"passed\":true,\"threshold_cases\":" << thresholds.size()
        << ",\"forced_branch_checks\":" << forced_checks << ",\"upstream_lane_decision_checks\":" << upstream_checks
        << ",\"positive_branch_product_underflow_preserved\":true}\n";
}
