// Evidence adapter to pinned official FrugalGBS. The timed sample generator
// stays unchanged. A separate forced-branch trace scores each retained output
// after generation; scoring is neither part of the official API nor evidence
// that its approximate law equals the target Gaussian boson distribution.
#include "fgbs/data.hpp"
#include "fgbs/layout.hpp"
#include "fgbs/sampler.hpp"
#include "frugal_log_trace.hpp"
#include "frugal_modes.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

namespace {
using Clock = std::chrono::steady_clock;
constexpr auto commit = "fa8b796f2afebd4845a2e335e457735952617ef8";
constexpr auto sampler_body_sha256 = "9c14a870f2875e808025d9e3535a71ba86144e5639ab14f0b7be5b594dabb3fb";
char const* current_phase = "options";

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void json_string(std::ostream& out, std::string const& value) {
    out << '"';
    for (const unsigned char c : value) {
        if (c == '"' || c == '\\') out << '\\' << c;
        else if (c == '\n') out << "\\n";
        else if (c == '\r') out << "\\r";
        else if (c == '\t') out << "\\t";
        else if (c < 0x20) out << "?";
        else out << c;
    }
    out << '"';
}

void phase_start(char const* name) {
    current_phase = name;
    std::cerr << "{\"phase\":\"" << name << "\",\"status\":\"starting\"}\n" << std::flush;
}

void phase_done(char const* name, double seconds) {
    std::cerr << std::setprecision(17) << "{\"phase\":\"" << name
        << "\",\"status\":\"completed\",\"seconds\":" << seconds << "}\n" << std::flush;
}

struct Options {
    std::string input;
    std::size_t modes = 0, cutoff = 5, samples = 1024;
    unsigned int seed = 0;
    int threads = 1;
    bool enumerate = false, raw = false, log_probabilities = false;
};

std::uint64_t unsigned_value(std::string const& value) {
    if (value.empty() || value.front() == '-') throw std::runtime_error("Expected a nonnegative integer");
    std::size_t consumed;
    const auto result = std::stoull(value, &consumed);
    if (consumed != value.size()) throw std::runtime_error("Invalid integer: " + value);
    return result;
}

Options options(int argc, char** argv) {
    Options result;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--enumerate") { result.enumerate = true; continue; }
        if (flag == "--raw-samples") { result.raw = true; continue; }
        if (flag == "--record-log-probabilities") { result.log_probabilities = true; continue; }
        if (flag == "--help") {
            std::cout << "frugal_evidence_adapter --input covariance.csv --modes N --cutoff K "
                "[--samples M --seed S --threads T --raw-samples --record-log-probabilities --enumerate]\n";
            std::exit(0);
        }
        if (++i >= argc) throw std::runtime_error("Missing argument for " + flag);
        const std::string value = argv[i];
        if (flag == "--input") result.input = value;
        else if (flag == "--modes") result.modes = unsigned_value(value);
        else if (flag == "--cutoff") result.cutoff = unsigned_value(value);
        else if (flag == "--samples") result.samples = unsigned_value(value);
        else if (flag == "--seed") {
            const auto seed = unsigned_value(value);
            if (seed > std::numeric_limits<unsigned int>::max()) throw std::runtime_error("Seed must fit uint32");
            result.seed = static_cast<unsigned int>(seed);
        } else if (flag == "--threads") {
            const auto threads = unsigned_value(value);
            if (threads == 0 || threads > 1024) throw std::runtime_error("Threads must lie in [1,1024]");
            result.threads = static_cast<int>(threads);
        } else throw std::runtime_error("Unknown option: " + flag);
    }
    if (result.input.empty()) throw std::runtime_error("--input is required");
    if (result.modes < 2 || result.modes > 144) throw std::runtime_error("Mode count outside [2,144]");
    if (result.cutoff < 1 || result.cutoff > std::min(std::size_t{5}, result.modes))
        throw std::runtime_error("Cutoff outside [1,min(5,modes)]");
    if (result.enumerate && result.modes > 14) throw std::runtime_error("Full-law enumeration is limited to 14 modes");
    return result;
}

fgbs::Matrix<fgbs::f64> read_covariance(std::string const& path, std::size_t modes) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open covariance: " + path);
    const auto dimension = 2 * modes;
    fgbs::Matrix<fgbs::f64> matrix(dimension, dimension);
    std::string line;
    std::size_t row = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (row == dimension) throw std::runtime_error("Too many covariance rows");
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream values(line);
        for (std::size_t col = 0; col < dimension; ++col) {
            double value;
            if (!(values >> value) || !std::isfinite(value)) throw std::runtime_error("Finite square covariance required");
            matrix(row, col) = value;
        }
        std::string extra;
        if (values >> extra) throw std::runtime_error("Too many covariance columns");
        ++row;
    }
    if (row != dimension) throw std::runtime_error("Wrong covariance row count");
    return matrix;
}

template<typename T> T checked(fgbs::Result<T>&& result) {
    if (!result.ok()) throw std::runtime_error(result.error());
    return std::move(result.value());
}

template<typename T> void json_vector(std::vector<T> const& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << values[i];
    }
    std::cout << ']';
}

template<fgbs::usize N> struct TraceWorkspace {
    using T = fgbs::cpu::ProbSimd;
    std::vector<T> partial, elided, partial_elided, double_elided;
    TraceWorkspace() :
        partial(fgbs::cpu::partial_marginal_arr_size<N>()),
        elided(fgbs::cpu::elided_marginal_arr_size<N>()),
        partial_elided(fgbs::cpu::partial_elided_marginal_arr_size<N>()),
        double_elided(fgbs::cpu::double_elided_marginal_arr_size<N>()) {}
    fgbs::cpu::TraceLogPathResult score(fgbs::BitstringLayout<N> const& bitstrings,
            fgbs::SamplingTarget<N> const& target, fgbs::Bitstring<N> const& sample) {
        return fgbs::cpu::trace_sample_log_probability<N>(bitstrings, target, sample,
            partial.data(), elided.data(), partial_elided.data(), double_elided.data());
    }
    static constexpr std::size_t bytes = sizeof(T) * (
        fgbs::cpu::partial_marginal_arr_size<N>() + fgbs::cpu::elided_marginal_arr_size<N>()
        + fgbs::cpu::partial_elided_marginal_arr_size<N>() + fgbs::cpu::double_elided_marginal_arr_size<N>());
};

template<fgbs::usize N> int run(Options const& option) {
    phase_start("input"); auto started = Clock::now();
    const auto covariance = read_covariance(option.input, N);
    const auto input_seconds = elapsed(started); phase_done("input", input_seconds);
    phase_start("bitstring_layout"); started = Clock::now();
    const auto bitstrings = checked(fgbs::generate_bitstring_layout<N>(option.cutoff));
    const auto bitstring_seconds = elapsed(started); phase_done("bitstring_layout", bitstring_seconds);
    phase_start("cumulant_layout"); started = Clock::now();
    const auto cumulants = checked(fgbs::generate_cumulant_layout<N>(option.cutoff));
    const auto cumulant_seconds = elapsed(started); phase_done("cumulant_layout", cumulant_seconds);
    phase_start("target"); started = Clock::now();
    const auto target = checked(fgbs::generate_sampling_target<N>(bitstrings, cumulants, covariance));
    const auto target_seconds = elapsed(started); phase_done("target", target_seconds);
    phase_start("sampling"); started = Clock::now();
    // The canonical public upstream API is called exactly once, unchanged.
    const auto samples = checked(fgbs::generate_samples<N>(bitstrings, target, option.samples, option.seed));
    const auto sampling_seconds = elapsed(started); phase_done("sampling", sampling_seconds);
    if (samples.size() != option.samples) throw std::runtime_error("Official API returned an incomplete batch");

    std::unique_ptr<TraceWorkspace<N>> workspace;
    std::vector<double> log_probabilities;
    std::uint64_t density_saturated = 0, density_nonfinite = 0;
    double density_seconds = 0;
    std::string evidence_error, error_phase;
    if (option.log_probabilities) {
        phase_start("density"); started = Clock::now();
        try {
            if (!samples.empty()) workspace = std::make_unique<TraceWorkspace<N>>();
            log_probabilities.reserve(samples.size());
            for (auto const& sample : samples) {
                const auto trace = workspace->score(bitstrings, target, sample);
                density_saturated += trace.saturated_threshold_path_steps;
                density_nonfinite += trace.nonfinite_threshold_path_steps;
                const double logp = static_cast<double>(trace.log_probability);
                if (trace.nonfinite_threshold_path_steps || !std::isfinite(logp) || logp > 0.0)
                    throw std::runtime_error("Nonfinite threshold or impossible sampled branch in density trace");
                log_probabilities.push_back(logp);
            }
        } catch (std::exception const& error) {
            evidence_error = error.what(); error_phase = "density";
        }
        density_seconds = elapsed(started);
        if (evidence_error.empty()) phase_done("density", density_seconds);
    }

    std::vector<double> distribution;
    long double probability_sum = 0;
    std::uint64_t saturated = 0, nonfinite = 0;
    double enumeration_seconds = 0;
    if (option.enumerate && evidence_error.empty()) {
        if constexpr (N <= 14) {
            phase_start("enumeration"); started = Clock::now();
            try {
                if (!workspace) workspace = std::make_unique<TraceWorkspace<N>>();
                distribution.resize(std::size_t{1} << N);
                for (std::size_t mask = 0; mask < distribution.size(); ++mask) {
                    fgbs::Bitstring<N> forced{};
                    for (std::size_t i = 0; i < N; ++i) if ((mask >> i) & 1) fgbs::set_bit<N>(forced, i);
                    const auto trace = workspace->score(bitstrings, target, forced);
                    distribution[mask] = static_cast<double>(trace.probability);
                    probability_sum += trace.probability;
                    saturated += trace.saturated_threshold_path_steps;
                    nonfinite += trace.nonfinite_threshold_path_steps;
                    if (!std::isfinite(distribution[mask]) || distribution[mask] < 0)
                        throw std::runtime_error("Invalid probability from full-law trace");
                }
                if (nonfinite || std::fabs(probability_sum - 1.0L) > 1e-11L)
                    throw std::runtime_error("Traced law failed finite/normalization check");
            } catch (std::exception const& error) {
                evidence_error = error.what(); error_phase = "enumeration";
            }
            enumeration_seconds = elapsed(started);
            if (evidence_error.empty()) phase_done("enumeration", enumeration_seconds);
        }
    }
    phase_start("histogram"); started = Clock::now();
    std::vector<std::uint64_t> histogram;
    if constexpr (N <= 14) histogram.assign(std::size_t{1} << N, 0);
    std::vector<std::uint64_t> per_mode(N, 0), click_counts(N + 1, 0);
    for (auto const& sample : samples) {
        std::size_t clicks = 0, mask = 0;
        for (std::size_t i = 0; i < N; ++i) {
            const bool bit = fgbs::get_bit<N>(sample, i);
            per_mode[i] += bit; clicks += bit;
            if constexpr (N <= 14) mask |= std::size_t(bit) << i;
        }
        ++click_counts[clicks];
        if constexpr (N <= 14) ++histogram[mask];
    }
    const auto histogram_seconds = elapsed(started); phase_done("histogram", histogram_seconds);
    struct rusage usage{}; getrusage(RUSAGE_SELF, &usage);

    std::cout << std::setprecision(17) << "{\"schema_version\":1,\"status\":\""
        << (evidence_error.empty() ? "ok" : "error") << "\",\"upstream_commit\":\"" << commit
        << "\",\"sampler_body_sha256\":\"" << sampler_body_sha256
        << "\",\"implementation\":\"FrugalGBS official CPU with separate evidence trace\""
        << ",\"sampling_api\":\"unmodified official generate_samples\""
        << ",\"covariance_convention\":\"xxpp; hbar=2; vacuum=I; zero displacement\""
        << ",\"mask_convention\":\"bit i is physical mode i; one is a click\""
        << ",\"modes\":" << N << ",\"cutoff\":" << option.cutoff
        << ",\"sample_count\":" << samples.size() << ",\"requested_sample_count\":" << option.samples
        << ",\"seed\":" << option.seed << ",\"omp_threads\":" << omp_get_max_threads()
        << ",\"simd_lanes\":" << fgbs::cpu::ProbSimd::size << ",\"real_precision\":2"
        << ",\"precision\":\"float64\",\"log_probability_enabled\":" << (option.log_probabilities ? "true" : "false")
        << ",\"sample_log_probability_semantics\":\"unconditional chosen-branch path mass of the computed upstream clipped-threshold law under independent continuous uniforms; not target GBS or finite-PRNG mass\""
        << ",\"sample_log_probability_accumulation\":\"long-double log/log1p of upstream double thresholds; float64 output\""
        << ",\"sample_log_probability_storage_bytes\":" << log_probabilities.size() * sizeof(double)
        << ",\"timings_seconds\":{\"input\":" << input_seconds
        << ",\"bitstring_layout\":" << bitstring_seconds << ",\"cumulant_layout\":" << cumulant_seconds
        << ",\"layout\":" << bitstring_seconds + cumulant_seconds << ",\"target\":" << target_seconds
        << ",\"sampling\":" << sampling_seconds << ",\"density\":" << density_seconds
        << ",\"histogram\":" << histogram_seconds << ",\"enumeration\":" << enumeration_seconds << "}"
        << ",\"histogram_counts\":";
    json_vector(histogram);
    std::cout << ",\"per_mode_click_counts\":"; json_vector(per_mode);
    std::cout << ",\"click_count_histogram\":"; json_vector(click_counts);
    std::cout << ",\"peak_rss_bytes_linux\":" << std::uint64_t(usage.ru_maxrss) * 1024;
    std::cout << ",\"layout_entries\":" << bitstrings.len
        << ",\"trace_workspace_bytes\":" << TraceWorkspace<N>::bytes;
    std::cout << ",\"raw_samples\":";
    if (option.raw) {
        std::cout << '[';
        for (std::size_t j = 0; j < samples.size(); ++j) {
            if (j) std::cout << ',';
            std::cout << '[';
            for (std::size_t i = 0; i < N; ++i) {
                if (i) std::cout << ',';
                std::cout << int(fgbs::get_bit<N>(samples[j], i));
            }
            std::cout << ']';
        }
        std::cout << ']';
    } else std::cout << "null";
    std::cout << ",\"sample_log_probabilities\":";
    if (option.log_probabilities && error_phase != "density") json_vector(log_probabilities);
    else std::cout << "null";
    std::cout << ",\"density_trace\":";
    if (option.log_probabilities) {
        std::cout << "{\"method\":\"generated forced-branch upstream SIMD body\",\"official_api\":false"
            << ",\"after_timed_sampling\":true,\"threads\":1,\"scored_samples\":" << log_probabilities.size()
            << ",\"saturated_threshold_path_steps\":" << density_saturated
            << ",\"nonfinite_threshold_path_steps\":" << density_nonfinite << "}";
    } else std::cout << "null";
    std::cout << ",\"distribution\":";
    if (option.enumerate && evidence_error.empty()) json_vector(distribution); else std::cout << "null";
    std::cout << ",\"trace\":";
    if (option.enumerate && evidence_error.empty()) {
        std::cout << "{\"method\":\"generated forced-branch upstream SIMD body\",\"official_api\":false"
            << ",\"probability_model\":\"continuous uniform thresholds; floating-point SIMD arithmetic\""
            << ",\"probability_sum\":" << static_cast<double>(probability_sum)
            << ",\"saturated_threshold_path_steps\":" << saturated
            << ",\"nonfinite_threshold_path_steps\":" << nonfinite << "}";
    } else std::cout << "null";
    if (!evidence_error.empty()) {
        std::cout << ",\"error_phase\":"; json_string(std::cout, error_phase);
        std::cout << ",\"error\":"; json_string(std::cout, evidence_error);
        std::cout << ",\"no_retry\":true,\"samples_postselected\":false";
    }
    std::cout << "}\n";
    return evidence_error.empty() ? 0 : 1;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const auto option = options(argc, argv);
        omp_set_dynamic(0); omp_set_num_threads(option.threads);
        switch (option.modes) {
#define GBSLAB_MODE_CASE(N) case N: return run<N>(option);
            GBSLAB_FRUGAL_MODES(GBSLAB_MODE_CASE)
#undef GBSLAB_MODE_CASE
            default: throw std::runtime_error("Mode count is not compiled; rebuild isolated evidence adapter");
        }
    } catch (std::exception const& error) {
        std::cout << "{\"schema_version\":1,\"status\":\"error\",\"error_phase\":";
        json_string(std::cout, current_phase);
        std::cout << ",\"error\":"; json_string(std::cout, error.what());
        std::cout << ",\"no_retry\":true,\"samples_postselected\":false}\n";
        return 1;
    }
}
