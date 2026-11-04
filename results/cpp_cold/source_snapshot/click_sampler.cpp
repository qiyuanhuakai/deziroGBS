// Native implementation of gbslab's retained-click sampling model.
// State is Gaussian conditional on ALL observed zeros only. All past clicks
// remain in the state, even when not selected in the current event ratio.
// No full target distribution is accepted or consulted by this executable.

#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;
using Matrix = Eigen::MatrixXd;
constexpr int max_modes = 256;
constexpr int max_clicks = 8;
constexpr int max_kernel_modes = max_clicks + 1;
constexpr int enumeration_cap = 14;
constexpr double probability_tolerance = 2e-10;
constexpr double max_relative_roundoff = 1e-6;
constexpr std::size_t raw_byte_cap = 256ULL * 1024 * 1024;
constexpr std::size_t uniform_byte_cap = 512ULL * 1024 * 1024;
constexpr std::size_t sample_byte_cap = 512ULL * 1024 * 1024;
constexpr std::size_t trace_step_cap = 1000000;
constexpr std::size_t step_cache_entry_cap = 65536;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string quote(std::string const& text) {
    std::ostringstream result;
    result << '"';
    for (const unsigned char c : text) {
        switch (c) {
            case '"': result << "\\\""; break;
            case '\\': result << "\\\\"; break;
            case '\n': result << "\\n"; break;
            case '\r': result << "\\r"; break;
            case '\t': result << "\\t"; break;
            default:
                if (c < 32) result << "\\u" << std::hex << std::setw(4)
                                   << std::setfill('0') << int(c) << std::dec;
                else result << c;
        }
    }
    result << '"';
    return result.str();
}

template<typename T>
void print_vector(std::vector<T> const& values) {
    std::cout << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i) std::cout << ',';
        std::cout << +values[i];
    }
    std::cout << ']';
}

void print_finite_or_null(double value) {
    if (std::isfinite(value)) std::cout << value;
    else std::cout << "null";
}

struct NumericalProbabilityError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

struct Options {
    std::string input, uniforms;
    std::string selector = "pairwise";
    int modes = 0, k = 2;
    std::size_t samples = 0;
    std::uint64_t seed = 0;
    std::vector<int> order;
    bool enumerate = false, raw_samples = false, profile = false, trace = false;
    bool step_cache = true;
};

std::uint64_t nonnegative_integer(std::string const& value) {
    if (value.empty() || value.front() == '-')
        throw std::invalid_argument("Expected a nonnegative integer: " + value);
    std::size_t consumed;
    const auto number = std::stoull(value, &consumed);
    if (consumed != value.size()) throw std::invalid_argument("Invalid integer: " + value);
    return number;
}

Options parse_options(int argc, char** argv) {
    Options result;
    std::string order_text;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        if (flag == "--enumerate") { result.enumerate = true; continue; }
        if (flag == "--raw-samples") { result.raw_samples = true; continue; }
        if (flag == "--profile") { result.profile = true; continue; }
        if (flag == "--trace-conditionals") { result.trace = true; continue; }
        if (flag == "--no-step-cache") { result.step_cache = false; continue; }
        if (flag == "--help") {
            std::cout << "click_sampler --input covariance.csv --modes M --selector "
                         "pairwise|click_influence --k K [--samples N --seed S] "
                         "[--order 0,1,... --uniforms matrix.csv --enumerate "
                         "--raw-samples --trace-conditionals --profile --no-step-cache]\n";
            std::exit(0);
        }
        if (++i >= argc) throw std::invalid_argument("Missing value for " + flag);
        const std::string value = argv[i];
        if (flag == "--input") result.input = value;
        else if (flag == "--uniforms") result.uniforms = value;
        else if (flag == "--selector") result.selector = value;
        else if (flag == "--order") order_text = value;
        else if (flag == "--samples") result.samples = nonnegative_integer(value);
        else if (flag == "--seed") result.seed = nonnegative_integer(value);
        else if (flag == "--modes" || flag == "--k") {
            const auto number = nonnegative_integer(value);
            if (number > max_modes) throw std::invalid_argument("Mode/budget value is too large");
            if (flag == "--modes") result.modes = static_cast<int>(number);
            else result.k = static_cast<int>(number);
        }
        else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (result.input.empty()) throw std::invalid_argument("--input is required");
    if (result.modes < 1 || result.modes > max_modes)
        throw std::invalid_argument("modes must lie in [1,256]");
    if (result.k < 0 || result.k > max_clicks)
        throw std::invalid_argument("retained-click budget must lie in [0,8]");
    if (result.selector != "pairwise" && result.selector != "click_influence")
        throw std::invalid_argument("selector must be pairwise or click_influence");
    if (result.enumerate && result.modes > enumeration_cap)
        throw std::invalid_argument("full-law enumeration is limited to 14 modes");
    const std::size_t words = (result.modes+63)/64;
    if (result.samples > sample_byte_cap / sizeof(std::uint64_t) / words)
        throw std::invalid_argument("materialized packed samples exceed 512 MiB");
    if (result.raw_samples && result.samples > raw_byte_cap / result.modes)
        throw std::invalid_argument("requested raw sample buffer exceeds 256 MiB");
    if (!result.uniforms.empty()
            && result.samples > uniform_byte_cap / sizeof(double) / result.modes)
        throw std::invalid_argument("uniform injection buffer exceeds 512 MiB");
    if (result.trace && result.samples > trace_step_cap / result.modes)
        throw std::invalid_argument("conditional trace exceeds one million steps");
    if (order_text.empty()) {
        result.order.resize(result.modes);
        std::iota(result.order.begin(), result.order.end(), 0);
    } else {
        std::istringstream input(order_text);
        std::string piece;
        while (std::getline(input, piece, ',')) {
            const auto mode = nonnegative_integer(piece);
            if (mode >= static_cast<std::uint64_t>(result.modes))
                throw std::invalid_argument("order contains an out-of-range mode");
            result.order.push_back(static_cast<int>(mode));
        }
        auto sorted = result.order;
        std::sort(sorted.begin(), sorted.end());
        if (sorted.size() != static_cast<std::size_t>(result.modes))
            throw std::invalid_argument("order must contain every mode once");
        for (int i = 0; i < result.modes; ++i)
            if (sorted[i] != i) throw std::invalid_argument("order is not a permutation");
    }
    return result;
}

std::vector<double> read_csv(std::string const& path, std::size_t rows, std::size_t columns) {
    std::ifstream input(path);
    if (!input) throw std::invalid_argument("Cannot open input: " + path);
    std::vector<double> values;
    values.reserve(rows * columns);  // dimensions/caps are checked before this call
    std::string line;
    std::size_t row = 0;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        if (row == rows) throw std::invalid_argument("Too many rows in " + path);
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream items(line);
        for (std::size_t col = 0; col < columns; ++col) {
            double value;
            if (!(items >> value) || !std::isfinite(value))
                throw std::invalid_argument("Missing or nonfinite CSV value in " + path);
            values.push_back(value);
        }
        std::string extra;
        if (items >> extra) throw std::invalid_argument("Too many columns in " + path);
        ++row;
    }
    if (row != rows) throw std::invalid_argument("Wrong row count in " + path);
    return values;
}

struct Context {
    std::string phase = "arguments";
    std::size_t sample_index = 0;
    int position = -1, mode = -1, zero_count = 0, click_count = 0;
    std::vector<int> kernel_modes;
    double last_indicator = 0, last_denominator = 1;
};

struct Counters {
    std::uint64_t conditional_evaluations = 0, selection_requests = 0;
    std::uint64_t selection_computations = 0, selection_score_evaluations = 0;
    std::uint64_t selection_conditional_kernel_calls = 0;
    std::uint64_t small_determinant_evaluations = 0;
    std::uint64_t selection_small_determinant_evaluations = 0;
    std::uint64_t pairwise_marginals_requested = 0, pairwise_determinant_evaluations = 0;
    std::uint64_t gaussian_rank_two_updates = 0, event_corrections = 0;
    std::uint64_t probability_corrections = 0, pairwise_probability_corrections = 0;
    std::uint64_t vacuum_probability_corrections = 0, guard_failures = 0;
    std::uint64_t step_vacuum_requests = 0, step_cache_hits = 0, step_cache_misses = 0;
    std::uint64_t step_cache_capacity_skips = 0;
    std::size_t step_cache_peak_entries = 0, step_cache_peak_table_bytes = 0;
    int maximum_determinant_dimension = 0, maximum_selection_candidates = 0;
    double maximum_cancellation_indicator = 0, maximum_pairwise_l1_correction = 0;
    double minimum_conditioning_probability = std::numeric_limits<double>::infinity();
    double minimum_click_probability = std::numeric_limits<double>::infinity();
    double maximum_click_probability = -std::numeric_limits<double>::infinity();

    [[noreturn]] void fail(std::string const& message) {
        ++guard_failures;
        throw NumericalProbabilityError(message);
    }

    void print() const {
        std::cout << "{\"conditional_evaluations\":" << conditional_evaluations
            << ",\"selection_requests\":" << selection_requests
            << ",\"selection_computations\":" << selection_computations
            << ",\"selection_score_evaluations\":" << selection_score_evaluations
            << ",\"selection_conditional_kernel_calls\":" << selection_conditional_kernel_calls
            << ",\"small_determinant_evaluations\":" << small_determinant_evaluations
            << ",\"selection_small_determinant_evaluations\":" << selection_small_determinant_evaluations
            << ",\"pairwise_marginals_requested\":" << pairwise_marginals_requested
            << ",\"pairwise_determinant_evaluations\":" << pairwise_determinant_evaluations
            << ",\"gaussian_rank_two_updates\":" << gaussian_rank_two_updates
            << ",\"event_corrections\":" << event_corrections
            << ",\"probability_corrections\":" << probability_corrections
            << ",\"pairwise_probability_corrections\":" << pairwise_probability_corrections
            << ",\"vacuum_probability_corrections\":" << vacuum_probability_corrections
            << ",\"guard_failures\":" << guard_failures
            << ",\"step_vacuum_requests\":" << step_vacuum_requests
            << ",\"step_cache_hits\":" << step_cache_hits
            << ",\"step_cache_misses\":" << step_cache_misses
            << ",\"step_cache_capacity_skips\":" << step_cache_capacity_skips
            << ",\"step_cache_peak_entries\":" << step_cache_peak_entries
            << ",\"step_cache_peak_table_bytes\":" << step_cache_peak_table_bytes
            << ",\"maximum_determinant_dimension\":" << maximum_determinant_dimension
            << ",\"maximum_selection_candidates\":" << maximum_selection_candidates
            << ",\"maximum_cancellation_indicator\":";
        print_finite_or_null(maximum_cancellation_indicator);
        std::cout << ",\"maximum_pairwise_l1_correction\":" << maximum_pairwise_l1_correction
            << ",\"minimum_conditioning_probability\":";
        print_finite_or_null(minimum_conditioning_probability);
        std::cout << ",\"minimum_click_probability\":";
        print_finite_or_null(minimum_click_probability);
        std::cout << ",\"maximum_click_probability\":";
        print_finite_or_null(maximum_click_probability);
        std::cout << '}';
    }
};

struct Profile {
    bool enabled;
    double state_update = 0, selection_inclusive = 0;
    double selection_event = 0, sampling_event = 0, scratch_cache = 0;
    explicit Profile(bool enabled_) : enabled(enabled_) {}
    void print() const {
        if (!enabled) { std::cout << "null"; return; }
        std::cout << "{\"state_update_seconds\":" << state_update
            << ",\"selection_seconds_inclusive\":" << selection_inclusive
            << ",\"selection_event_seconds\":" << selection_event
            << ",\"selection_overhead_seconds\":" << selection_inclusive-selection_event
            << ",\"sampling_event_seconds\":" << sampling_event
            << ",\"scratch_cache_seconds\":" << scratch_cache << '}';
    }
};

struct Timer {
    bool enabled;
    double& destination;
    Clock::time_point started;
    Timer(bool enabled_, double& destination_) : enabled(enabled_), destination(destination_) {
        if (enabled) started = Clock::now();
    }
    ~Timer() { if (enabled) destination += elapsed(started); }
};

struct State {
    Matrix covariance;
    std::vector<int> active;
    std::array<int, max_modes> positions{};
    int zero_count = 0;

    explicit State(Matrix const& base) : covariance(base), active(base.rows()/2) {
        positions.fill(-1);
        std::iota(active.begin(), active.end(), 0);
        for (std::size_t j = 0; j < active.size(); ++j) positions[j] = static_cast<int>(j);
    }

    void project_zero(int physical_mode, Counters& counts, Profile& profile) {
        Timer timer(profile.enabled, profile.state_update);
        const int n = static_cast<int>(active.size());
        const int measured = positions[physical_mode];
        if (measured < 0) throw std::logic_error("Cannot project a previously removed mode");
        Eigen::Matrix2d block;
        block << covariance(measured, measured)+1.0, covariance(measured, measured+n),
                 covariance(measured+n, measured), covariance(measured+n, measured+n)+1.0;
        Eigen::LLT<Eigen::Matrix2d> factor(block);
        if (factor.info() != Eigen::Success)
            counts.fail("Gaussian vacuum update is not positive definite");
        const double l00 = factor.matrixL()(0,0);
        const double l10 = factor.matrixL()(1,0);
        const double l11 = factor.matrixL()(1,1);
        if (!(l00 > 0) || !(l11 > 0) || !std::isfinite(l00) || !std::isfinite(l11))
            counts.fail("Gaussian vacuum update has an invalid Cholesky pivot");
        const int remaining = n-1, dimension = 2*remaining;
        std::array<int, 2*max_modes> keep{};
        for (int j = 0, next = 0; j < n; ++j) if (j != measured) {
            keep[next] = j;
            keep[next+remaining] = j+n;
            ++next;
        }
        std::array<double, 2*max_modes> first{}, second{};
        for (int r = 0; r < dimension; ++r) {
            first[r] = covariance(keep[r], measured) / l00;
            second[r] = (covariance(keep[r], measured+n)-l10*first[r]) / l11;
        }
        Matrix next(dimension, dimension);
        // Compute one symmetric triangle: an explicit rank-two update.
        for (int col = 0; col < dimension; ++col) {
            for (int row = 0; row <= col; ++row) {
                const double value = covariance(keep[row], keep[col])
                    - (first[row]*first[col] + second[row]*second[col]);
                if (!std::isfinite(value)) counts.fail("Nonfinite Gaussian conditional covariance");
                next(row,col) = value;
                next(col,row) = value;
            }
        }
        covariance.swap(next);
        active.erase(active.begin()+measured);
        positions.fill(-1);
        for (int j = 0; j < remaining; ++j) positions[active[j]] = j;
        ++zero_count;
        ++counts.gaussian_rank_two_updates;
    }
};

enum class KernelRole { sampling, selection, setup };

// This cache lives for ONE conditional evaluation. Its ordered physical-mode
// key preserves each determinant's row/column order, so reuse does not even
// introduce a permutation-related floating-point change. A generation reset
// discards all old values before any different zero-conditioned state is used.
// The open-addressed table is bounded and reused as workspace across steps.
class StepVacuumCache {
public:
    struct Key { std::uint64_t low = 0, high = 0; };
    struct Entry {
        Key key;
        double probability = 0;
        std::uint32_t generation = 0;
    };
    std::vector<Entry> table;
    std::uint32_t generation = 0;
    std::size_t entries = 0;

    void reset(std::size_t expected_entries, Counters& counts) {
        const std::size_t desired = std::min(expected_entries, step_cache_entry_cap);
        std::size_t capacity = 16;
        while (capacity < 2*desired) capacity <<= 1;
        if (table.size() < capacity) table.resize(capacity);
        if (++generation == 0) {
            for (auto& item : table) item.generation = 0;
            generation = 1;
        }
        entries = 0;
        counts.step_cache_peak_table_bytes = std::max(counts.step_cache_peak_table_bytes,
                                                       table.size()*sizeof(Entry));
    }

    static std::uint64_t hash(Key const& key) {
        std::uint64_t value = key.low ^ (key.high*0x9e3779b97f4a7c15ULL);
        value ^= value >> 30; value *= 0xbf58476d1ce4e5b9ULL;
        value ^= value >> 27; value *= 0x94d049bb133111ebULL;
        return value ^ (value >> 31);
    }

    Entry* lookup(Key const& key, bool& hit) {
        const auto mask = table.size()-1;
        std::size_t position = hash(key) & mask;
        for (;;) {
            auto& entry = table[position];
            if (entry.generation != generation) { hit = false; return &entry; }
            if (entry.key.low == key.low && entry.key.high == key.high) {
                hit = true;
                return &entry;
            }
            position = (position+1) & mask;
        }
    }

    void insert(Entry* location, Key const& key, double value, Counters& counts) {
        if (entries >= std::min(step_cache_entry_cap, table.size()/2)) {
            ++counts.step_cache_capacity_skips;
            return;
        }
        location->key = key;
        location->probability = value;
        location->generation = generation;
        ++entries;
        counts.step_cache_peak_entries = std::max(counts.step_cache_peak_entries, entries);
    }
};

template<int Dimension>
double fixed_vacuum(State const& state, std::array<int, 2*max_kernel_modes> const& indices,
                    Counters& counts) {
    using Fixed = Eigen::Matrix<double, Dimension, Dimension>;
    Fixed shifted;
    for (int col = 0; col < Dimension; ++col)
        for (int row = 0; row < Dimension; ++row)
            shifted(row,col) = 0.5 * (state.covariance(indices[row],indices[col])
                                     + (row == col ? 1.0 : 0.0));
    Eigen::LLT<Fixed> factor(shifted);
    if (factor.info() != Eigen::Success)
        counts.fail("Conditioned vacuum determinant is not positive definite");
    double log_sqrt_det = 0;
    for (int i = 0; i < Dimension; ++i) {
        const double diagonal = factor.matrixL()(i,i);
        if (!(diagonal > 0) || !std::isfinite(diagonal))
            counts.fail("Conditioned vacuum factor is not positive finite");
        log_sqrt_det += std::log(diagonal);
    }
    const double probability = std::exp(-log_sqrt_det);
    if (!std::isfinite(probability) || probability > 1.0 + probability_tolerance)
        counts.fail("Conditioned vacuum probability is nonfinite or exceeds one");
    if (probability > 1.0) {
        ++counts.vacuum_probability_corrections;
        return 1.0;
    }
    return probability;
}

double vacuum(State const& state, std::vector<int> const& modes,
              unsigned int mask, Counters& counts, KernelRole role,
              StepVacuumCache* cache = nullptr) {
    if (!mask) return 1.0;
    const int size = std::popcount(mask);
    std::array<int, 2*max_kernel_modes> indices{};
    StepVacuumCache::Key key;
    const int n = static_cast<int>(state.active.size());
    for (std::size_t j = 0, next = 0; j < modes.size(); ++j) if (mask & (1U << j)) {
        const int position = state.positions[modes[j]];
        if (position < 0) throw std::logic_error("Event references a zero-observed mode");
        indices[next] = position;
        indices[next+size] = position+n;
        if (cache) {
            const auto physical = static_cast<std::uint64_t>(modes[j]);
            if (next < 8) key.low |= physical << (8*next);
            else key.high = physical;
        }
        ++next;
    }
    StepVacuumCache::Entry* location = nullptr;
    if (cache) {
        key.high |= static_cast<std::uint64_t>(size) << 8;
        ++counts.step_vacuum_requests;
        bool hit;
        location = cache->lookup(key, hit);
        if (hit) {
            ++counts.step_cache_hits;
            return location->probability;
        }
        ++counts.step_cache_misses;
    }
    if (role == KernelRole::sampling) ++counts.small_determinant_evaluations;
    else if (role == KernelRole::selection) ++counts.selection_small_determinant_evaluations;
    else ++counts.pairwise_determinant_evaluations;
    counts.maximum_determinant_dimension = std::max(counts.maximum_determinant_dimension, 2*size);
    double value;
    switch (size) {
        case 1: value = fixed_vacuum<2>(state, indices, counts); break;
        case 2: value = fixed_vacuum<4>(state, indices, counts); break;
        case 3: value = fixed_vacuum<6>(state, indices, counts); break;
        case 4: value = fixed_vacuum<8>(state, indices, counts); break;
        case 5: value = fixed_vacuum<10>(state, indices, counts); break;
        case 6: value = fixed_vacuum<12>(state, indices, counts); break;
        case 7: value = fixed_vacuum<14>(state, indices, counts); break;
        case 8: value = fixed_vacuum<16>(state, indices, counts); break;
        case 9: value = fixed_vacuum<18>(state, indices, counts); break;
        default: throw std::logic_error("Retained-event kernel dimension exceeds cap");
    }
    if (cache) cache->insert(location, key, value, counts);
    return value;
}

struct Conditional {
    double p_one, indicator, denominator;
};

Conditional event_conditional(State const& state, std::vector<int> const& retained,
                              int mode, Counters& counts, Profile& profile,
                              Context& context, KernelRole role,
                              StepVacuumCache* cache = nullptr) {
    Timer timer(profile.enabled,
                role == KernelRole::selection ? profile.selection_event : profile.sampling_event);
    std::vector<int> modes = retained;
    modes.push_back(mode);
    context.kernel_modes = modes;
    if (modes.size() > max_kernel_modes) throw std::logic_error("Event kernel too large");
    if (role == KernelRole::selection) ++counts.selection_conditional_kernel_calls;
    else ++counts.conditional_evaluations;
    const unsigned int half = 1U << retained.size();
    std::array<long double, 1U << max_kernel_modes> probabilities{};
    probabilities[0] = 1.0L;
    for (unsigned int mask = 1; mask < 2*half; ++mask)
        probabilities[mask] = vacuum(state, modes, mask, counts, role, cache);
    long double denominator = 0, numerator = 0, den_abs = 0, num_abs = 0;
    for (unsigned int mask = 0; mask < half; ++mask) {
        const int sign = std::popcount(mask) % 2 ? -1 : 1;
        denominator += sign * probabilities[mask];
        numerator += sign * probabilities[mask+half];
        den_abs += std::abs(probabilities[mask]);
        num_abs += std::abs(probabilities[mask+half]);
    }
    const long double factor = 8.0 * std::numeric_limits<double>::epsilon()
        * std::max(1, state.zero_count + static_cast<int>(modes.size()));
    const long double den_indicator = factor*den_abs, num_indicator = factor*num_abs;
    context.last_denominator = static_cast<double>(denominator);
    if (numerator < 0) {
        if (-numerator > num_indicator)
            counts.fail("Negative conditioned event exceeds roundoff indicator");
        numerator = 0;
        ++counts.event_corrections;
    }
    if (!(denominator > 0))
        counts.fail("Retained-click event has zero or numerically unresolved probability");
    long double p_zero = numerator / denominator;
    const double indicator = static_cast<double>((num_indicator + std::abs(p_zero)*den_indicator) / denominator);
    context.last_indicator = indicator;
    counts.maximum_cancellation_indicator = std::max(counts.maximum_cancellation_indicator, indicator);
    counts.minimum_conditioning_probability = std::min(counts.minimum_conditioning_probability,
                                                       static_cast<double>(denominator));
    if (!std::isfinite(p_zero) || !std::isfinite(indicator))
        counts.fail("Nonfinite conditioned-event ratio or cancellation indicator");
    if (indicator > max_relative_roundoff) {
        std::ostringstream message;
        message << "Conditioned-event cancellation indicator " << std::setprecision(17)
                << indicator << " exceeds " << max_relative_roundoff;
        counts.fail(message.str());
    }
    if (p_zero < 0 || p_zero > 1) {
        const long double deviation = std::max(-p_zero, p_zero-1.0L);
        if (deviation > indicator)
            counts.fail("Conditioned no-click probability outside [0,1] beyond its guard");
        p_zero = std::clamp(p_zero, 0.0L, 1.0L);
        ++counts.probability_corrections;
    }
    return {static_cast<double>(1.0L-p_zero), indicator, static_cast<double>(denominator)};
}

class Sampler {
public:
    Options const& option;
    Matrix base, pairwise_mi;
    double minimum_uncertainty_eigenvalue = 0;
    double covariance_symmetry_correction = 0;
    Counters setup_counts, sampling_counts, enumeration_counts;
    Profile sampling_profile, enumeration_profile;
    Context& context;
    StepVacuumCache step_cache;

    Sampler(Options const& option_, Matrix covariance, Context& context_)
        : option(option_), base(std::move(covariance)),
          sampling_profile(option.profile), enumeration_profile(option.profile), context(context_) {
        const double scale = std::max(1.0, base.cwiseAbs().maxCoeff());
        const double asymmetry = (base-base.transpose()).cwiseAbs().maxCoeff();
        if (asymmetry > 1e-10*scale)
            throw std::invalid_argument("Covariance is not symmetric within tolerance");
        covariance_symmetry_correction = 0.5*asymmetry;
        base = (0.5*(base+base.transpose())).eval();
        Eigen::MatrixXcd uncertainty = base.cast<std::complex<double>>();
        for (int j = 0; j < option.modes; ++j) {
            uncertainty(j,j+option.modes) += std::complex<double>(0,1);
            uncertainty(j+option.modes,j) -= std::complex<double>(0,1);
        }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> solver(uncertainty, Eigen::EigenvaluesOnly);
        if (solver.info() != Eigen::Success)
            throw std::invalid_argument("Covariance physicality eigensolve failed");
        minimum_uncertainty_eigenvalue = solver.eigenvalues()(0);
        if (!std::isfinite(minimum_uncertainty_eigenvalue)
                || minimum_uncertainty_eigenvalue < -1e-9*scale)
            throw std::invalid_argument("Covariance violates Gaussian uncertainty relation");
        if (option.selector == "pairwise" && option.k > 0) precompute_pairwise();
    }

    void precompute_pairwise() {
        const State state(base);
        pairwise_mi = Matrix::Zero(option.modes, option.modes);
        std::vector<double> singles(option.modes);
        for (int j = 0; j < option.modes; ++j)
            singles[j] = vacuum(state, std::vector<int>{j}, 1, setup_counts, KernelRole::setup);
        for (int first = 0; first < option.modes; ++first) {
            for (int second = first+1; second < option.modes; ++second) {
                const long double a = singles[first], b = singles[second];
                const long double c = vacuum(state, std::vector<int>{first,second}, 3,
                                             setup_counts, KernelRole::setup);
                // The same two-bit upper-set Mobius inversion as ThresholdModel.
                std::array<double,4> joint{static_cast<double>(c), static_cast<double>(b-c),
                    static_cast<double>(a-c), static_cast<double>((1.0L-a)-(b-c))};
                const auto raw = joint;
                long double sum = 0;
                for (double p : joint) {
                    if (!std::isfinite(p) || p < -probability_tolerance)
                        setup_counts.fail("Invalid unconditional pair marginal");
                    sum += p;
                }
                if (std::abs(sum-1.0L) > probability_tolerance)
                    setup_counts.fail("Pair marginal normalization error exceeds tolerance");
                sum = 0;
                for (double& p : joint) { p = std::max(0.0,p); sum += p; }
                double correction = 0;
                for (std::size_t j = 0; j < 4; ++j) {
                    joint[j] = static_cast<double>(joint[j]/sum);
                    correction += std::abs(joint[j]-raw[j]);
                }
                if (correction > 10*probability_tolerance)
                    setup_counts.fail("Pair marginal correction exceeds tolerance");
                if (correction) ++setup_counts.pairwise_probability_corrections;
                setup_counts.maximum_pairwise_l1_correction = std::max(
                    setup_counts.maximum_pairwise_l1_correction, correction);
                const std::array<double,2> first_marginal{joint[0]+joint[2],joint[1]+joint[3]};
                const std::array<double,2> second_marginal{joint[0]+joint[1],joint[2]+joint[3]};
                double mi = 0;
                for (int bits = 0; bits < 4; ++bits) if (joint[bits] > 0) {
                    const double product = first_marginal[bits&1]*second_marginal[bits>>1];
                    mi += joint[bits]*std::log(joint[bits]/product);
                }
                if (!std::isfinite(mi) || mi < -1e-12)
                    setup_counts.fail("Invalid unconditional pairwise mutual information");
                pairwise_mi(first,second) = pairwise_mi(second,first) = std::max(0.0,mi);
                ++setup_counts.pairwise_marginals_requested;
            }
        }
    }

    std::vector<int> select(State const& state, std::vector<int> const& clicks,
                            int mode, Counters& counts, Profile& profile,
                            StepVacuumCache* cache) {
        ++counts.selection_requests;
        if (option.k == 0) return {};
        if (clicks.size() <= static_cast<std::size_t>(option.k)) return clicks;
        Timer timer(profile.enabled, profile.selection_inclusive);
        ++counts.selection_computations;
        counts.maximum_selection_candidates = std::max(counts.maximum_selection_candidates,
                                                        static_cast<int>(clicks.size()));
        if (option.selector == "pairwise") {
            auto selected = clicks;
            std::sort(selected.begin(), selected.end(), [&](int a, int b) {
                const double first = pairwise_mi(mode,a), second = pairwise_mi(mode,b);
                return first == second ? a < b : first > second;
            });
            selected.resize(option.k);
            return selected;
        }
        auto available = clicks;
        std::sort(available.begin(), available.end());
        std::vector<int> selected;
        double baseline = event_conditional(state, selected, mode, counts, profile,
                                            context, KernelRole::selection, cache).p_one;
        for (int round = 0; round < option.k; ++round) {
            int best = -1;
            double best_score = -1, best_probability = 0;
            for (int candidate : available) {
                auto retained = selected;
                retained.push_back(candidate);
                const double probability = event_conditional(state, retained, mode, counts,
                    profile, context, KernelRole::selection, cache).p_one;
                ++counts.selection_score_evaluations;
                const double score = std::abs(probability-baseline);
                if (score > best_score || (score == best_score && candidate < best)) {
                    best = candidate;
                    best_score = score;
                    best_probability = probability;
                }
            }
            if (best < 0) throw std::logic_error("Influence selector found no candidate");
            selected.push_back(best);
            baseline = best_probability;
            available.erase(std::find(available.begin(), available.end(), best));
        }
        return selected;
    }

    Conditional conditional(State const& state, std::vector<int> const& clicks,
                            int mode, Counters& counts, Profile& profile,
                            std::vector<int>& retained) {
        context.mode = mode;
        context.zero_count = state.zero_count;
        context.click_count = static_cast<int>(clicks.size());
        StepVacuumCache* cache = nullptr;
        if (option.step_cache && option.selector == "click_influence" && option.k > 0
                && clicks.size() > static_cast<std::size_t>(option.k)) {
            Timer cache_timer(profile.enabled, profile.scratch_cache);
            step_cache.reset(clicks.size()*((1U << (option.k+1))-1), counts);
            cache = &step_cache;
        }
        retained = select(state, clicks, mode, counts, profile, cache);
        // SchurMemorySampler sorts the final retained set before its kernel.
        std::sort(retained.begin(), retained.end());
        const auto result = event_conditional(state, retained, mode, counts, profile,
                                               context, KernelRole::sampling, cache);
        if (!std::isfinite(result.p_one) || result.p_one < 0 || result.p_one > 1)
            counts.fail("Invalid final click probability");
        counts.minimum_click_probability = std::min(counts.minimum_click_probability, result.p_one);
        counts.maximum_click_probability = std::max(counts.maximum_click_probability, result.p_one);
        return result;
    }

    void advance(State& state, std::vector<int>& clicks, int mode, bool one,
                 Counters& counts, Profile& profile) {
        if (one) clicks.push_back(mode);
        else state.project_zero(mode, counts, profile);
    }

    std::vector<double> enumerate() {
        if (option.modes > enumeration_cap) throw std::invalid_argument("Enumeration cap exceeded");
        // Every bit shift below is guarded by M<=14, including physical mode IDs.
        std::vector<double> probabilities(std::size_t{1} << option.modes, 0.0);
        std::vector<int> clicks;
        std::function<void(int,State const&,std::uint64_t,double)> visit;
        visit = [&](int position, State const& state, std::uint64_t mask, double weight) {
            if (position == option.modes) { probabilities[mask] = weight; return; }
            context.position = position;
            const int mode = option.order[position];
            std::vector<int> retained;
            const auto result = conditional(state, clicks, mode, enumeration_counts,
                                           enumeration_profile, retained);
            if (position+1 == option.modes) {
                if (result.p_one < 1) probabilities[mask] = weight*(1.0-result.p_one);
                if (result.p_one > 0)
                    probabilities[mask | (std::uint64_t{1} << mode)] = weight*result.p_one;
                return;
            }
            if (result.p_one < 1) {
                State next = state;
                advance(next, clicks, mode, false, enumeration_counts, enumeration_profile);
                visit(position+1, next, mask, weight*(1.0-result.p_one));
            }
            if (result.p_one > 0) {
                // Outcome one changes only the click history, never the Gaussian state.
                clicks.push_back(mode);
                visit(position+1, state, mask | (std::uint64_t{1} << mode), weight*result.p_one);
                clicks.pop_back();
            }
        };
        const State initial(base);
        visit(0, initial, 0, 1.0);
        long double sum = 0;
        for (double p : probabilities) {
            if (!std::isfinite(p) || p < 0) enumeration_counts.fail("Invalid enumerated probability");
            sum += p;
        }
        if (std::abs(sum-1.0L) > 1e-11L)
            enumeration_counts.fail("Enumerated law failed normalization without correction");
        return probabilities;
    }
};

struct TraceRow {
    std::size_t sample_index;
    int position, mode;
    std::vector<int> retained;
    double p_one, cancellation_indicator;
};

void print_context(Context const& context) {
    std::cout << "{\"phase\":" << quote(context.phase)
        << ",\"sample_index\":" << context.sample_index
        << ",\"position\":" << context.position << ",\"mode\":" << context.mode
        << ",\"zero_count\":" << context.zero_count << ",\"click_count\":" << context.click_count
        << ",\"kernel_modes\":";
    print_vector(context.kernel_modes);
    std::cout << ",\"cancellation_indicator\":";
    print_finite_or_null(context.last_indicator);
    std::cout << ",\"conditioning_probability\":";
    print_finite_or_null(context.last_denominator);
    std::cout << '}';
}

int run(int argc, char** argv) {
    std::cout << std::setprecision(17);
    Context context;
    std::unique_ptr<Sampler> sampler;
    std::size_t completed_samples = 0;
    try {
        const auto option = parse_options(argc, argv);
        context.phase = "input";
        auto started = Clock::now();
        const int dimension = 2*option.modes;
        const auto values = read_csv(option.input, dimension, dimension);
        Matrix covariance(dimension,dimension);
        for (int row = 0; row < dimension; ++row)
            for (int col = 0; col < dimension; ++col)
                covariance(row,col) = values[row*dimension+col];
        std::vector<double> uniforms;
        if (!option.uniforms.empty()) {
            uniforms = read_csv(option.uniforms, option.samples, option.modes);
            for (double value : uniforms)
                if (!(value >= 0 && value < 1))
                    throw std::invalid_argument("Injected uniforms must lie in [0,1)");
        }
        const double input_seconds = elapsed(started);
        context.phase = "setup";
        started = Clock::now();
        sampler = std::make_unique<Sampler>(option, std::move(covariance), context);
        const double setup_seconds = elapsed(started);
        context.phase = "sampling";
        started = Clock::now();
        // Materialize draws even when raw JSON output is not requested, just
        // as the official baseline does. Include RNG/buffer initialization.
        const std::size_t words_per_sample = (option.modes+63)/64;
        std::vector<std::uint64_t> packed_samples(option.samples*words_per_sample,0);
        std::vector<TraceRow> trace;
        if (option.trace) trace.reserve(option.samples*option.modes);
        std::mt19937_64 engine(option.seed);
        for (std::size_t sample_index = 0; sample_index < option.samples; ++sample_index) {
            context.sample_index = sample_index;
            State state(sampler->base);
            std::vector<int> clicks;
            for (int position = 0; position < option.modes; ++position) {
                context.position = position;
                const int mode = option.order[position];
                std::vector<int> retained;
                const auto result = sampler->conditional(state, clicks, mode,
                    sampler->sampling_counts, sampler->sampling_profile, retained);
                const double uniform = uniforms.empty()
                    ? static_cast<double>(engine() >> 11) * 0x1.0p-53
                    : uniforms[sample_index*option.modes+position];
                const bool one = uniform < result.p_one;
                if (option.trace) trace.push_back({sample_index, position, mode,
                    retained, result.p_one, result.indicator});
                if (one) packed_samples[sample_index*words_per_sample+mode/64]
                    |= std::uint64_t{1} << (mode%64);
                // No later probability needs the final outcome's updated state.
                if (position+1 < option.modes)
                    sampler->advance(state, clicks, mode, one,
                        sampler->sampling_counts, sampler->sampling_profile);
            }
            ++completed_samples;
        }
        const double sampling_seconds = elapsed(started);
        context.phase = "aggregation";
        started = Clock::now();
        std::vector<std::uint64_t> histogram;
        if (option.modes <= enumeration_cap)
            histogram.resize(std::size_t{1} << option.modes, 0);
        std::vector<std::uint64_t> click_histogram(option.modes+1,0), mode_counts(option.modes,0);
        for (std::size_t i = 0; i < option.samples; ++i) {
            const auto start = i*words_per_sample;
            int total_clicks = 0;
            for (std::size_t word = 0; word < words_per_sample; ++word)
                total_clicks += std::popcount(packed_samples[start+word]);
            ++click_histogram[total_clicks];
            for (int mode = 0; mode < option.modes; ++mode)
                mode_counts[mode] += (packed_samples[start+mode/64] >> (mode%64)) & 1U;
            // A dense histogram exists only for M<=14; its mask fits word zero.
            if (!histogram.empty()) ++histogram[packed_samples[start]];
        }
        const double aggregation_seconds = elapsed(started);
        std::vector<double> distribution;
        double enumeration_seconds = 0;
        if (option.enumerate) {
            context.phase = "enumeration";
            started = Clock::now();
            distribution = sampler->enumerate();
            enumeration_seconds = elapsed(started);
        }
        std::cout << "{\"schema_version\":1,\"status\":\"ok\""
            << ",\"implementation\":\"C++ retained-click sampler\""
            << ",\"modes\":" << option.modes << ",\"selector\":" << quote(option.selector)
            << ",\"k\":" << option.k << ",\"sample_count\":" << option.samples
            << ",\"seed\":" << option.seed << ",\"threads\":1,\"order\":";
        print_vector(option.order);
        std::cout << ",\"covariance_convention\":\"xxpp; hbar=2; vacuum=I; zero displacement\""
            << ",\"mask_convention\":\"bit i is physical mode i; one is a click\""
            << ",\"uniform_source\":" << quote(option.uniforms.empty() ? "mt19937_64 upper 53 bits / 2^53" : "injected CSV by measurement step")
            << ",\"profile_enabled\":" << (option.profile ? "true" : "false")
            << ",\"step_cache_enabled\":" << (option.step_cache ? "true" : "false")
            << ",\"step_cache_entry_cap\":" << step_cache_entry_cap
            << ",\"step_cache_scope\":\"one conditional; ordered physical subsets; no cross-step reuse\""
            << ",\"sample_storage\":{\"format\":\"packed uint64 physical bits\",\"words_per_sample\":"
            << words_per_sample << ",\"bytes\":" << packed_samples.size()*sizeof(std::uint64_t)
            << ",\"materialized_without_raw_output\":true}"
            << ",\"enumerated_law\":\"product of computed conditional probabilities under ideal continuous uniforms; not a finite PRNG-state enumeration\""
            << ",\"max_relative_roundoff\":" << max_relative_roundoff
            << ",\"probability_tolerance\":" << probability_tolerance
            << ",\"minimum_uncertainty_eigenvalue\":" << sampler->minimum_uncertainty_eigenvalue
            << ",\"covariance_symmetry_correction\":" << sampler->covariance_symmetry_correction
            << ",\"timings_seconds\":{\"input\":" << input_seconds
            << ",\"setup\":" << setup_seconds << ",\"sampling\":" << sampling_seconds
            << ",\"cold_batch\":" << setup_seconds+sampling_seconds
            << ",\"aggregation\":" << aggregation_seconds
            << ",\"enumeration\":" << enumeration_seconds << '}'
            << ",\"diagnostics\":";
        sampler->sampling_counts.print();
        std::cout << ",\"sampling_diagnostics\":"; sampler->sampling_counts.print();
        std::cout << ",\"setup_diagnostics\":"; sampler->setup_counts.print();
        std::cout << ",\"enumeration_diagnostics\":"; sampler->enumeration_counts.print();
        std::cout << ",\"sampling_profile\":"; sampler->sampling_profile.print();
        std::cout << ",\"enumeration_profile\":"; sampler->enumeration_profile.print();
        std::cout << ",\"profiling_notes\":\"Selection inclusive contains selection event time; use overhead plus event for nonoverlapping totals. Scratch-cache preparation is separate; loop/output work is unclassified. Timers run only with --profile.\""
            << ",\"histogram_counts\":";
        if (histogram.empty()) std::cout << "null"; else print_vector(histogram);
        std::cout << ",\"click_count_histogram\":"; print_vector(click_histogram);
        std::cout << ",\"mode_click_counts\":"; print_vector(mode_counts);
        std::cout << ",\"distribution\":";
        if (option.enumerate) print_vector(distribution); else std::cout << "null";
        std::cout << ",\"raw_samples\":";
        if (option.raw_samples) {
            std::cout << '[';
            for (std::size_t i = 0; i < option.samples; ++i) {
                if (i) std::cout << ',';
                std::cout << '[';
                for (int j = 0; j < option.modes; ++j) {
                    if (j) std::cout << ',';
                    std::cout << ((packed_samples[i*words_per_sample+j/64] >> (j%64)) & 1U);
                }
                std::cout << ']';
            }
            std::cout << ']';
        } else std::cout << "null";
        std::cout << ",\"conditional_trace\":";
        if (option.trace) {
            std::cout << '[';
            for (std::size_t i = 0; i < trace.size(); ++i) {
                if (i) std::cout << ',';
                auto const& row = trace[i];
                std::cout << "{\"sample_index\":" << row.sample_index
                    << ",\"position\":" << row.position << ",\"mode\":" << row.mode
                    << ",\"retained\":";
                print_vector(row.retained);
                std::cout << ",\"p_one\":" << row.p_one
                    << ",\"cancellation_indicator\":" << row.cancellation_indicator << '}';
            }
            std::cout << ']';
        } else std::cout << "null";
        std::cout << "}\n";
        return 0;
    } catch (std::exception const& error) {
        const auto* numerical = dynamic_cast<NumericalProbabilityError const*>(&error);
        const auto* invalid = dynamic_cast<std::invalid_argument const*>(&error);
        const auto* range = dynamic_cast<std::out_of_range const*>(&error);
        const std::string type = numerical ? "numerical_probability"
            : (invalid || range ? "invalid_argument" : "internal_error");
        std::cout << "{\"schema_version\":1,\"status\":\"error\",\"error_type\":"
            << quote(type) << ",\"error\":" << quote(error.what())
            << ",\"completed_samples\":" << completed_samples << ",\"context\":";
        print_context(context);
        if (sampler) {
            std::cout << ",\"diagnostics\":";
            if (context.phase == "enumeration") sampler->enumeration_counts.print();
            else if (context.phase == "setup") sampler->setup_counts.print();
            else sampler->sampling_counts.print();
        }
        std::cout << "}\n";
        return numerical ? 2 : 1;
    }
}
} // namespace

int main(int argc, char** argv) {
    Eigen::setNbThreads(1);
    return run(argc, argv);
}
