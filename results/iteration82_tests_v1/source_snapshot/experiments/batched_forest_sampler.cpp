// Complete native forest sampler.  The legacy selector is included unchanged;
// the shared event/precision policy is linked as a separate translation unit.
// No probability prefix cache, resampling, or successful-only normalization.
#define main legacy_retained_main
#include "click_sampler.cpp"
#undef main
#include "batched_forest_kernel.h"
#include <tuple>

namespace forest_cli {

struct Configuration {
    Options legacy;
    std::string forest = "chain", precision = "auto";
    int precision_code = BF_AUTO;
};

Configuration configuration(int argc, char** argv) {
    Configuration result;
    std::vector<std::string> arguments{argv[0]};
    bool has_selector = false, has_kernel = false;
    for (int j=1; j<argc; ++j) {
        const std::string flag = argv[j];
        if (flag == "--help") {
            std::cout << "batched_forest_sampler --input covariance.csv --modes M --k K "
                "[--forest chain|static_mi_tree --precision auto|longdouble|quad "
                "--selector click_influence|pairwise --samples N --seed S "
                "--uniforms matrix.csv --enumerate --raw-samples "
                "--trace-conditionals --record-log-probabilities --profile "
                "--vacuum-kernel product|log --no-step-cache]\n";
            std::exit(0);
        }
        if (flag == "--forest" || flag == "--precision") {
            if (++j >= argc) throw std::invalid_argument("Missing value for " + flag);
            if (flag == "--forest") result.forest = argv[j];
            else result.precision = argv[j];
        } else {
            arguments.push_back(flag);
            has_selector = has_selector || flag == "--selector";
            has_kernel = has_kernel || flag == "--vacuum-kernel";
        }
    }
    if (!has_selector) { arguments.push_back("--selector"); arguments.push_back("click_influence"); }
    if (!has_kernel) { arguments.push_back("--vacuum-kernel"); arguments.push_back("product"); }
    std::vector<char*> pointers;
    for (auto& value : arguments) pointers.push_back(value.data());
    result.legacy = parse_options(static_cast<int>(pointers.size()), pointers.data());
    if (result.forest != "chain" && result.forest != "static_mi_tree")
        throw std::invalid_argument("forest must be chain or static_mi_tree");
    if (result.precision == "auto") result.precision_code = BF_AUTO;
    else if (result.precision == "longdouble") result.precision_code = BF_LONG_DOUBLE;
    else if (result.precision == "quad") result.precision_code = BF_QUAD;
    else throw std::invalid_argument("precision must be auto, longdouble, or quad");
    if (result.legacy.prefix_cache_entries != 0)
        throw std::invalid_argument("forest CLI requires prefix_cache_entries=0");
    for (int j=0; j<result.legacy.modes; ++j)
        if (result.legacy.order[j] != j)
            throw std::invalid_argument("forest CLI currently requires natural measurement order");
    return result;
}

void finite_ld(long double value) {
    const double converted = static_cast<double>(value);
    print_finite_or_null(converted);
}

struct Step {
    std::size_t sample_index = 0;
    int position = 0, mode = 0;
    std::vector<int> prefix, retained, selection_order, omitted, edges;
    BFConditional conditional{};
    bool accepted = false;
    std::string error;
};

void print_edges(std::vector<int> const& edges) {
    std::cout << '[';
    for (std::size_t j=0; j<edges.size(); j+=2) {
        if (j) std::cout << ',';
        std::cout << '[' << edges[j] << ',' << edges[j+1] << ']';
    }
    std::cout << ']';
}

void print_step(Step const& step) {
    std::cout << "{\"sample_index\":" << step.sample_index
        << ",\"position\":" << step.position << ",\"mode\":" << step.mode
        << ",\"prefix\":"; print_vector(step.prefix);
    std::cout << ",\"retained_clicks\":"; print_vector(step.retained);
    std::cout << ",\"selection_order\":"; print_vector(step.selection_order);
    std::cout << ",\"omitted_clicks\":"; print_vector(step.omitted);
    std::cout << ",\"forest_edges\":"; print_edges(step.edges);
    std::cout << ",\"accepted\":" << (step.accepted ? "true" : "false")
        << ",\"p_one\":";
    if (step.accepted) std::cout << step.conditional.probability; else std::cout << "null";
    std::cout << ",\"probability\":";
    if (step.accepted) std::cout << step.conditional.probability; else std::cout << "null";
    std::cout << ",\"precision_used\":"
        << quote(step.conditional.precision_used == BF_QUAD ? "quad" : "longdouble")
        << ",\"refinement_attempted\":"
        << (step.conditional.refinement_attempted ? "true" : "false")
        << ",\"zero_score_reasons\":[" << step.conditional.zero_score_reason[0]
        << ',' << step.conditional.zero_score_reason[1] << ']'
        << ",\"log_scores\":[";
    for (int a=0; a<2; ++a) {
        if (a) std::cout << ',';
        if (step.accepted && !step.conditional.zero_score_reason[a])
            finite_ld(step.conditional.log_scores[a]);
        else std::cout << "null";
    }
    std::cout << "],\"maximum_event_relative_indicator\":";
    finite_ld(step.conditional.maximum_event_relative_indicator);
    std::cout << ",\"maximum_log_score_indicator\":";
    finite_ld(step.conditional.maximum_log_score_indicator);
    std::cout << ",\"first_error\":" << quote(step.conditional.first_error)
        << ",\"error\":" << (step.error.empty() ? "null" : quote(step.error)) << '}';
}

void print_steps(std::vector<Step> const& rows) {
    std::cout << '[';
    for (std::size_t j=0; j<rows.size(); ++j) {
        if (j) std::cout << ',';
        print_step(rows[j]);
    }
    std::cout << ']';
}

struct TreeCounters {
    std::uint64_t calls=0, vertices=0, edges=0, weight_lookups=0, cut_candidates=0, comparisons=0;
    double seconds=0;
    void print() const {
        std::cout << "{\"calls\":" << calls << ",\"vertices\":" << vertices
            << ",\"edges\":" << edges << ",\"weight_lookups\":" << weight_lookups
            << ",\"cut_candidates\":" << cut_candidates << ",\"comparisons\":"
            << comparisons << ",\"seconds\":" << seconds << '}';
    }
};

struct KernelDeleter { void operator()(BFHandle* handle) const { if (handle) bf_destroy(handle); } };
using Kernel = std::unique_ptr<BFHandle,KernelDeleter>;

class Engine {
public:
    Configuration const& config;
    Context& context;
    Sampler selector;
    Kernel kernel;
    TreeCounters sampling_tree, enumeration_tree;
    Step last_step;

    Engine(Configuration const& config_, Matrix covariance, Context& context_)
        : config(config_), context(context_), selector(config.legacy,std::move(covariance),context) {
        BFFormats formats{};
        if (bf_formats(&formats) != BF_OK || formats.abi_version != BF_ABI_VERSION
                || formats.query_bytes != sizeof(BFQuery)
                || formats.output_bytes != sizeof(BFOutput)
                || formats.conditional_bytes != sizeof(BFConditional)
                || formats.diagnostics_bytes != sizeof(BFDiagnostics))
            throw std::runtime_error("shared forest kernel ABI mismatch");
        if (config.forest == "static_mi_tree" && selector.pairwise_mi.size() == 0)
            selector.precompute_pairwise();
        const int dimension = 2*config.legacy.modes;
        std::vector<double> row_major(static_cast<std::size_t>(dimension)*dimension);
        for (int row=0; row<dimension; ++row)
            for (int column=0; column<dimension; ++column)
                row_major[row*dimension+column] = selector.base(row,column);
        char error[BF_ERROR_CAPACITY]{};
        kernel.reset(bf_create(config.legacy.modes,row_major.data(),config.legacy.order.data(),error,sizeof error));
        if (!kernel) throw std::runtime_error(std::string("kernel setup failed: ")+error);
    }

    std::vector<int> choose_edges(std::vector<int> const& omitted, TreeCounters& counts) {
        const auto started = Clock::now();
        ++counts.calls; counts.vertices += omitted.size();
        std::vector<int> edges;
        if (omitted.size() >= 2 && config.forest == "chain") {
            for (std::size_t j=1; j<omitted.size(); ++j) {
                edges.push_back(omitted[j-1]); edges.push_back(omitted[j]);
            }
        } else if (omitted.size() >= 2) {
            auto vertices = omitted;
            std::sort(vertices.begin(),vertices.end());
            using Rank = std::tuple<double,int,int>;
            auto rank = [&](int first,int second) {
                ++counts.weight_lookups;
                const double weight = selector.pairwise_mi(first,second);
                if (!std::isfinite(weight) || weight < 0)
                    throw std::logic_error("invalid static-MI edge weight");
                return Rank{-weight,std::min(first,second),std::max(first,second)};
            };
            const std::size_t n=vertices.size();
            std::vector<Rank> best(n);
            std::vector<int> parent(n,vertices[0]);
            std::vector<bool> used(n,false);
            used[0]=true;
            for (std::size_t j=1;j<n;++j) best[j]=rank(vertices[0],vertices[j]);
            for (std::size_t round=1;round<n;++round) {
                std::size_t chosen=n;
                for (std::size_t j=1;j<n;++j) if (!used[j]) {
                    ++counts.cut_candidates;
                    if (chosen==n || best[j]<best[chosen]) chosen=j;
                }
                if (chosen==n) throw std::logic_error("static-MI tree did not span omitted vertices");
                edges.push_back(std::min(parent[chosen],vertices[chosen]));
                edges.push_back(std::max(parent[chosen],vertices[chosen]));
                used[chosen]=true;
                for (std::size_t j=1;j<n;++j) if (!used[j]) {
                    const auto candidate=rank(vertices[chosen],vertices[j]);
                    ++counts.comparisons;
                    if (candidate<best[j]) { best[j]=candidate; parent[j]=vertices[chosen]; }
                }
            }
        }
        if (edges.size()/2 != (omitted.empty() ? 0 : omitted.size()-1))
            throw std::logic_error("forest has an incorrect spanning edge count");
        counts.edges += edges.size()/2;
        counts.seconds += elapsed(started);
        return edges;
    }

    double conditional(State const& state, std::vector<int> const& clicks,
            std::vector<int> const& prefix, Counters& counts, Profile& profile,
            TreeCounters& tree_counts) {
        const int mode=static_cast<int>(prefix.size());
        context.position=mode; context.mode=mode;
        context.zero_count=state.zero_count; context.click_count=static_cast<int>(clicks.size());
        context.last_indicator=0; context.last_denominator=1;
        last_step=Step{};
        last_step.sample_index=context.sample_index;
        last_step.position=mode; last_step.mode=mode; last_step.prefix=prefix;
        StepVacuumCache* cache=nullptr;
        if (config.legacy.step_cache && config.legacy.selector=="click_influence"
                && config.legacy.k>0 && clicks.size()>static_cast<std::size_t>(config.legacy.k)) {
            Timer cache_timer(profile.enabled,profile.scratch_cache);
            selector.step_cache.reset(clicks.size()*((1U<<(config.legacy.k+1))-1),counts);
            cache=&selector.step_cache;
        }
        try {
            last_step.selection_order=selector.select(state,clicks,mode,counts,profile,cache);
            last_step.retained=last_step.selection_order;
            std::sort(last_step.retained.begin(),last_step.retained.end());
            for (int click : clicks)
                if (!std::binary_search(last_step.retained.begin(),last_step.retained.end(),click))
                    last_step.omitted.push_back(click);
            last_step.edges=choose_edges(last_step.omitted,tree_counts);
            std::vector<int> zeros;
            for (int j=0;j<mode;++j) if (!prefix[j]) zeros.push_back(j);
            context.kernel_modes=last_step.retained; context.kernel_modes.push_back(mode);
            const BFQuery query{zeros.data(),last_step.retained.data(),last_step.omitted.data(),
                last_step.edges.data(),mode,static_cast<int>(zeros.size()),
                static_cast<int>(last_step.retained.size()),static_cast<int>(last_step.omitted.size()),
                static_cast<int>(last_step.edges.size()/2)};
            const std::size_t size=1+2*(1+last_step.omitted.size()+last_step.edges.size()/2);
            std::vector<long double> values(size),indicators(size);
            std::vector<std::uint8_t> exact_zero(size);
            BFOutput output{values.data(),indicators.data(),exact_zero.data(),size};
            char error[BF_ERROR_CAPACITY]{};
            const int status=bf_probability(kernel.get(),&query,config.precision_code,
                static_cast<long double>(max_relative_roundoff),&output,&last_step.conditional,error,sizeof error);
            context.last_indicator=static_cast<double>(last_step.conditional.maximum_log_score_indicator);
            context.last_denominator=static_cast<double>(values[0]);
            if (status==BF_NUMERICAL_ERROR) throw NumericalProbabilityError(error);
            if (status!=BF_OK) throw std::runtime_error("forest kernel status "+std::to_string(status)+": "+error);
            const double probability=last_step.conditional.probability;
            if (!std::isfinite(probability) || probability<0 || probability>1)
                throw std::logic_error("shared kernel returned invalid accepted probability");
            ++counts.conditional_evaluations;
            counts.minimum_click_probability=std::min(counts.minimum_click_probability,probability);
            counts.maximum_click_probability=std::max(counts.maximum_click_probability,probability);
            last_step.accepted=true;
            return probability;
        } catch (std::exception const& error) {
            last_step.error=error.what();
            throw;
        }
    }
};

struct Failure {
    std::vector<int> prefix;
    int position=0;
    std::string phase,error;
    long double mass=0;
};

void print_failures(std::vector<Failure> const& failures) {
    std::cout << '[';
    for (std::size_t j=0;j<failures.size();++j) {
        if (j) std::cout << ',';
        auto const& failure=failures[j];
        std::cout << "{\"prefix\":"; print_vector(failure.prefix);
        std::cout << ",\"position\":" << failure.position << ",\"phase\":"
            << quote(failure.phase) << ",\"error\":" << quote(failure.error)
            << ",\"mass\":"; finite_ld(failure.mass); std::cout << '}';
    }
    std::cout << ']';
}

struct Enumeration {
    std::vector<double> distribution;
    std::vector<Failure> failures;
    std::vector<Step> trace;
    long double failure_mass=0;
};

double represented_mass(long double value) {
    const double result=static_cast<double>(value);
    if (!std::isfinite(value) || value<0 || !std::isfinite(result) || (value>0 && result==0))
        throw std::runtime_error("enumeration weight cannot be represented; enumeration aborted");
    return result;
}

Enumeration enumerate(Engine& engine) {
    Enumeration result;
    const int modes=engine.config.legacy.modes;
    result.distribution.assign(std::size_t{1}<<modes,0.0);
    bf_reset(engine.kernel.get());
    auto record_failure=[&](std::vector<int> const& prefix,int position,std::string phase,
                            std::string error,long double mass) {
        represented_mass(mass);
        result.failure_mass+=mass;
        result.failures.push_back({prefix,position,std::move(phase),std::move(error),mass});
    };
    std::function<void(State const&,std::vector<int> const&,std::vector<int>&,std::uint64_t,long double)> visit;
    visit=[&](State const& state,std::vector<int> const& clicks,std::vector<int>& prefix,
              std::uint64_t mask,long double weight) {
        const int position=static_cast<int>(prefix.size());
        if (position==modes) { result.distribution[mask]=represented_mass(weight); return; }
        represented_mass(weight);
        double probability;
        try {
            probability=engine.conditional(state,clicks,prefix,engine.selector.enumeration_counts,
                engine.selector.enumeration_profile,engine.enumeration_tree);
        } catch (NumericalProbabilityError const& error) {
            if (engine.config.legacy.trace) result.trace.push_back(engine.last_step);
            record_failure(prefix,position,"conditional",error.what(),weight);
            return;
        }
        if (engine.config.legacy.trace) result.trace.push_back(engine.last_step);
        for (int one=0;one<2;++one) {
            const long double factor=one ? static_cast<long double>(probability) : 1.0L-probability;
            if (factor==0) continue;
            const long double child_weight=weight*factor;
            if (!(child_weight>0)) throw std::runtime_error("positive enumeration branch underflowed");
            prefix.push_back(one);
            const auto child_mask=mask | (one ? std::uint64_t{1}<<position : 0);
            if (position+1==modes) {
                result.distribution[child_mask]=represented_mass(child_weight);
            } else if (one) {
                auto child_clicks=clicks; child_clicks.push_back(position);
                visit(state,child_clicks,prefix,child_mask,child_weight);
            } else {
                State child_state=state;
                bool updated=false;
                try {
                    child_state.project_zero(position,engine.selector.enumeration_counts,
                                             engine.selector.enumeration_profile);
                    updated=true;
                } catch (NumericalProbabilityError const& error) {
                    record_failure(prefix,position,"zero_state_update",error.what(),child_weight);
                }
                if (updated) visit(child_state,clicks,prefix,child_mask,child_weight);
            }
            prefix.pop_back();
        }
    };
    const State initial(engine.selector.base,engine.config.legacy.vacuum_kernel);
    std::vector<int> prefix,clicks;
    visit(initial,clicks,prefix,0,1.0L);
    long double total=result.failure_mass;
    for (double probability : result.distribution) total+=probability;
    if (!std::isfinite(total) || std::abs(total-1.0L)>2e-11L)
        throw std::runtime_error("full law plus failure mass failed normalization; no correction applied");
    represented_mass(result.failure_mass);
    return result;
}

void diagnostics(BFDiagnostics const& end, BFDiagnostics const& start) {
    static const char* names[]={"queries","query_successes","numeric_failures","state_rebuilds",
        "state_reuses","state_appends","zero_updates","zero_factorizations","zero_solve_columns",
        "zero_schur_blocks","subsets","subset_factorizations","subset_dimension_sum","subset_cubic_work",
        "shared_solve_columns","shared_solve_dot_terms","sparse_schur_blocks","sparse_schur_dot_terms",
        "local_determinants","local_dimension_sum","local_cubic_work","event_accumulations",
        "tracked_add_sub","tracked_multiply","tracked_divide","tracked_sqrt","output_conversions",
        "analytic_zero_events","maximum_retained","maximum_omitted","maximum_zeros","state_invalidations"};
    static_assert(sizeof(names)/sizeof(names[0])==BF_COUNTER_COUNT);
    std::cout << '{';
    for (int precision=0;precision<2;++precision) {
        if (precision) std::cout << ',';
        std::cout << quote(precision ? "quad" : "longdouble") << ":{";
        for (int j=0;j<BF_COUNTER_COUNT;++j) {
            if (j) std::cout << ',';
            const bool maximum=j==BF_MAXIMUM_RETAINED || j==BF_MAXIMUM_OMITTED || j==BF_MAXIMUM_ZEROS;
            std::cout << quote(names[j]) << ':' << (maximum ? end.counters[precision][j]
                    : end.counters[precision][j]-start.counters[precision][j]);
        }
        std::cout << ",\"query_seconds\":" << end.seconds[precision][BF_QUERY_SECONDS]-start.seconds[precision][BF_QUERY_SECONDS]
            << ",\"state_seconds\":" << end.seconds[precision][BF_STATE_SECONDS]-start.seconds[precision][BF_STATE_SECONDS]
            << ",\"event_seconds\":" << end.seconds[precision][BF_EVENT_SECONDS]-start.seconds[precision][BF_EVENT_SECONDS]
            << ",\"output_seconds\":" << end.seconds[precision][BF_OUTPUT_SECONDS]-start.seconds[precision][BF_OUTPUT_SECONDS]
            << ",\"state_payload_bytes\":" << end.state_payload_bytes[precision]
            << ",\"peak_state_payload_bytes\":" << end.peak_state_payload_bytes[precision]
            << ",\"peak_workspace_payload_bytes\":" << end.peak_workspace_payload_bytes[precision] << '}';
    }
    std::cout << ",\"probability_calls\":" << end.probability_calls-start.probability_calls
        << ",\"auto_refinements\":" << end.auto_refinements-start.auto_refinements
        << ",\"auto_refinement_successes\":" << end.auto_refinement_successes-start.auto_refinement_successes
        << ",\"auto_refinement_failures\":" << end.auto_refinement_failures-start.auto_refinement_failures
        << ",\"reset_requests\":" << end.reset_requests-start.reset_requests
        << ",\"original_input_bytes\":" << end.original_input_bytes
        << ",\"peak_kernel_payload_bytes\":" << end.peak_kernel_payload_bytes
        << ",\"kernel_setup_seconds\":" << end.setup_seconds
        << ",\"peak_fields_scope\":\"cumulative process peak, not phase increment\"}";
}

int run(int argc,char** argv) {
    std::cout << std::setprecision(17);
    const auto config=configuration(argc,argv);
    auto const& option=config.legacy;
    Context context;
    std::unique_ptr<Engine> engine;
    std::vector<std::uint64_t> packed;
    std::vector<double> logs;
    std::vector<Step> trace;
    std::vector<int> active_prefix;
    std::size_t completed=0,uniforms_consumed=0;
    const std::size_t words=(option.modes+63)/64;
    double input_seconds=0,setup_seconds=0,sampling_seconds=0,aggregation_seconds=0,enumeration_seconds=0;
    BFDiagnostics after_setup{},after_sampling{},after_enumeration{};
    Enumeration law;
    bool enumeration_complete=false,numerical_failure=false;
    std::string error,failed_phase;
    auto phase_started=Clock::now();
    try {
        context.phase="input";
        const int dimension=2*option.modes;
        const auto values=read_csv(option.input,dimension,dimension);
        Matrix covariance(dimension,dimension);
        for (int row=0;row<dimension;++row) for (int column=0;column<dimension;++column)
            covariance(row,column)=values[row*dimension+column];
        std::vector<double> uniforms;
        if (!option.uniforms.empty()) {
            uniforms=read_csv(option.uniforms,option.samples,option.modes);
            for (double value : uniforms) if (!(value>=0 && value<1))
                throw std::invalid_argument("injected uniforms must lie in [0,1)");
        }
        input_seconds=elapsed(phase_started);
        context.phase="setup"; phase_started=Clock::now();
        engine=std::make_unique<Engine>(config,std::move(covariance),context);
        if (bf_diagnostics(engine->kernel.get(),&after_setup)!=BF_OK)
            throw std::runtime_error("cannot read kernel setup diagnostics");
        setup_seconds=elapsed(phase_started);
        context.phase="sampling"; phase_started=Clock::now();
        packed.assign(option.samples*words,0);
        if (option.record_log_probabilities) logs.reserve(option.samples);
        if (option.trace) trace.reserve(option.samples*option.modes);
        std::mt19937_64 random(option.seed);
        for (std::size_t sample=0;sample<option.samples;++sample) {
            context.sample_index=sample;
            bf_reset(engine->kernel.get());
            State state(engine->selector.base,option.vacuum_kernel);
            std::vector<int> clicks;
            active_prefix.clear();
            long double log_probability=0;
            for (int position=0;position<option.modes;++position) {
                context.phase="sampling";
                const double probability=engine->conditional(state,clicks,active_prefix,
                    engine->selector.sampling_counts,engine->selector.sampling_profile,engine->sampling_tree);
                if (option.trace) trace.push_back(engine->last_step);
                const double uniform=uniforms.empty()
                    ? static_cast<double>(random()>>11)*0x1.0p-53
                    : uniforms[sample*option.modes+position];
                ++uniforms_consumed;
                const int one=uniform<probability;
                if (option.record_log_probabilities) {
                    const long double term=one ? std::log(static_cast<long double>(probability))
                        : std::log1p(-static_cast<long double>(probability));
                    if (!std::isfinite(term) || term>0)
                        throw NumericalProbabilityError("sampled branch has invalid actual log probability");
                    log_probability+=term;
                }
                if (one) packed[sample*words+position/64] |= std::uint64_t{1}<<(position%64);
                active_prefix.push_back(one);
                if (position+1<option.modes) {
                    if (one) clicks.push_back(position);
                    else {
                        context.phase="sampling_zero_state_update";
                        state.project_zero(position,engine->selector.sampling_counts,engine->selector.sampling_profile);
                    }
                }
            }
            if (option.record_log_probabilities) {
                const double value=static_cast<double>(log_probability);
                if (!std::isfinite(value) || value>0)
                    throw NumericalProbabilityError("completed sample log probability is not representable");
                logs.push_back(value);
            }
            ++completed;
            active_prefix.clear();
        }
        sampling_seconds=elapsed(phase_started);
        if (bf_diagnostics(engine->kernel.get(),&after_sampling)!=BF_OK)
            throw std::runtime_error("cannot read sampling diagnostics");
        after_enumeration=after_sampling;
        if (option.enumerate) {
            context.phase="enumeration"; context.sample_index=0; phase_started=Clock::now();
            law=enumerate(*engine);
            enumeration_seconds=elapsed(phase_started);
            enumeration_complete=true;
            if (bf_diagnostics(engine->kernel.get(),&after_enumeration)!=BF_OK)
                throw std::runtime_error("cannot read enumeration diagnostics");
        }
    } catch (std::exception const& exception) {
        error=exception.what(); failed_phase=context.phase;
        numerical_failure=dynamic_cast<NumericalProbabilityError const*>(&exception)!=nullptr;
        if (context.phase=="input") input_seconds=elapsed(phase_started);
        else if (context.phase=="setup") setup_seconds=elapsed(phase_started);
        else if (context.phase.rfind("sampling",0)==0) sampling_seconds=elapsed(phase_started);
        else if (context.phase=="enumeration") enumeration_seconds=elapsed(phase_started);
        if (engine) {
            BFDiagnostics current{};
            if (bf_diagnostics(engine->kernel.get(),&current)==BF_OK) {
                if (context.phase=="enumeration") after_enumeration=current;
                else { after_sampling=current; after_enumeration=current; }
            }
        }
    }
    phase_started=Clock::now();
    std::vector<std::uint64_t> histogram,click_histogram(option.modes+1,0),mode_counts(option.modes,0);
    if (option.modes<=enumeration_cap) histogram.assign(std::size_t{1}<<option.modes,0);
    for (std::size_t sample=0;sample<completed;++sample) {
        int clicks=0;
        for (int mode=0;mode<option.modes;++mode) {
            const bool bit=(packed[sample*words+mode/64]>>(mode%64))&1;
            clicks+=bit; mode_counts[mode]+=bit;
        }
        ++click_histogram[clicks];
        if (!histogram.empty()) ++histogram[packed[sample*words]];
    }
    aggregation_seconds=elapsed(phase_started);
    std::cout << "{\"status\":" << quote(error.empty()?"ok":"error")
        << ",\"error\":" << (error.empty()?"null":quote(error))
        << ",\"numerical_failure\":" << (numerical_failure?"true":"false")
        << ",\"failed_phase\":" << (error.empty()?"null":quote(failed_phase))
        << ",\"implementation\":\"native_batched_forest_v1\",\"abi_version\":" << BF_ABI_VERSION
        << ",\"modes\":" << option.modes << ",\"k\":" << option.k
        << ",\"selector\":" << quote(option.selector) << ",\"forest\":" << quote(config.forest)
        << ",\"precision_policy\":" << quote(config.precision)
        << ",\"vacuum_kernel\":" << quote(option.vacuum_kernel==VacuumKernel::product?"product":"log")
        << ",\"maximum_relative_roundoff\":" << max_relative_roundoff
        << ",\"seed\":" << option.seed << ",\"order\":"; print_vector(option.order);
    std::cout << ",\"requested_samples\":" << option.samples << ",\"completed_samples\":" << completed
        << ",\"sample_count\":" << completed << ",\"uniforms_consumed\":" << uniforms_consumed
        << ",\"uniform_source\":" << quote(option.uniforms.empty()?"mt19937_64_top53":"injected")
        << ",\"prefix_cache_entries\":0,\"selector_step_cache\":" << (option.step_cache?"true":"false")
        << ",\"partial_sample_prefix\":"; print_vector(active_prefix);
    std::cout << ",\"sample_law\":\"actual rounded conditional chain under ideal continuous uniforms; failures explicit\""
        << ",\"timings\":{\"input_seconds\":" << input_seconds << ",\"setup_seconds\":" << setup_seconds
        << ",\"sampling_seconds\":" << sampling_seconds << ",\"cold_batch_seconds\":" << setup_seconds+sampling_seconds
        << ",\"aggregation_seconds\":" << aggregation_seconds << ",\"enumeration_seconds\":" << enumeration_seconds
        << ",\"cold_enumeration_seconds\":" << setup_seconds+enumeration_seconds << '}'
        << ",\"packed_sample_bytes_allocated\":" << packed.size()*sizeof(std::uint64_t)
        << ",\"log_probability_enabled\":" << (option.record_log_probabilities?"true":"false")
        << ",\"sample_log_probabilities\":";
    if (option.record_log_probabilities) print_vector(logs); else std::cout << "null";
    std::cout << ",\"raw_samples\":";
    if (option.raw_samples) {
        std::cout << '[';
        for (std::size_t sample=0;sample<completed;++sample) {
            if (sample) std::cout << ',';
            std::cout << '[';
            for (int mode=0;mode<option.modes;++mode) {
                if (mode) std::cout << ',';
                std::cout << ((packed[sample*words+mode/64]>>(mode%64))&1);
            }
            std::cout << ']';
        }
        std::cout << ']';
    } else std::cout << "null";
    // A stopped batch always exposes a lossless completed-sample checkpoint,
    // even if expanded raw JSON was not requested. Integers are decimal uint64.
    std::cout << ",\"completed_packed_words\":";
    if (!error.empty()) {
        std::vector<std::uint64_t> checkpoint(packed.begin(),packed.begin()+completed*words);
        print_vector(checkpoint);
    } else std::cout << "null";
    std::cout << ",\"words_per_sample\":" << words << ",\"histogram_counts\":";
    if (!histogram.empty()) print_vector(histogram); else std::cout << "null";
    std::cout << ",\"click_count_histogram\":"; print_vector(click_histogram);
    std::cout << ",\"mode_click_counts\":"; print_vector(mode_counts);
    std::cout << ",\"enumeration_complete\":" << (enumeration_complete?"true":"false")
        << ",\"distribution\":";
    if (enumeration_complete) print_vector(law.distribution); else std::cout << "null";
    std::cout << ",\"failure_mass\":";
    if (enumeration_complete) finite_ld(law.failure_mass); else std::cout << "null";
    std::cout << ",\"failures\":"; print_failures(law.failures);
    std::cout << ",\"conditional_trace\":";
    if (option.trace) print_steps(trace); else std::cout << "null";
    std::cout << ",\"enumeration_trace\":";
    if (option.trace && enumeration_complete) print_steps(law.trace); else std::cout << "null";
    std::cout << ",\"failure_context\":";
    if (!error.empty()) print_context(context); else std::cout << "null";
    std::cout << ",\"failed_conditional\":";
    if (!error.empty() && engine && !engine->last_step.accepted) print_step(engine->last_step); else std::cout << "null";
    if (engine) {
        std::cout << ",\"minimum_uncertainty_eigenvalue\":" << engine->selector.minimum_uncertainty_eigenvalue
            << ",\"covariance_symmetry_correction\":" << engine->selector.covariance_symmetry_correction
            << ",\"setup_diagnostics\":"; engine->selector.setup_counts.print();
        std::cout << ",\"selector_sampling_diagnostics\":"; engine->selector.sampling_counts.print();
        std::cout << ",\"selector_enumeration_diagnostics\":"; engine->selector.enumeration_counts.print();
        std::cout << ",\"sampling_tree_diagnostics\":"; engine->sampling_tree.print();
        std::cout << ",\"enumeration_tree_diagnostics\":"; engine->enumeration_tree.print();
        std::cout << ",\"setup_kernel_diagnostics\":"; diagnostics(after_setup,BFDiagnostics{});
        std::cout << ",\"sampling_kernel_diagnostics\":"; diagnostics(after_sampling,after_setup);
        std::cout << ",\"enumeration_kernel_diagnostics\":"; diagnostics(after_enumeration,after_sampling);
        std::cout << ",\"sampling_profile\":"; engine->selector.sampling_profile.print();
        std::cout << ",\"enumeration_profile\":"; engine->selector.enumeration_profile.print();
        std::cout << ",\"static_pairwise_mi\":";
        if (option.trace && config.forest=="static_mi_tree") {
            std::cout << '[';
            for (int row=0;row<option.modes;++row) {
                if (row) std::cout << ',';
                std::cout << '[';
                for (int column=0;column<option.modes;++column) {
                    if (column) std::cout << ',';
                    std::cout << engine->selector.pairwise_mi(row,column);
                }
                std::cout << ']';
            }
            std::cout << ']';
        } else std::cout << "null";
    }
    std::cout << "}\n";
    return error.empty()?0:(numerical_failure?2:1);
}
} // namespace forest_cli

int main(int argc,char** argv) {
    try { return forest_cli::run(argc,argv); }
    catch (std::exception const& error) {
        std::cout << "{\"status\":\"error\",\"error\":" << quote(error.what())
            << ",\"completed_samples\":0,\"sample_count\":0,\"numerical_failure\":false}\n";
        return 1;
    }
}
