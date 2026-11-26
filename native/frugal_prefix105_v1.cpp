#include "fgbs/data.hpp"
#include "fgbs/sampler.hpp"
#include "frugal_extended_hooks.hpp"
#include <omp.h>
#include <stdexcept>
#include <cstdio>

template<class T> static T checked(fgbs::Result<T>&& result) {
    if (!result.ok()) throw std::runtime_error(result.error());
    return std::move(result.value());
}

extern "C" __declspec(dllexport) int frugal105_tree(const double* matrix, double* thresholds, char* error, int capacity) {
    constexpr size_t width = 1152, depth = 16;
    using namespace fgbs;
    using namespace fgbs::cpu;
    const auto previous = omp_get_max_threads();
    int status = 0;
    try {
        omp_set_num_threads(1);
        Matrix<f64> cov(2*width,2*width);
        for (size_t i=0; i<2*width; ++i)
            for (size_t j=0; j<2*width; ++j) cov(i,j)=matrix[i*2*width+j];
        auto layout=checked(generate_bitstring_layout<width>(2));
        auto cumulants=checked(generate_cumulant_layout<width>(2));
        auto target=checked(generate_sampling_target<width>(layout,cumulants,cov));
        // Upstream addresses temporary arrays by the current end port. The
        // instrumented recurrence returns before reading or writing port 16.
        std::vector<ProbSimd> partial(partial_marginal_arr_size<depth>()), elided(elided_marginal_arr_size<depth>());
        std::vector<ProbSimd> pe(partial_elided_marginal_arr_size<depth>()), de(double_elided_marginal_arr_size<depth>());
        for (size_t length=0; length<depth; ++length) {
            const size_t count=size_t{1}<<length;
            for (size_t bits=0; bits<count; ++bits) {
                std::string prefix(length,'0');
                for (size_t i=0; i<length; ++i) if ((bits>>i)&1) prefix[i]='1';
                thresholds[count-1+bits]=trace_prefix<width>(layout,target,prefix,partial.data(),elided.data(),pe.data(),de.data()).threshold;
            }
        }
    } catch(const std::exception& ex) {
        std::snprintf(error,capacity,"%s",ex.what());
        status=1;
    }
    omp_set_num_threads(previous);
    return status;
}
