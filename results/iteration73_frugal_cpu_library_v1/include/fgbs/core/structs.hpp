#ifndef FGBS_CORE_STRUCTS_HPP
#define FGBS_CORE_STRUCTS_HPP

#include "fgbs/core/consts.hpp"
#include "fgbs/core/slice.hpp"
#include "fgbs/core/type_utils.hpp"

#include <array>
#include <utility>
#include <vector>

namespace fgbs {

template<usize N>
struct BitstringLayout {
    const usize cutoff_order;
    const std::array<usize, N + 1> sum_index_arr;
    const usize len = sum_index_arr[N];

    std::vector<upto<N>> order_vec;
    std::vector<upto<N>> ones_vec;

    std::vector<Bitstring<N>> layout_bitstring_vec;

    BitstringLayout(usize cutoff_order) : cutoff_order{cutoff_order}, sum_index_arr{gbs_index_arrays<N>[cutoff_order]} {}

    upto<N> const* get_ones_ptr(usize layout_index) const {
        return &ones_vec[cutoff_order * layout_index];
    }

    upto<N>* get_ones_ptr_mut(usize layout_index) {
        return &ones_vec[cutoff_order * layout_index];
    }

    Slice<upto<N>> get_ones_slice(usize layout_index) const {
        return Slice<upto<N>>{get_ones_ptr(layout_index), cutoff_order};
    }

    SliceMut<upto<N>> get_ones_slice_mut(usize layout_index) {
        return SliceMut<upto<N>>{get_ones_ptr_mut(layout_index), cutoff_order};
    }
};

template<usize N>
struct CumulantLayout {
    const usize cutoff_order;

    std::vector<usize> cumulant_term_range_vec;
    std::vector<usize> cumulant_order_range_vec;
    std::vector<usize> cumulant_index_range_vec;

    std::vector<usize> cumulant_term_vec;
    std::vector<usize> cumulant_order_vec;
    std::vector<usize> cumulant_index_vec;

    CumulantLayout(usize cutoff_order) : cutoff_order{cutoff_order} {}
};

template<usize N>
struct SublistLayout {
    const usize cutoff_order;

    std::vector<usize> sublist_index_vec;
    std::vector<usize> sublist_size_vec;
    std::vector<usize> sublist_offset_vec;

    std::vector<Bitstring<N>> layout_bitstring_sublists;
    std::vector<upto<N>> ones_sublists;

    SublistLayout(usize cutoff_order) : cutoff_order{cutoff_order} {}

    template<typename T>
    Slice<T> get_sublist(std::vector<T> const& sublists, usize order) const {
        return sublists.empty() ? Slice<T>(sublists.data(), 0) : Slice<T>(sublists.data() + sublist_offset_vec[order], sublist_size_vec[order]);
    }

    template<typename T>
    SliceMut<T> get_sublist_mut(std::vector<T>& sublists, usize order) const {
        return sublists.empty() ? SliceMut<T>(sublists.data(), 0) : SliceMut<T>(sublists.data() + sublist_offset_vec[order], sublist_size_vec[order]);
    }

    upto<N> const* get_ones_sublist_ptr(usize order, usize sublist_index) const {
        return &ones_sublists[cutoff_order * (sublist_offset_vec[order] + sublist_index)];
    }

    upto<N>* get_ones_sublist_ptr_mut(usize order, usize sublist_index) {
        return &ones_sublists[cutoff_order * (sublist_offset_vec[order] + sublist_index)];
    }

    Slice<upto<N>> get_ones_sublist(usize order, usize sublist_index) const {
        return Slice<upto<N>>{get_ones_sublist_ptr(order, sublist_index), cutoff_order};
    }

    SliceMut<upto<N>> get_ones_sublist_mut(usize order, usize sublist_index) {
        return SliceMut<upto<N>>{get_ones_sublist_ptr_mut(order, sublist_index), cutoff_order};
    }
};

template<usize N>
struct SamplingTarget {
    std::vector<real> correlator_cumulant_vec;
    std::vector<real> first_correlator_cumulant_vec;
};

template<usize N>
struct StatsTarget {
    std::vector<real> moment_cumulant_sublists;
};

}

#endif
