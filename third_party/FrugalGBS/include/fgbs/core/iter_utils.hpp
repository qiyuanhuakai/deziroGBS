#ifndef FGBS_CORE_ITER_UTILS_HPP
#define FGBS_CORE_ITER_UTILS_HPP

#include "fgbs/core/type_utils.hpp"

#include <algorithm>
#include <iterator>
#include <utility>
#include <vector>

namespace fgbs {

template<typename ITERATOR>
using IteratorVector = std::vector<std::iter_value_t<ITERATOR>>;

template<typename ITERATOR>
using PartitionVector = std::vector<IteratorVector<ITERATOR>>;

template<typename ITERATOR, typename FUNCTION>
auto map(ITERATOR begin, ITERATOR end, FUNCTION&& f) {
    std::vector<decltype(f(*begin))> vec(static_cast<usize>(end - begin));
    std::transform(begin, end, vec.begin(), std::forward<FUNCTION>(f));
    return vec;
}

template<typename ITERABLE, typename FUNCTION>
auto map(ITERABLE&& iterable, FUNCTION&& f) {
    return map(iterable.cbegin(), iterable.cend(), std::forward<FUNCTION>(f));
}

// FUNCTION = void(IteratorVector<ITERATOR>&&)
template<typename ITERATOR, typename FUNCTION>
void for_each_combination(ITERATOR begin, ITERATOR end, usize r, FUNCTION&& f) {
    const auto n = static_cast<usize>(end - begin);
    if (r == n) {
        IteratorVector<ITERATOR> vec;
        std::copy(begin, end, std::back_inserter(vec));
        f(std::move(vec));
    }
    else if (r < n) {
        std::vector<usize> indices(r);
        for (usize i = 0; i < r; ++i) {
            indices[i] = i;
        }

        auto invoke = [=, &f, &indices] {
            IteratorVector<ITERATOR> combo(r);
            for (usize i = 0; i < r; ++i) {
                combo[i] = begin[indices[i]];
            }
            f(std::move(combo));
        };

        invoke();

        const auto s = static_cast<isize>(r) - 1;
        const auto m = n - r;

        while (true) {
            isize i;
            bool flag = false;
            for (i = s; i >= 0; --i) {
                if (indices[i] != static_cast<usize>(i) + m) {
                    flag = true;
                    break;
                }
            }
            if (!flag) {
                return;
            }

            indices[i] += 1;
            for (usize j = static_cast<usize>(i) + 1; j < r; ++j) {
                indices[j] = indices[j - 1] + 1;
            }
            invoke();
        }
    }
}

// Based on function by alexis: https://stackoverflow.com/a/30134039
template<typename ITERATOR, typename FUNCTION>
void for_each_partition_internal(ITERATOR begin, ITERATOR end, usize limit, PartitionVector<ITERATOR>& partition, FUNCTION& f) {
    if (begin == end) {
        f(PartitionVector<ITERATOR>{partition});
        return;
    }

    const auto next_begin = begin + 1;
    for (usize i = 0; i < partition.size(); ++i) {
        if (limit > 0 && partition[i].size() + 1 > limit) {
            continue;
        }

        partition[i].push_back(*begin);
        for_each_partition_internal(next_begin, end, limit, partition, f);
        partition[i].pop_back();
    }

    partition.push_back(IteratorVector<ITERATOR>{*begin});
    for_each_partition_internal(next_begin, end, limit, partition, f);
    partition.pop_back();
}

// FUNCTION = void(PartitionVector<ITERATOR>&&)
template<typename ITERATOR, typename FUNCTION>
void for_each_partition(ITERATOR begin, ITERATOR end, usize limit, FUNCTION&& f) {
    PartitionVector<ITERATOR> partition;
    for_each_partition_internal(begin, end, limit, partition, f);
}

}

#endif
