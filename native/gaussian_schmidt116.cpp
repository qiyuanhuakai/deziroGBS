#include <algorithm>
#include <cmath>
#include <cstdint>
#include <queue>
#include <vector>

// Each occupation multiset is a sorted list of mode indices. Its unique
// parent removes its last index if repeated (or the first 0); otherwise
// decrements that index. Equal-probability configurations remain distinct.
struct Term {
    long double weight;
    int last;
    bool operator<(const Term& other) const { return weight < other.weight; }
};

extern "C" int schmidt_mass116(const double* ratio, int modes,
                               const int* ranks, int count, double* mass) {
    if (modes < 1 || count < 1 || ranks[0] < 1) return 1;
    long double vacuum = 1;
    for (int j = 0; j < modes; ++j) {
        if (!(ratio[j] >= 0 && ratio[j] < 1) || (j && ratio[j] > ratio[j-1])) return 2;
        vacuum *= 1 - static_cast<long double>(ratio[j]);
    }
    for (int j = 1; j < count; ++j) if (ranks[j] <= ranks[j-1]) return 3;
    std::priority_queue<Term> heap;
    if (ratio[0] > 0) heap.push({vacuum * ratio[0], 0});
    long double sum = vacuum, correction = 0;
    for (int rank = 1, k = 0; ; ++rank) {
        if (rank == ranks[k]) {
            mass[k++] = static_cast<double>(sum);
            if (k == count) return 0;
        }
        if (heap.empty()) {
            while (k < count) mass[k++] = static_cast<double>(sum);
            return 0;
        }
        const auto term = heap.top();
        heap.pop();
        const auto j = term.last;
        const long double add = term.weight - correction;
        const long double next = sum + add;
        correction = (next - sum) - add;
        sum = next;
        heap.push({term.weight * ratio[j], j});
        if (j + 1 < modes && ratio[j+1] > 0)
            heap.push({term.weight * (ratio[j+1] / static_cast<long double>(ratio[j])), j+1});
    }
}
