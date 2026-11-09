#include "fgbs/layout.hpp"

namespace fgbs {

usize get_parallel_fork_depth(int thread_count) {
    return log2(static_cast<usize>(thread_count));
}

}
