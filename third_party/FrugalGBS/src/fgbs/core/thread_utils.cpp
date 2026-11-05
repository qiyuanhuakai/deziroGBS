#include "fgbs/core/thread_utils.hpp"

#include <thread>
#include <vector>

namespace fgbs {

void join_threads(std::vector<std::thread>& threads) {
    for (auto& thread : threads) {
        if (thread.joinable()) {
            thread.join();
        }
    }
}

}
