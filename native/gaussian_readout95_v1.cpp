// Score the measured format directly: one byte per fiber loop, high bit first.
#include <array>
#include <bit>
#include <cstdint>
#include <vector>
#include <omp.h>

extern "C" __declspec(dllexport) int readout95(const unsigned char* packed,int rows,int workers,
        const double* score,double* scores,std::uint64_t* loop_hist,std::uint32_t* total) {
    if(!packed||!score||!scores||!loop_hist||!total||rows<1||workers<1||workers>64) return 1;
    constexpr int loops=144;
    std::array<unsigned char,256> pop{};
    for(unsigned i=0;i<256;++i) pop[i]=static_cast<unsigned char>(std::popcount(i));
    std::fill(loop_hist,loop_hist+loops*9,0);
    #pragma omp parallel num_threads(workers)
    {
        std::array<std::uint64_t,loops*9> local{};
        #pragma omp for schedule(static)
        for(int i=0;i<rows;++i) {
            const auto* record=packed+std::size_t(i)*loops;
            double sum=0;
            unsigned clicks=0;
            for(int j=0;j<loops;++j) {
                int byte=record[j],n=pop[byte];
                sum+=score[j*256+byte];
                ++local[j*9+n];
                clicks+=n;
            }
            scores[i]=sum;total[i]=clicks;
        }
        #pragma omp critical
        for(int j=0;j<loops*9;++j) loop_hist[j]+=local[j];
    }
    return 0;
}
