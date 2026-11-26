#include <bit>
#include <cstdint>
#include <cstring>

extern "C" __declspec(dllexport) void ordered104(const unsigned char* records,int n,unsigned short* out){
    #pragma omp parallel for num_threads(8) schedule(static)
    for(int i=0;i<n;++i){
        const auto* p=records+std::size_t(i)*144;
        unsigned count=0;
        for(int j=0;j<18;++j){
            std::uint64_t word;
            std::memcpy(&word,p+8*j,8);
            count+=std::popcount(word);
        }
        out[i]=static_cast<unsigned short>(count);
    }
}
