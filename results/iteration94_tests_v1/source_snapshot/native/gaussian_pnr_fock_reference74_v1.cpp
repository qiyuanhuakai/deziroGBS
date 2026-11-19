// Independent pure-Gaussian Fock-amplitude recurrence (no P moments).
#include <complex>
#include <vector>
#include <unordered_map>
#include <functional>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#define API extern "C" __declspec(dllexport)
API std::int64_t fock74_reference(int modes,int cutoff,const double* matrix,double vacuum,std::uint64_t capacity,std::uint64_t* codes,double* probabilities,char* message,int cap){try{
    if(modes<1||modes>8||cutoff<0||cutoff>128||!matrix||!codes||!probabilities||!(vacuum>0&&vacuum<=1))throw std::runtime_error("Invalid Fock reference arguments");
    using Z=std::complex<long double>;std::vector<Z>a(modes*modes);for(int i=0;i<modes*modes;++i)a[i]=Z(matrix[2*i],matrix[2*i+1]);
    std::vector<std::uint64_t>powers(modes,1);for(int i=1;i<modes;++i){if(powers[i-1]>UINT64_MAX/std::uint64_t(cutoff+1))throw std::runtime_error("Reference index overflow");powers[i]=powers[i-1]*(cutoff+1);}
    std::unordered_map<std::uint64_t,Z>memo;memo[0]=1;std::vector<int>count(modes,0);
    std::function<Z(std::uint64_t,int)> amplitude=[&](std::uint64_t code,int total)->Z{
        if(total%2)return 0;auto found=memo.find(code);if(found!=memo.end())return found->second;
        int i=0;while(count[i]==0)++i;int ni=count[i];--count[i];Z sum=0;
        for(int j=0;j<modes;++j)if(count[j]){long double factor=std::sqrt((long double)count[j]);--count[j];sum+=a[i*modes+j]*factor*amplitude(code-powers[i]-powers[j],total-2);++count[j];}
        ++count[i];return memo[code]=sum/std::sqrt((long double)ni);
    };
    std::uint64_t rows=0;std::function<void(int,int,int,std::uint64_t)> enumerate=[&](int at,int remaining,int total,std::uint64_t code){
        if(at==modes-1){count[at]=remaining;code+=remaining*powers[at];if(rows>=capacity)throw std::runtime_error("Fock output capacity exceeded");codes[rows]=code;probabilities[rows]=double((long double)vacuum*std::norm(amplitude(code,total)));++rows;return;}
        for(int value=0;value<=remaining;++value){count[at]=value;enumerate(at+1,remaining-value,total,code+value*powers[at]);}
    };
    for(int total=0;total<=cutoff;++total)enumerate(0,total,total,0);return std::int64_t(rows);
}catch(const std::exception& e){if(message&&cap>0){auto n=std::min(std::strlen(e.what()),std::size_t(cap-1));std::memcpy(message,e.what(),n);message[n]=0;}return -1;}}
