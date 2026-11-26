#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

using Word=std::uint64_t;
using Law=std::map<Word,double>;
struct Occupation { int port,count; };
static void expand(const std::vector<Occupation>& occupied,std::size_t i,Word word,double prob,
                   const double kernel[16][9][9],Law& law) {
    if(i==occupied.size()){law[word]+=prob;return;}
    auto [port,count]=occupied[i];
    for(int j=0;j<=count;++j)
        expand(occupied,i+1,word|(Word(j)<<(4*port)),prob*kernel[port][count][j],kernel,law);
}
extern "C" __declspec(dllexport) int thin99(int n,const unsigned char* states,const double* prob,
        const double* eta,int capacity,Word* keys,double* probabilities){
    if(n<1||capacity<1||!states||!prob||!eta||!keys||!probabilities)return -1;
    double k[16][9][9]={};
    for(int i=0;i<16;++i){
        if(eta[i]<0||eta[i]>1)return -1;
        k[i][0][0]=1;
        for(int s=1;s<=8;++s)for(int j=0;j<=s;++j)
            k[i][s][j]=k[i][s-1][j]*(1-eta[i])+(j?k[i][s-1][j-1]*eta[i]:0.);
    }
    Law law;std::vector<Occupation> occupied;
    for(int r=0;r<n;++r){
        if(prob[r]==0)continue;
        int total=0;occupied.clear();
        for(int i=0;i<16;++i)if(int s=states[16*r+i]){
            total+=s;occupied.push_back({i,s});
        }
        if(total>8||!std::isfinite(prob[r])||prob[r]<0)return -2;
        expand(occupied,0,0,prob[r],k,law);
    }
    if(law.size()>std::size_t(capacity))return -3;
    int index=0;
    for(auto [key,value]:law){keys[index]=key;probabilities[index++]=value;}
    return index;
}
