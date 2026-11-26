#include "exact_ring99_v1.cpp"
#include <unordered_map>

extern "C" __declspec(dllexport) int verify99_fock(int keys_count,const unsigned* keys,
        const std::int32_t* re,const std::int32_t* im,int rows,const unsigned char* occupations,
        double p,double* probabilities){
    if(keys_count<1||rows<1||!keys||!re||!im||!occupations||!probabilities||p<0||p>=1)return 1;
    std::unordered_map<unsigned,Z> coeff;
    for(int i=0;i<keys_count;++i)coeff.emplace(keys[i],Z(re[i],im[i]));
    for(int r=0;r<rows;++r){
        auto n=occupations+16*std::size_t(r);int m[2]={},signal=0;unsigned input[2]={};
        double ratio=1;
        for(int j=0;j<8;++j){
            if(n[j]>8||n[8+j]>8)return 1;
            m[j/4]+=n[8+j];input[j/4]|=unsigned(n[8+j])<<(4*(j%4));signal+=n[j];
            ratio*=double(f.fac[n[j]])/f.fac[n[8+j]];
        }
        probabilities[r]=0;
        if(signal!=m[0]+m[1])continue;
        int total0=n[3]+n[4],total1=n[7]+n[0];
        int sides0=m[0]-n[1]-n[2],sides1=m[1]-n[5]-n[6];Z amp{};
        for(int r0=0;r0<=sides0;++r0){
            int l0=sides0-r0,l1=total0-r0,r1=sides1-l1;
            if(l1<0||r1<0||l0+r1!=total1)continue;
            unsigned k0=(input[0]<<16)|(r0<<12)|(n[2]<<8)|(n[1]<<4)|l0;
            unsigned k1=(input[1]<<16)|(r1<<12)|(n[6]<<8)|(n[5]<<4)|l1;
            auto a=coeff.find(k0),b=coeff.find(k1);
            if(a!=coeff.end()&&b!=coeff.end())
                amp+=a->second*b->second*std::int64_t(bs(r0,l1,n[3])*bs(r1,l0,n[7]));
        }
        probabilities[r]=double(std::norm(amp))*ratio*std::pow(1-p,8)*std::pow(p/4,signal)/std::ldexp(1.,total0+total1);
    }
    return 0;
}
