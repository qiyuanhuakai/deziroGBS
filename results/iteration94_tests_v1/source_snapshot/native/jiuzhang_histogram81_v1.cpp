// MSB-first detector bits, as specified by the experiment's public README.
#include <cstdint>
#include <cstring>
#include <vector>
#include <omp.h>
extern "C" __declspec(dllexport) int jz81_hist(const unsigned char* data,std::uint64_t records,int groups,const int* ports,std::uint64_t pilot,int threads,std::uint64_t* output,std::uint64_t* totals){
 if(!data||!ports||!output||!totals||records==0||groups<1||groups>16||pilot>=records||threads<1||threads>64)return 1;
 for(int g=0;g<groups;++g)for(int j=0;j<8;++j){if(ports[g*8+j]<0||ports[g*8+j]>=1152)return 2;for(int k=0;k<j;++k)if(ports[g*8+j]==ports[g*8+k])return 3;}
 std::memset(output,0,sizeof(std::uint64_t)*groups*2*256);std::memset(totals,0,sizeof(std::uint64_t)*1153);
 #pragma omp parallel num_threads(threads)
 {
  std::vector<std::uint64_t> local(groups*2*256),count(1153);
  #pragma omp for schedule(static)
  for(std::uint64_t r=0;r<records;++r){const auto* row=data+144*r;int n=0;for(int b=0;b<144;++b)n+=__builtin_popcount(unsigned(row[b]));++count[n];
   for(int g=0;g<groups;++g){int pattern=0;for(int j=0;j<8;++j){int p=ports[g*8+j];pattern|=((row[p/8]>>(7-p%8))&1)<<j;}++local[(g*2+int(r>=pilot))*256+pattern];}
  }
  #pragma omp critical
  {for(std::size_t i=0;i<local.size();++i)output[i]+=local[i];for(int i=0;i<1153;++i)totals[i]+=count[i];}
 }
 return 0;
}
