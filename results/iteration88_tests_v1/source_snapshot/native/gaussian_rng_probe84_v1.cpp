#include "gaussian_rng_domain84_v1.hpp"
#include <algorithm>
#include <cstring>
#define API84 extern "C" __declspec(dllexport)
static void rng_error84(char* message,int cap,const char* error){if(message&&cap>0){auto n=std::min<std::size_t>(cap-1,std::strlen(error));std::memcpy(message,error,n);message[n]=0;}}
API84 int rng84_block(const std::uint32_t* counter,const std::uint32_t* key,std::uint32_t* output){if(!counter||!key||!output)return 1;auto result=philox84({counter[0],counter[1],counter[2],counter[3]},{key[0],key[1]});std::copy(result.begin(),result.end(),output);return 0;}
API84 int rng84_stream(std::uint64_t experiment,unsigned sector,unsigned purpose,std::uint64_t record,std::uint64_t start,int count,std::uint64_t* output,char* message,int cap){try{
 if(count<0||count>1048576||!output)throw std::runtime_error("Invalid RNG probe count");DomainRng84 rng(experiment,sector,purpose,record,start);for(int i=0;i<count;++i)output[i]=rng();return 0;
 }catch(const std::exception& e){rng_error84(message,cap,e.what());return 1;}}
API84 int rng84_address(std::uint64_t experiment,unsigned sector,unsigned purpose,std::uint64_t record,std::uint64_t draw,std::uint32_t* output,int* lane,char* message,int cap){try{
 if(!output||!lane)throw std::runtime_error("Invalid RNG address output");DomainRng84 rng(experiment,sector,purpose,record);auto c=rng.address(draw);output[0]=rng.key[0];output[1]=rng.key[1];std::copy(c.begin(),c.end(),output+2);*lane=int(draw&1);return 0;
 }catch(const std::exception& e){rng_error84(message,cap,e.what());return 1;}}
