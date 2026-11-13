// Philox4x32-10: Random123-compatible round function and key schedule.
// Upstream copyright/license: random123_notice84.txt; pinned KAT provenance
// is archived in results/iteration84_rng_reference_v1/source.json.
// Injective input address: key=experiment[64]; counter=(block[64],record[32],
// sector[16]|purpose[16]). A 128-bit block supplies two disjoint uint64 lanes.
// This prevents deterministic domain reuse, not a proof of mathematical iid.
#pragma once
#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
using RngCounter84=std::array<std::uint32_t,4>;
using RngKey84=std::array<std::uint32_t,2>;
static RngCounter84 philox84(RngCounter84 c,RngKey84 k){
 for(int round=0;round<10;++round){std::uint64_t p0=std::uint64_t(0xD2511F53U)*c[0],p1=std::uint64_t(0xCD9E8D57U)*c[2];c={std::uint32_t(p1>>32)^c[1]^k[0],std::uint32_t(p1),std::uint32_t(p0>>32)^c[3]^k[1],std::uint32_t(p0)};if(round!=9){k[0]+=0x9E3779B9U;k[1]+=0xBB67AE85U;}}return c;
}
struct DomainRng84 {
 using result_type=std::uint64_t;
 RngKey84 key;std::uint32_t record,domain;std::uint64_t position=0,used=0,cached_block=0;bool cached=false;RngCounter84 cache{};
 DomainRng84(std::uint64_t experiment,unsigned sector,unsigned purpose,std::uint64_t row,std::uint64_t start=0):key{std::uint32_t(experiment),std::uint32_t(experiment>>32)},position(start){
  if(sector>65535||purpose>65535||row>0xffffffffULL)throw std::runtime_error("Random domain field overflow");record=std::uint32_t(row);domain=(sector<<16)|purpose;
 }
 static constexpr result_type min(){return 0;}
 static constexpr result_type max(){return std::numeric_limits<result_type>::max();}
 RngCounter84 address(std::uint64_t draw)const{auto block=draw/2;return {std::uint32_t(block),std::uint32_t(block>>32),record,domain};}
 result_type operator()(){
  if(position==max())throw std::runtime_error("Random draw counter exhausted; no wrap");auto block=position/2;if(!cached||cached_block!=block){cache=philox84(address(position),key);cached_block=block;cached=true;}unsigned lane=unsigned(position&1)*2;result_type result=result_type(cache[lane])|(result_type(cache[lane+1])<<32);++position;++used;return result;
 }
};
enum RngPurpose84:unsigned{Waiting84=1,Detector84=2,Displacement84=3,Resplitting84=4,Click84=5};
static double uniform84(DomainRng84& rng){return double(rng()>>11)*0x1.0p-53;}
