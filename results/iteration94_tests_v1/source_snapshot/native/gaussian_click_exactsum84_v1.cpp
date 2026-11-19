// Exact fixed-width integer dot accumulation of stored Weight83 components
// and x87 long doubles. Only final conversion/division uses binary128 pairs.
// Exponent spans outside the accumulator domain fall back to frozen Weight83.
// This preserves the ideal finite-budget law, not an exact GBS law.
#include "gaussian_click_weights83_v1.cpp"
#include <cstring>
#include <climits>
#include <cfloat>
static_assert(sizeof(Q82)==16 && sizeof(long double)==16 && LDBL_MANT_DIG==64);
using U128_84=unsigned __int128;
struct QuadBits84{std::uint64_t low=0,high=0;int exponent=0;bool negative=false;bool nonzero()const{return low||high;}};
struct LongBits84{std::uint64_t mantissa=0;int exponent=0;bool negative=false;};
static QuadBits84 decode84(Q82 q){
 std::array<std::uint64_t,2> bits;std::memcpy(bits.data(),&q,16);unsigned e=(bits[1]>>48)&32767;need(e!=32767,"ExactSum84 nonfinite weight");
 return {bits[0],(bits[1]&0x0000ffffffffffffULL)|(e?0x0001000000000000ULL:0),int(e?e:1)-16383-112,bool(bits[1]>>63)};
}
static LongBits84 decode84(long double x){
 std::uint64_t mantissa;std::uint16_t signexp;std::memcpy(&mantissa,&x,8);std::memcpy(&signexp,reinterpret_cast<const char*>(&x)+8,2);unsigned e=signexp&32767;need(e!=32767,"ExactSum84 nonfinite state");return {mantissa,int(e?e:1)-16383-63,bool(signexp>>15)};
}
using Limbs84=std::array<std::uint64_t,8>;
static bool add_word84(Limbs84& a,unsigned at,std::uint64_t word){
 while(word){if(at>=a.size())return false;auto before=a[at];a[at]+=word;word=a[at]<before;++at;}return true;
}
static bool add_product84(Limbs84& a,const QuadBits84& q,const LongBits84& x,int base){
 U128_84 first=U128_84(q.low)*x.mantissa,second=U128_84(q.high)*x.mantissa+(first>>64);
 std::array<std::uint64_t,3> product{std::uint64_t(first),std::uint64_t(second),std::uint64_t(second>>64)};
 unsigned shift=unsigned(q.exponent+x.exponent-base),at=shift/64,bits=shift%64;
 for(unsigned j=0;j<3;++j){if(!add_word84(a,at+j,product[j]<<bits))return false;if(bits&&!add_word84(a,at+j+1,product[j]>>(64-bits)))return false;}return true;
}
static int compare84(const Limbs84& a,const Limbs84& b){for(int j=7;j>=0;--j)if(a[j]!=b[j])return a[j]>b[j]?1:-1;return 0;}
static Limbs84 subtract84(const Limbs84& a,const Limbs84& b){
 Limbs84 result{};std::uint64_t borrow=0;for(unsigned j=0;j<8;++j){U128_84 v=U128_84(b[j])+borrow;result[j]=a[j]-std::uint64_t(v);borrow=U128_84(a[j])<v;}need(!borrow,"ExactSum84 subtraction order");return result;
}
struct DotWeights84{std::vector<std::array<QuadBits84,2>> bits;std::vector<Weight83> weights;};
static DotWeights84 prepare84(const std::vector<WeightBranch83>& branches){DotWeights84 out;for(auto& b:branches){out.bits.push_back({decode84(b.weight.hi),decode84(b.weight.lo)});out.weights.push_back(b.weight);}return out;}
static Weight83 dot84(const DotWeights84& weights,const std::vector<long double>& values,bool& fallback){
 const auto count=weights.weights.size();need(count==values.size()&&count<=256,"ExactSum84 dot dimension");std::array<LongBits84,256> xb;int base=INT_MAX,largest=INT_MIN;
 for(std::size_t k=0;k<count;++k){xb[k]=decode84(values[k]);if(!xb[k].mantissa)continue;for(auto& q:weights.bits[k])if(q.nonzero()){int e=q.exponent+xb[k].exponent;base=std::min(base,e);largest=std::max(largest,e);}}
 auto slow=[&](){fallback=true;Weight83 result;for(std::size_t k=0;k<count;++k)result=result+weights.weights[k]*Weight83(Q82(values[k]));return result;};
 fallback=false;if(base==INT_MAX)return Weight83(0);if(largest-base>310)return slow();
 Limbs84 positive{},negative{};
 for(std::size_t k=0;k<count;++k)if(xb[k].mantissa)for(auto& q:weights.bits[k])if(q.nonzero())if(!add_product84(q.negative!=xb[k].negative?negative:positive,q,xb[k],base))return slow();
 int sign=compare84(positive,negative);if(!sign)return Weight83(0);auto magnitude=sign>0?subtract84(positive,negative):subtract84(negative,positive);Weight83 result;
 for(int j=7;j>=0;--j)if(magnitude[j]){Q82 part=scalbnq(Q82(magnitude[j]),base+64*j);if(!finite82(part)||part==0)return slow();result=result+Weight83(part);}
 return sign>0?result:-result;
}
struct SumTrace84{double exact=0,fallback=0;};
static void exact_update84(std::vector<WeightBranch83>& branches,const WeightStep83& step,bool click,int budget,StableTrace82& trace,SumTrace84& sums){
 Weight83 probability=click?step.p1:step.p0;need(probability>Weight83(0),"Weight83 forced zero-probability branch");std::vector<WeightBranch83> next;next.reserve(branches.size()*(click?2:1));
 for(std::size_t k=0;k<branches.size();++k){auto& v=branches[k];auto& p=step.parts[k];if(click)next.push_back({v.x,v.weight/probability});if(p.p0>Weight83(0)&&(v.weight.hi!=0||v.weight.lo!=0)){WeightBranch83 b=v;Mat82 u=p.xa*(p.i00*p.xa+p.i01*p.xb).transpose()+p.xb*(p.i01*p.xa+p.i11*p.xb).transpose();b.x-=u;b.x=((b.x+b.x.transpose())/2).eval();b.weight=v.weight*p.p0/probability*Weight83(click?-1:1);need(b.x.allFinite(),"Weight83 covariance overflow");next.push_back(std::move(b));}}
 need(!next.empty(),"Weight83 empty branch list");Weight83 mass;for(auto& b:next)mass=mass+b.weight;need(mass>Weight83(0)&&finite83(mass),"Weight83 normalization failed");for(auto& b:next)b.weight=b.weight/mass;Weight83 after,condition;for(auto& b:next){after=after+b.weight;condition=condition+abs83(b.weight);}Weight83 residual=abs83(after-Weight83(1));need(finite83(residual)&&residual<Weight83(Q82(1e-18L)),"Weight83 normalization lost");trace.residual=std::max(trace.residual,ld83(residual));trace.condition=std::max(trace.condition,ld83(condition));trace.peak=std::max(trace.peak,double(next.size()));
 if(next.size()>std::size_t(budget)){int d=next[0].x.rows();WeightBranch83 merged{Mat82::Zero(d,d),Weight83(1)};auto weights=prepare84(next);std::vector<long double> values(next.size());for(int i=0;i<d;++i)for(int j=0;j<=i;++j){for(std::size_t k=0;k<next.size();++k)values[k]=next[k].x(i,j);bool fallback;auto value=dot84(weights,values,fallback);if(fallback)++sums.fallback;else ++sums.exact;merged.x(i,j)=merged.x(j,i)=ld83(value/after);}need(merged.x.allFinite(),"ExactSum84 moment compression overflow");next.clear();next.push_back(std::move(merged));++trace.merges;}branches=std::move(next);
}
// Decimal arrays separated by spaces. Independent tests use exactly
// representable dyadic inputs to avoid source-decimal conversion ambiguity.
API int sum84_probe(int count,const char* input,char* output,int size,char* message,int cap){try{
 PrecisionGuard precision;need(count>=0&&count<=256&&input&&output&&size>0,"ExactSum84 probe arguments");DotWeights84 weights;std::vector<long double> values;const char* cursor=input;
 for(int j=0;j<count;++j){Q82 q[2];for(int h=0;h<2;++h){char* end;auto value=strtoflt128(cursor,&end);need(end!=cursor&&finite82(value),"ExactSum84 invalid quad input");q[h]=value;cursor=end;}char* end;long double x=strtold(cursor,&end);need(end!=cursor&&std::isfinite(x),"ExactSum84 invalid long input");cursor=end;weights.weights.emplace_back(q[0],q[1]);weights.bits.push_back({decode84(q[0]),decode84(q[1])});values.push_back(x);}
 while(*cursor==' '||*cursor=='\n'||*cursor=='\r'||*cursor=='\t')++cursor;need(!*cursor,"ExactSum84 trailing input");bool fallback;auto result=dot84(weights,values,fallback);int used=quadmath_snprintf(output,size,"%.85Qg",result.hi);need(used>0&&used+3<size,"ExactSum84 short output");output[used++]=' ';int length=quadmath_snprintf(output+used,size-used,"%.85Qg",result.lo);need(length>0&&used+length+3<size,"ExactSum84 short output");used+=length;output[used++]=' ';output[used++]=fallback?'1':'0';output[used]=0;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc84e_prefix(void* ptr,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&bits&&probability&&stats&&budget>=1&&budget<=128,"Invalid ExactSum84 prefix arguments");auto& c=*static_cast<ClickModel70*>(ptr);need(length>=0&&length<=std::min(16,c.model->m),"ExactSum84 prefix audit limit");std::fill(stats,stats+8,0);std::vector<WeightBranch83> branches{{c.model->initial.cast<long double>(),Weight83(1)}};StableTrace82 trace;SumTrace84 sums;Weight83 joint(1);
 for(int i=0;i<length;++i){need(bits[i]<=1,"Invalid ExactSum84 bit");auto p=weight_step83(c,branches,i,trace);joint=joint*(bits[i]?p.p1:p.p0);if(!(joint>Weight83(0)))break;if(i+1<length)exact_update84(branches,p,bits[i],budget,trace,sums);stats[5]=i+1;}
 *probability=double83(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=double(trace.condition);stats[4]=double(trace.residual);stats[6]=sums.exact;stats[7]=sums.fallback;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc84e_sample(void* ptr,int count,std::uint64_t seed,int threads,int budget,const double* draws,std::uint8_t* bits,double* probabilities,int* errors,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&count>=0&&count<=4096&&threads>=1&&threads<=64&&budget>=1&&budget<=128&&bits&&probabilities&&errors&&stats,"Invalid ExactSum84 sample arguments");auto& c=*static_cast<ClickModel70*>(ptr);int m=c.model->m;std::fill(bits,bits+std::size_t(count)*m,0);std::fill(probabilities,probabilities+std::size_t(count)*m,std::numeric_limits<double>::quiet_NaN());std::fill(errors,errors+2*count,0);std::fill(stats,stats+8*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){PrecisionGuard precision;for(;;){int row=next.fetch_add(1);if(row>=count)break;int mode=0,stage=1;StableTrace82 trace;SumTrace84 sums;try{std::mt19937_64 rng(mix(seed+2ULL*row+1));std::vector<WeightBranch83> branches{{c.model->initial.cast<long double>(),Weight83(1)}};for(;mode<m;++mode){stage=1;double u=draws?draws[std::size_t(row)*m+mode]:uniform(rng);need(std::isfinite(u)&&u>=0&&u<1,"Invalid ExactSum84 draw");auto p=weight_step83(c,branches,mode,trace);bool bit=Weight83(Q82(u))<p.p1;probabilities[std::size_t(row)*m+mode]=double83(p.p1);bits[std::size_t(row)*m+mode]=bit;if(mode+1<m){stage=2;exact_update84(branches,p,bit,budget,trace,sums);}}}catch(const std::exception& e){errors[2*row]=stage;errors[2*row+1]=mode;messages[row]=e.what();}stats[8*row]=trace.evaluations;stats[8*row+1]=trace.merges;stats[8*row+2]=trace.peak;stats[8*row+3]=double(trace.condition);stats[8*row+4]=double(trace.residual);stats[8*row+5]=mode;stats[8*row+6]=sums.exact;stats[8*row+7]=sums.fallback;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" mode="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
