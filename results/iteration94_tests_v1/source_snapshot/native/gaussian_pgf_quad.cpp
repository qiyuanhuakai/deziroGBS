// Complete finite PNR coefficient table for a centered Gaussian generating
// function G(z)=D(z)^(-1/2). Each variable in determinant D has degree <=2.
// Euler differentiation gives n D0 p_k = -sum_l (n-|l|/2) D_l p_(k-l).
// Oracle only: enumerated table size is combinatorial in mode count.
#include <quadmath.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#define API extern "C" __declspec(dllexport)
using Q=__float128;
static void require70(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static void message70(char* out,int cap,const std::string& s){if(out&&cap>0){auto n=std::min(s.size(),std::size_t(cap-1));std::memcpy(out,s.data(),n);out[n]=0;}}
static std::string decimal70(Q x){char text[128];quadmath_snprintf(text,sizeof(text),"%.36Qg",x);return text;}
struct Term70 {Q coefficient;int degree=0;int power[4]={};std::uint32_t offset=0;};
API int g70_pgf(int modes,int cutoff,const char* coefficient_text,std::uint64_t capacity,std::uint32_t* codes,double* probabilities,unsigned char* binary128,char* summary,int summary_cap,char* error,int error_cap){try{
    static_assert(sizeof(Q)==16&&FLT128_MANT_DIG==113,"IEEE binary128 required");
    require70(modes>=1&&modes<=4&&cutoff>=0&&cutoff<=64&&coefficient_text&&codes&&probabilities&&binary128,"Invalid PNR oracle buffers/dimensions");
    int coefficient_count=1;std::uint64_t grid=1;std::uint32_t strides[4]={};for(int j=0;j<modes;++j){coefficient_count*=3;strides[j]=std::uint32_t(grid);grid*=cutoff+1;}require70(grid<=20000000,"PNR oracle grid exceeds explicit memory limit");
    std::istringstream input(coefficient_text);std::string token;std::vector<Q> coefficients;for(int i=0;i<coefficient_count;++i){require70(bool(input>>token),"Missing determinant coefficient");char* end=nullptr;Q value=strtoflt128(token.c_str(),&end);require70(end&&*end==0&&!isnanq(value)&&!isinfq(value),"Invalid determinant coefficient");coefficients.push_back(value);}require70(!(input>>token),"Extra determinant coefficient");require70(coefficients[0]>0,"Nonpositive determinant constant");
    std::vector<Term70> terms;for(int i=1;i<coefficient_count;++i){if(coefficients[i]==0)continue;Term70 t;t.coefficient=coefficients[i];int label=i;for(int j=0;j<modes;++j){t.power[j]=label%3;label/=3;t.degree+=t.power[j];t.offset+=t.power[j]*strides[j];}terms.push_back(t);}
    std::vector<Q> table(grid,0),totals(cutoff+1,0);table[0]=1/sqrtq(coefficients[0]);std::uint64_t written=0;int occupation[4]={};Q minimum=table[0],mass=0;
    for(int degree=0;degree<=cutoff;++degree){
        std::function<void(int,int,std::uint32_t)> visit=[&](int mode,int left,std::uint32_t code){
            if(mode<modes-1){for(int count=0;count<=left;++count){occupation[mode]=count;visit(mode+1,left-count,code+count*strides[mode]);}return;}
            occupation[mode]=left;code+=left*strides[mode];Q probability=table[0];
            if(degree){Q sum=0;for(const auto& term:terms){if(term.degree>degree)continue;bool valid=true;for(int j=0;j<modes;++j)if(term.power[j]>occupation[j]){valid=false;break;}if(valid)sum+=((Q)degree-(Q)term.degree/2)*term.coefficient*table[code-term.offset];}probability=-sum/((Q)degree*coefficients[0]);table[code]=probability;}
            require70(!isnanq(probability)&&!isinfq(probability)&&probability>=0,"Negative/nonfinite PNR probability; no clipping");require70(written<capacity,"PNR output capacity exhausted");codes[written]=code;probabilities[written]=(double)probability;std::memcpy(binary128+16*written,&probability,16);++written;totals[degree]+=probability;mass+=probability;minimum=std::min(minimum,probability);
        };visit(0,degree,0);
    }
    require70(written==capacity,"PNR output capacity differs from complete simplex");std::ostringstream text;text<<"{\"states\":"<<written<<",\"grid_entries\":"<<grid<<",\"mantissa_bits\":113,\"mass\":\""<<decimal70(mass)<<"\",\"minimum\":\""<<decimal70(minimum)<<"\",\"total_count_probabilities\":[";for(int k=0;k<=cutoff;++k){if(k)text<<',';text<<'"'<<decimal70(totals[k])<<'"';}text<<"]}";require70(text.str().size()<std::size_t(summary_cap),"PNR summary buffer too short");message70(summary,summary_cap,text.str());return 0;
}catch(const std::exception& e){message70(error,error_cap,e.what());return 1;}}
