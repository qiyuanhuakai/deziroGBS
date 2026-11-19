// Independent numerical-reference entry point. Production v1 remains frozen.
#include "gaussian_click_guarded87_v1.cpp"
API int ball87_gram_probe(int clicks,const double* input,char* midpoint,char* radius,int size,char* message,int cap){try{
 SharedRoundGuard86 guard;need(clicks>=0&&clicks<=7&&input&&midpoint&&radius&&size>=128,"Guard87 Gram probe arguments");int d=2*(clicks+1);Mat82 gram(d,d);for(int i=0;i<d;++i)for(int j=0;j<d;++j){need(std::isfinite(input[i*d+j]),"Guard87 finite Gram input");gram(i,j)=input[i*d+j];}need(gram==gram.transpose(),"Guard87 symmetric Gram probe");auto s=vacuum_sum86<Ball87>(gram,clicks);auto p=(s.mass-s.zero)/s.mass;int nm=quadmath_snprintf(midpoint,size,"%.85Qg",p.value),nr=std::snprintf(radius,size,"%.85Le",p.radius);need(nm>0&&nr>0&&nm<size&&nr<size,"Guard87 Gram probe buffer");return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
