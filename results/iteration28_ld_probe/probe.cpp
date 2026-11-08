#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <quadmath.h>
void show(const char* name,__float128 value){char b[256];quadmath_snprintf(b,sizeof(b),"%Qa",value);std::printf("%s %s\n",name,b);}
int main(){
 const char* text="0x216be56141e04efp-5901";
 long double a=std::strtold(text,nullptr), b=(long double)strtoflt128(text,nullptr);int ea,eb;
 show("strtold",(__float128)a);show("quad_cast",(__float128)b);
 long double fa=std::frexp(a,&ea);__float128 fb=frexpq((__float128)b,&eb);
 show("frexp_ld",(__float128)fa);show("frexp_quad",fb);std::printf("exponents %d %d\n",ea,eb);
 show("ldexp_ld",(__float128)std::ldexp(b,5901));show("ldexp_quad",scalbnq((__float128)b,5901));
 unsigned long long whole=0xffffffffffffffffULL;long double direct=(long double)whole;
 show("cast_u64",(__float128)direct);show("ldexp_max",(__float128)std::ldexp(direct,-64));
 char raw[256];std::snprintf(raw,sizeof(raw),"%La",b);std::printf("printf_ld %s\n",raw);
}