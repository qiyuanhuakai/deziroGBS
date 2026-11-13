#include <quadmath.h>
#include <xmmintrin.h>
#include <cstdlib>
extern "C" __declspec(dllexport) int quad86_probe(int op,const char* aa,const char* bb,const char* cc,unsigned* state,char* result,int cap){
 unsigned saved=_mm_getcsr();unsigned short x87;asm volatile("fnstcw %0":"=m"(x87));unsigned short fixed=(x87&~0x0f00u)|0x0300u;asm volatile("fldcw %0"::"m"(fixed));_mm_setcsr(saved&~0x6000u);
 volatile __float128 a=strtoflt128(aa,nullptr),b=strtoflt128(bb,nullptr),c=strtoflt128(cc,nullptr),r=0;unsigned short before;asm volatile("fnstcw %0":"=m"(before));state[0]=before;state[1]=_mm_getcsr();
 if(op==0)r=sqrtq(a);else if(op==1)r=fmaq(a,b,c);else if(op==2)r=a/b;else if(op==3){volatile double d=static_cast<double>(a);r=d;}else if(op==4){volatile long double d=static_cast<long double>(a);r=d;}
 unsigned short after;asm volatile("fnstcw %0":"=m"(after));state[2]=after;state[3]=_mm_getcsr();int n=quadmath_snprintf(result,cap,"%.85Qg",__float128(r));_mm_setcsr(saved);asm volatile("fldcw %0"::"m"(x87));return n>=0&&n<cap?0:1;
}
