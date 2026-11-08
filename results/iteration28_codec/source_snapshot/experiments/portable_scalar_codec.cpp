// Isolated 24-byte codec prototype. Does not alter or enable forest ABI 1.
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <quadmath.h>
using U128=unsigned __int128;
using Wire=std::array<unsigned char,24>;

static int fail(char* error,size_t cap,const char* message) {
    if(error && cap) std::snprintf(error,cap,"%s",message);
    return 1;
}
template<class T> static bool finite(T x);
template<> bool finite(long double x){return std::isfinite(x);}
template<> bool finite(__float128 x){return finiteq(x);}
template<class T> static T shift(T x,int e);
template<> long double shift(long double x,int e){return std::ldexp(x,e);}
template<> __float128 shift(__float128 x,int e){return scalbnq(x,e);}
template<class T> static T fraction(T x,int* e);
template<> long double fraction(long double x,int* e){return std::frexp(x,e);}
template<> __float128 fraction(__float128 x,int* e){return frexpq(x,e);}
template<class T> static int bits();
template<> int bits<long double>(){return std::numeric_limits<long double>::digits;}
template<> int bits<__float128>(){return FLT128_MANT_DIG;}

template<class T> static bool encode(T value,Wire& out) {
    out.fill(0);
    if(!finite(value)) return false;
    if(value==0) return true;
    out[0]=1; out[1]=value<0;
    T positive=value<0?-value:value; int exponent;
    T frac=fraction(positive,&exponent);
    U128 mantissa=static_cast<U128>(shift(frac,bits<T>())); exponent-=bits<T>();
    if(!mantissa) return false;
    while((mantissa&1)==0){mantissa>>=1;++exponent;}
    uint32_t e=static_cast<uint32_t>(exponent);
    for(int i=0;i<4;++i) out[4+i]=static_cast<unsigned char>(e>>(8*i));
    for(int i=0;i<16;++i) out[8+i]=static_cast<unsigned char>(mantissa>>(8*i));
    return true;
}
template<class T> static bool decode(const Wire& wire,T& result) {
    if(wire[0]==0){for(auto x:wire) if(x) return false; result=0; return true;}
    if(wire[0]!=1 || wire[1]>1 || wire[2] || wire[3]) return false;
    uint32_t e=0; for(int i=0;i<4;++i) e|=uint32_t(wire[4+i])<<(8*i);
    int64_t signed_e=(e&0x80000000u)?int64_t(e)-0x100000000LL:int64_t(e);
    // Native supported exponent range is much narrower; reject before conversion.
    if(signed_e < -20000 || signed_e > 20000) return false;
    U128 mantissa=0; for(int i=0;i<16;++i) mantissa|=U128(wire[8+i])<<(8*i);
    if(!mantissa || !(mantissa&1)) return false;
    int length=0; for(U128 copy=mantissa;copy;copy>>=1) ++length;
    if(length>bits<T>()) return false;
    T value=shift(static_cast<T>(mantissa),static_cast<int>(signed_e));
    if(!finite(value) || value==0) return false;
    if(wire[1]) value=-value;
    Wire roundtrip; if(!encode(value,roundtrip) || roundtrip!=wire) return false;
    result=value; return true;
}

extern "C" int scalar_wire_format(int* output,size_t count) {
    if(!output || count<4) return 1;
    output[0]=1; output[1]=24; output[2]=bits<long double>(); output[3]=bits<__float128>(); return 0;
}
extern "C" int scalar_wire_from_text(const char* input,int precision,unsigned char* output,size_t cap,char* error,size_t errcap) {
    if(!input || !output || cap!=24) return fail(error,errcap,"Invalid buffer");
    char* end=nullptr; Wire wire; errno=0;
    if(precision==0){
        long double value=std::strtold(input,&end);
        if(end==input || *end || (errno==ERANGE && value==0) || !encode(value,wire)) return fail(error,errcap,"Invalid or unrepresentable scalar");
    } else if(precision==1){
        __float128 value=strtoflt128(input,&end);
        if(end==input || *end || (errno==ERANGE && value==0) || !encode(value,wire)) return fail(error,errcap,"Invalid or unrepresentable scalar");
    } else return fail(error,errcap,"Invalid precision");
    std::memcpy(output,wire.data(),24); return 0;
}
extern "C" int scalar_wire_to_hex(const unsigned char* input,size_t count,int precision,char* output,size_t cap,char* error,size_t errcap) {
    if(!input || count!=24 || !output || cap<128) return fail(error,errcap,"Invalid buffer");
    Wire wire; std::memcpy(wire.data(),input,24);
    if(precision==0){
        long double value; if(!decode(wire,value)) return fail(error,errcap,"Noncanonical or unrepresentable wire scalar");
        std::snprintf(output,cap,"%La",value);
    } else if(precision==1){
        __float128 value; if(!decode(wire,value)) return fail(error,errcap,"Noncanonical or unrepresentable wire scalar");
        quadmath_snprintf(output,cap,"%Qa",value);
    } else return fail(error,errcap,"Invalid precision");
    return 0;
}
