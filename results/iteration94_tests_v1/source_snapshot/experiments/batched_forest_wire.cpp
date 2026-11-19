// Explicit scalar transport around the actual, unchanged ABI-1 scientific kernel.
// No long double crosses this boundary. Tested host: Windows x86-64 GCC.
#include "batched_forest_kernel.cpp"
#include "portable_scalar_codec.cpp"
#include <xmmintrin.h>

struct NativeEnvironment {
    FpuScope x87;
    unsigned int mxcsr;
    NativeEnvironment():mxcsr(_mm_getcsr()) {
        // Nearest rounding, no flush-to-zero or denormals-are-zero during work.
        _mm_setcsr(mxcsr & ~0xe040u);
    }
    ~NativeEnvironment(){_mm_setcsr(mxcsr);}
};
struct BFWConditional {
    double probability;
    int precision_used,refinement_attempted,zero_score_reason[2];
    unsigned char scalars[6*24]; // log scores, log indicators, two maxima
    char first_error[BF_ERROR_CAPACITY];
};
static void wire_scalar(long double value,unsigned char* destination) {
    Wire wire;
    if(!encode(value,wire)) throw NumericError("nonfinite wire output");
    std::memcpy(destination,wire.data(),24);
}
static void wire_results(const BFOutput& local,unsigned char* values,unsigned char* indicators,size_t count) {
    for(size_t j=0;j<count;++j){wire_scalar(local.values[j],values+24*j);wire_scalar(local.indicators[j],indicators+24*j);}
}
extern "C" int bfw_formats(int* result,size_t count) {
    if(!result || count<10)return BF_INVALID_ARGUMENT;
    int values[10]={2,24,bits<long double>(),bits<__float128>(),BF_MAX_MODES,BF_MAX_RETAINED,
        int(sizeof(BFQuery)),int(sizeof(BFDiagnostics)),int(sizeof(BFWConditional)),int(sizeof(double))};
    std::copy(values,values+10,result);return BF_OK;
}
extern "C" unsigned int bfw_mxcsr(){return _mm_getcsr();}
extern "C" BFHandle* bfw_create(int modes,const double* covariance,const int* order,char* error,size_t cap) {
    NativeEnvironment env;return bf_create(modes,covariance,order,error,cap);
}
extern "C" void bfw_destroy(BFHandle* handle){NativeEnvironment env;bf_destroy(handle);}
extern "C" void bfw_reset(BFHandle* handle){NativeEnvironment env;bf_reset(handle);}
extern "C" int bfw_diagnostics(BFHandle* handle,BFDiagnostics* result){NativeEnvironment env;return bf_diagnostics(handle,result);}
extern "C" int bfw_query(BFHandle* handle,const BFQuery* query,int precision,int force,
    unsigned char* values,unsigned char* indicators,uint8_t* zeros,size_t count,char* error,size_t cap) {
    NativeEnvironment env;
    return protected_call(error,cap,[&]{
        if(!values || !indicators || !zeros || count<1 || count>4096)throw std::invalid_argument("invalid wire output buffers");
        std::vector<long double> v(count),e(count);BFOutput out{v.data(),e.data(),zeros,count};
        QueryData q=validate_query(handle,query,&out);raw_query(handle,q,precision,force!=0,&out);
        wire_results(out,values,indicators,q.count());
    });
}
extern "C" int bfw_probability(BFHandle* handle,const BFQuery* query,int policy,const unsigned char* guard,size_t guard_bytes,
    unsigned char* values,unsigned char* indicators,uint8_t* zeros,size_t count,BFWConditional* result,char* error,size_t cap) {
    NativeEnvironment env;
    if(!result || !guard || guard_bytes!=24 || !values || !indicators || !zeros || count<1 || count>4096)
        return fail(error,cap,"invalid wire probability buffers");
    *result=BFWConditional{};
    Wire wire;std::memcpy(wire.data(),guard,24);long double native_guard;
    if(!decode(wire,native_guard))return fail(error,cap,"noncanonical or unrepresentable guard");
    int status=BF_OK;
    int wrapper_status=protected_call(error,cap,[&]{
        std::vector<long double> v(count),e(count);BFOutput out{v.data(),e.data(),zeros,count};BFConditional conditional{};
        status=bf_probability(handle,query,policy,native_guard,&out,&conditional,error,cap);
        result->probability=conditional.probability;result->precision_used=conditional.precision_used;
        result->refinement_attempted=conditional.refinement_attempted;
        std::copy(conditional.zero_score_reason,conditional.zero_score_reason+2,result->zero_score_reason);
        std::memcpy(result->first_error,conditional.first_error,BF_ERROR_CAPACITY);
        if(status!=BF_OK)return; // Numerical failure is not a valid event batch.
        QueryData q=validate_query(handle,query,&out);wire_results(out,values,indicators,q.count());
        long double numbers[6]={conditional.log_scores[0],conditional.log_scores[1],conditional.log_indicators[0],conditional.log_indicators[1],conditional.maximum_event_relative_indicator,conditional.maximum_log_score_indicator};
        for(int j=0;j<6;++j)wire_scalar(numbers[j],result->scalars+24*j);
    });
    return wrapper_status==BF_OK ? status : wrapper_status;
}
