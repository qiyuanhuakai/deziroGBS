// Independent executable reference: direct native values printed in hex,
// versus wire calls under deliberately different host precision controls.
#include "batched_forest_wire.cpp"
#include <iostream>

static std::string quoted(const char* s){std::string out="\"";for(;*s;++s){if(*s=='"'||*s=='\\')out+='\\';out+=*s;}return out+'"';}
static std::string hex_scalar(long double x){char buffer[128];std::snprintf(buffer,sizeof(buffer),"%La",x);return quoted(buffer);}
static std::string hex_bytes(const unsigned char* bytes,size_t count){static const char* digits="0123456789abcdef";std::string out="\"";for(size_t i=0;i<count;++i){out+=digits[bytes[i]>>4];out+=digits[bytes[i]&15];}return out+'"';}
int main(int argc,char** argv){
    if(argc!=3)return 2;bool wire=std::string(argv[1])=="wire";int controls=std::atoi(argv[2]);
    int m,mode,nz,nr,nf,ne,policy,operation;std::cin>>m>>mode>>nz>>nr>>nf>>ne>>policy>>operation;
    if(!std::cin || m<1 || m>512)return 3;
    std::string scalar;std::cin>>scalar;long double guard=std::strtold(scalar.c_str(),nullptr);
    std::vector<double> V(4*m*m);for(auto& x:V){std::cin>>scalar;x=std::strtod(scalar.c_str(),nullptr);}
    std::vector<int> order(m),z(nz),r(nr),f(nf),edges(2*ne);for(auto* list:{&order,&z,&r,&f,&edges})for(auto& x:*list)std::cin>>x;
    if(!std::cin)return 4;BFQuery q{z.data(),r.data(),f.data(),edges.data(),mode,nz,nr,nf,ne};size_t count=1+2*(1+nf+ne);
    Wire guard_wire;if(!encode(guard,guard_wire))return 5;
    unsigned short cw=control_word();unsigned int mx=_mm_getcsr();
    if(controls==1){cw=static_cast<unsigned short>((cw&~0x0f00u)|0x0200u);mx&=~0xe040u;}
    if(controls==2){cw=static_cast<unsigned short>((cw&~0x0f00u)|0x0a00u);mx=(mx&~0x6000u)|0xc040u;}
    asm volatile("fldcw %0"::"m"(cw));_mm_setcsr(mx);const unsigned short before=control_word();const unsigned int mx_before=_mm_getcsr();
    char error[512]{};BFHandle* h=wire?bfw_create(m,V.data(),order.data(),error,512):bf_create(m,V.data(),order.data(),error,512);if(!h)return 6;
    std::vector<long double> values(count),indicators(count);std::vector<uint8_t> flags(count);std::vector<unsigned char> wvalues(24*count),windicators(24*count);BFOutput out{values.data(),indicators.data(),flags.data(),count};BFConditional condition{};BFWConditional wcondition{};
    int status;
    if(wire)status=operation?bfw_probability(h,&q,policy,guard_wire.data(),24,wvalues.data(),windicators.data(),flags.data(),count,&wcondition,error,512):bfw_query(h,&q,policy,0,wvalues.data(),windicators.data(),flags.data(),count,error,512);
    else status=operation?bf_probability(h,&q,policy,guard,&out,&condition,error,512):bf_query(h,&q,policy,0,&out,error,512);
    BFDiagnostics diagnostics{};if(wire)bfw_diagnostics(h,&diagnostics);else bf_diagnostics(h,&diagnostics);
    if(wire)bfw_destroy(h);else bf_destroy(h);const unsigned short after=control_word();const unsigned int mx_after=_mm_getcsr();
    // Restore a standard environment before output formatting; record controls first.
    unsigned short standard=0x037f;asm volatile("fldcw %0"::"m"(standard));_mm_setcsr(0x1f80);
    std::cout<<"{\"status\":"<<status<<",\"error\":"<<quoted(error)<<",\"control_before\":"<<before<<",\"control_after\":"<<after<<",\"mxcsr_controls_before\":"<<(mx_before&0xffc0u)<<",\"mxcsr_controls_after\":"<<(mx_after&0xffc0u)<<",\"longdouble_queries\":"<<diagnostics.counters[0][BF_QUERIES]<<",\"quad_queries\":"<<diagnostics.counters[1][BF_QUERIES]<<",\"auto_refinements\":"<<diagnostics.auto_refinements;
    if(status==0){
        std::cout<<",\"values\":[";for(size_t j=0;j<count;++j){if(j)std::cout<<',';std::cout<<(wire?hex_bytes(wvalues.data()+24*j,24):hex_scalar(values[j]));}std::cout<<"],\"indicators\":[";
        for(size_t j=0;j<count;++j){if(j)std::cout<<',';std::cout<<(wire?hex_bytes(windicators.data()+24*j,24):hex_scalar(indicators[j]));}std::cout<<"],\"exact_zero\":[";
        for(size_t j=0;j<count;++j){if(j)std::cout<<',';std::cout<<int(flags[j]);}std::cout<<']';
        if(operation){
            double probability=wire?wcondition.probability:condition.probability;uint64_t bits;std::memcpy(&bits,&probability,8);std::cout<<",\"probability_bits\":\""<<std::hex<<bits<<std::dec<<"\",\"precision_used\":"<<(wire?wcondition.precision_used:condition.precision_used)<<",\"refinement_attempted\":"<<(wire?wcondition.refinement_attempted:condition.refinement_attempted)<<",\"first_error\":"<<quoted(wire?wcondition.first_error:condition.first_error)<<",\"zero_score_reasons\":[";
            for(int j=0;j<2;++j){if(j)std::cout<<',';std::cout<<(wire?wcondition.zero_score_reason[j]:condition.zero_score_reason[j]);}std::cout<<"],\"scalars\":[";
            long double numbers[6]={condition.log_scores[0],condition.log_scores[1],condition.log_indicators[0],condition.log_indicators[1],condition.maximum_event_relative_indicator,condition.maximum_log_score_indicator};
            for(int j=0;j<6;++j){if(j)std::cout<<',';std::cout<<(wire?hex_bytes(wcondition.scalars+24*j,24):hex_scalar(numbers[j]));}std::cout<<']';
        }
    }
    std::cout<<"}\n";return 0;
}
