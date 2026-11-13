#include "gaussian_pnr_polynomial73_v1.cpp"
API int drift74(void* handle,int count,std::uint64_t seed,double* output,char* message,int cap){try{
    auto& c=*static_cast<ClickModel70*>(handle);PrecisionGuard guard;LD73 first=0,second=0,before=0,after=0,stops=0;Stats73 stats;
    for(int row=0;row<count;++row){std::mt19937_64 waits(mix(seed+2ULL*row)),detectors(mix(seed+2ULL*row+1));State73 state{c.model->initial,Vector::Zero(2*c.model->n),{}};
        if(!flow73(c,state,uniform(waits),stats)){++stops;continue;}before+=state.x.trace()/2;int mode=detector73(c,state,uniform(detectors));state.factors.push_back(mode);compress73(c,state,stats);LD73 n=state.x.trace()/2;after+=n;first+=1+n;
        if(flow73(c,state,uniform(waits),stats)){mode=detector73(c,state,uniform(detectors));state.factors.push_back(mode);compress73(c,state,stats);second+=2+state.x.trace()/2;}else second+=1;
    }
    output[0]=double(first/count);output[1]=double(second/count);output[2]=double(before/count);output[3]=double(after/count);output[4]=double(stops/count);return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
