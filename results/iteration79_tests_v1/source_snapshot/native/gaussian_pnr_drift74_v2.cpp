#include "gaussian_pnr_polynomial73_v1.cpp"
API int drift74(void* handle,int count,std::uint64_t seed,double* output,std::uint32_t* counts,char* message,int cap){try{
    auto& c=*static_cast<ClickModel70*>(handle);PrecisionGuard guard;LD73 sums[8]={},squares[8]={};double maximum=0;int horizons[8]={1,2,3,4,8,16,32,64};Stats73 stats;
    for(int row=0;row<count;++row){std::mt19937_64 waits(mix(seed+2ULL*row)),detectors(mix(seed+2ULL*row+1));State73 state{c.model->initial,Vector::Zero(2*c.model->n),{}};int photons=0;bool alive=true;
        while(alive){alive=flow73(c,state,uniform(waits),stats);if(alive){int mode=detector73(c,state,uniform(detectors));state.factors.push_back(mode);compress73(c,state,stats);++photons;maximum=std::max(maximum,state.x.trace()/2);}
            for(int k=0;k<8;++k)if((alive&&photons==horizons[k])||(!alive&&photons<horizons[k])){LD73 value=photons+(alive?state.x.trace()/2:0);sums[k]+=value;squares[k]+=value*value;}}
        counts[row]=photons;
    }
    for(int k=0;k<8;++k){output[k]=double(sums[k]/count);output[8+k]=double(std::sqrt((squares[k]/count-(sums[k]/count)*(sums[k]/count))/count));}output[16]=maximum;return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
