#include "pair_matching108_v1.cpp"

// Orthogonal spectra with identical optical parameters share the conditional
// tables. A separate counter identifies every occupied spectral replica.
extern "C" __declspec(dllexport) int repeat108_draw(int m,int k,int rows,int replicas,int n,int cap,int tries,
    int pnr,int wide,int workers,std::uint64_t seed,std::uint32_t off,const C108* u,const double* lambda,
    const double* hazard,const double* first,const double* lev,const double* leverage,const double* columns,
    const double* pair,const int* idx,const double* cached,void* output,std::uint64_t* tally){
    if(replicas<1||replicas>65536||workers<1||k<1||m<1||n<1)return 2;
    Circuit108 input{m,k,rows,u,lev,leverage,columns};
    int err=0,size=3*(cap+2);
    #pragma omp parallel num_threads(workers) reduction(max:err)
    {
        std::vector<std::uint64_t> counts(size);
        #pragma omp for schedule(static)
        for(int j=0;j<n;++j){
            auto *narrow=static_cast<unsigned char*>(output)+std::size_t(j)*m;
            auto *large=wide?static_cast<std::uint32_t*>(output)+std::size_t(j)*m:nullptr;
            if(wide)std::fill_n(large,m,std::uint32_t(0));else std::fill_n(narrow,m,(unsigned char)0);
            R108 skip(seed,0,332,off+j);int cur=0,used=0;
            while(cur<replicas && hazard[k]>0){
                double gap=-std::log(open108(skip))/hazard[k];
                if(gap>=replicas-cur)break;
                cur+=int(gap);++used;
                int rc=wide?sample_row108(input,1,cap,tries,pnr,seed,off+j,lambda,hazard,first,large,counts.data(),nullptr,pair,cur,true,false,cached,idx)
                        :sample_row108(input,1,cap,tries,pnr,seed,off+j,lambda,hazard,first,narrow,counts.data(),nullptr,pair,cur,true,false,cached,idx);
                err=std::max(err,rc);++cur;
            }
            counts[0]+=replicas-used;
        }
        #pragma omp critical
        for(int j=0;j<size;++j)tally[j]+=counts[j];
    }
    return err;
}
