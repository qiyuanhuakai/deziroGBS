#include "pair_matching108_v1.cpp"

extern "C" __declspec(dllexport) int compact108_draw(int m,int k,int spectra,int rows,int n,int cap,int tries,int pnr,int workers,
    int packed,int slots,int wide,std::uint64_t seed,std::uint32_t offset,const C108* u,const double* lambda,const double* hazard,
    const double* first,const double* lev,const double* leverage_cdf,const double* column,const double* single,
    void* dense,int* lengths,int* ids,std::uint32_t* multiplicity,std::uint64_t* tally){
    if(m<1||k<1||rows<m||n<1||slots<1||cap<1||cap>6||workers<1)return 2;
    Circuit108 c{m,k,rows,u,lev,leverage_cdf,column};
    int error=0,size=spectra*(cap+2)*3;
    #pragma omp parallel num_threads(workers) reduction(max:error)
    {
        std::vector<std::uint64_t> counts(size);
        std::vector<int> hits;
        #pragma omp for schedule(static)
        for(int j=0;j<n;++j){
            int rc;
            if(packed){
                hits.clear();
                rc=sample_row108(c,spectra,cap,tries,pnr,seed,offset+j,lambda,hazard,first,
                    static_cast<unsigned char*>(nullptr),counts.data(),&hits,single);
                std::sort(hits.begin(),hits.end());
                int len=0;
                for(std::size_t a=0;a<hits.size();){
                    std::size_t b=a+1;while(b<hits.size()&&hits[b]==hits[a])++b;
                    if(len<slots){
                        ids[std::size_t(j)*slots+len]=hits[a];
                        if(pnr)multiplicity[std::size_t(j)*slots+len]=std::uint32_t(b-a);
                    }
                    ++len;a=b;
                }
                lengths[j]=len;if(len>slots && !rc)rc=3;
            }else if(wide){
                rc=sample_row108(c,spectra,cap,tries,pnr,seed,offset+j,lambda,hazard,first,
                    static_cast<std::uint32_t*>(dense)+std::size_t(m)*j,counts.data(),nullptr,single);
            }else{
                rc=sample_row108(c,spectra,cap,tries,pnr,seed,offset+j,lambda,hazard,first,
                    static_cast<unsigned char*>(dense)+std::size_t(m)*j,counts.data(),nullptr,single);
            }
            // Fatal statuses take priority over recoverable capacity retries.
            error=std::max(error,rc==2?4:rc);
        }
        #pragma omp critical
        for(int j=0;j<size;++j)tally[j]+=counts[j];
    }
    return error;
}
