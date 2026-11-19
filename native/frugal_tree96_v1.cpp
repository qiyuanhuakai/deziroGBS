#include "frugal_prefix81_adapter_v1.cpp"

API int fc96_full16(int cutoff, const double* cov, double* threshold, double* prev,
                    double* proposed, double* stats, char* error, int cap) {
    try {
        if(cutoff<1 || cutoff>5 || !cov || !threshold || !prev || !proposed || !stats)
            throw std::invalid_argument("invalid prefix arguments");
        Threads72 guard(1);
        prefix81<16>(cutoff,16,cov,threshold,prev,proposed,stats);
        return 0;
    } catch(const std::exception& e) {
        if(error && cap>0) std::snprintf(error,cap,"%s",e.what());
        return 1;
    }
}

API int fc96_sample16(int cutoff,int n,unsigned seed,const double* cov,unsigned char* out,
                      double* stats,char* error,int cap) {
    try {
        if(cutoff<1 || cutoff>5 || n<1 || !cov || !out || !stats)
            throw std::invalid_argument("invalid sampling arguments");
        Threads72 guard(1);
        run72<16>(cutoff,n,seed,cov,out,stats);
        return 0;
    } catch(const std::exception& e) {
        if(error && cap>0) std::snprintf(error,cap,"%s",e.what());
        return 1;
    }
}
