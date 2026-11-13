// v1 sampling arithmetic is frozen and unchanged. This adds an independent
// evaluator of its actual integer alias/rejection law on <=8 physical ports.
#include "gaussian_pnr_factor82_v1.cpp"
API int factor82_vacuum(void* ptr,int n,double z,int depth,const int* ports,int threads,double* vacuum,double* low,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(ptr&&ports&&vacuum&&low&&stats&&n>=1&&n<=128&&z>=0&&z<1&&depth>=1&&depth<=8&&threads>=1&&threads<=64,"Invalid factor vacuum audit arguments");auto& p=*static_cast<FactorPlan82*>(ptr);need(n==p.owner->model->n&&depth<=p.m&&std::uint64_t(p.latent)*p.m*p.m<=20000000000ULL,"Factor vacuum audit work bound");
 std::vector<int> selected(p.m,-1);for(int i=0;i<depth;++i){need(ports[i]>=0&&ports[i]<p.m&&selected[ports[i]]<0,"Invalid or duplicate physical port");selected[ports[i]]=i;}
 std::vector<long double> proposal(std::size_t(p.m)*p.m);for(int i=0;i<p.m;++i){auto q=p.proposals[i]->law();for(int j=0;j<p.m;++j)proposal[std::size_t(i)*p.m+j]=q[j];}auto weights=p.labels->law();int patterns=1<<depth;std::vector<std::vector<long double>> sums(threads,std::vector<long double>(patterns)),lows=sums;std::vector<long double> minima(threads,1);std::vector<std::exception_ptr> failures(threads);std::atomic<bool> failed{false};long double nb[5],shape=n/2.L;nb[0]=std::pow(1-static_cast<long double>(z),shape);for(int j=1;j<=4;++j)nb[j]=nb[j-1]*z*(shape+j-1)/j;
 #pragma omp parallel num_threads(threads)
 {int id=omp_get_thread_num();PrecisionGuard precision;
  #pragma omp for schedule(static)
  for(int l=0;l<p.latent;++l){if(failed.load())continue;try{auto root=p.roots[l]->law();long double row[8]={},col[8]={},pair[8][8]={};const auto* acceptance=p.accept.data()+std::size_t(l)*p.m;
   for(int i=0;i<p.m;++i){if(root[i]==0)continue;const auto* q=proposal.data()+std::size_t(i)*p.m;long double denominator=0;for(int j=0;j<p.m;++j)denominator+=q[j]*acceptance[j];need(denominator>0,"Zero reachable exact acceptance denominator");int ri=selected[i];if(ri>=0)row[ri]=root[i];for(int j=0;j<depth;++j){long double value=root[i]*q[ports[j]]*acceptance[ports[j]]/denominator;col[j]+=value;if(ri>=0)pair[ri][j]=value;}}
   long double miss[256];miss[0]=1;for(int mask=0;mask<patterns;++mask){if(mask){int bit=__builtin_ctz(unsigned(mask)),previous=mask&(mask-1);miss[mask]=miss[previous]-row[bit]-col[bit]+pair[bit][bit];for(int j=0;j<depth;++j)if(previous>>j&1)miss[mask]+=pair[bit][j]+pair[j][bit];if(depth==p.m&&mask==patterns-1)miss[mask]=0;}
    long double s=miss[mask];need(s>=0&&s<=1,"Integer factor no-click probability outside [0,1]");minima[id]=std::min(minima[id],s);long double full=std::exp(shape*(std::log1p(-static_cast<long double>(z))-std::log1p(-static_cast<long double>(z)*s))),small=nb[4];for(int j=3;j>=0;--j)small=small*s+nb[j];sums[id][mask]+=weights[l]*full;lows[id][mask]+=weights[l]*small;
   }
  }catch(...){failures[id]=std::current_exception();failed.store(true);}}
 }
 for(auto& e:failures)if(e)std::rethrow_exception(e);for(int mask=0;mask<patterns;++mask){long double full=0,small=0;for(int t=0;t<threads;++t){full+=sums[t][mask];small+=lows[t][mask];}vacuum[mask]=double(full);low[mask]=double(small);}stats[0]=*std::min_element(minima.begin(),minima.end());stats[1]=0;for(auto x:nb)stats[1]+=double(x);stats[2]=double(std::uint64_t(p.latent)*p.m*p.m);stats[3]=threads;return 0;
}catch(const std::exception& e){error(message,cap,e.what());return 1;}}
