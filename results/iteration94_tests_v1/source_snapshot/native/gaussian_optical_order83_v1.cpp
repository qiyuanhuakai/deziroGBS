// Input-only nearest-neighbour order in normalized occupied optical row space.
// Keep the first pinned ports unchanged for directly comparable prefix laws.
// Heuristic, not a claim of full-law improvement. Never inspect sample records.
#include "gaussian_reset_cpu.cpp"
#include <omp.h>
API int order83_optical(int m,int n,int pinned,const double* tt,int threads,int* order,double* stats,char* message,int cap){try{
 PrecisionGuard guard;need(m>=n&&m<=4096&&n>=1&&n<=128&&pinned>=1&&pinned<=m&&tt&&order&&stats&&threads>=1&&threads<=64,"Invalid optical order inputs");Eigen::setNbThreads(threads);Complex t(m,n);for(int i=0;i<m;++i)for(int j=0;j<n;++j){auto k=2*(std::size_t(i)*n+j);need(std::isfinite(tt[k])&&std::isfinite(tt[k+1]),"Nonfinite optical map");t(i,j)={tt[k],tt[k+1]};}Eigen::MatrixXcd gram=t*t.adjoint();Vector norms=gram.diagonal().real();Real similarity=Real::Zero(m,m);stats[0]=0;stats[1]=0;stats[2]=0;stats[3]=0;
 for(int i=0;i<m;++i)for(int j=0;j<i;++j){double denominator=norms[i]*norms[j];double s=denominator>0?std::norm(gram(i,j))/denominator:0;need(s<=1+1e-12&&s>=0&&std::isfinite(s),"Invalid optical row overlap");similarity(i,j)=similarity(j,i)=s;if(s>1-1e-8)stats[0]++;}
 std::vector<bool> used(m,false);for(int i=0;i<pinned;++i){order[i]=i;used[i]=true;}int current=pinned-1;for(int i=pinned;i<m;++i){int best=-1;for(int j=0;j<m;++j)if(!used[j]&&(best<0||similarity(current,j)>similarity(current,best)))best=j;order[i]=best;used[best]=true;current=best;}
 for(int i=1;i<m;++i){stats[1]+=similarity(i-1,i);stats[2]+=similarity(order[i-1],order[i]);if(similarity(order[i-1],order[i])>1-1e-8)stats[3]++;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
