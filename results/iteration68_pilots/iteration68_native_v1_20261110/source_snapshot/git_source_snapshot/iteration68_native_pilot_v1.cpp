// Exploratory fused CPU trajectory engine for the existing finite-MPS POVM.
// No new Gaussian approximation and no claim of exact floating-point probabilities.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>
#define API extern "C" __declspec(dllexport)
struct Site { int l,d,r; std::vector<double> re,im; };
struct Model { std::vector<Site> sites; int bond=1; };
static void error_text(char* out,int cap,const std::string& value) {
    if(out && cap>0){const auto n=std::min(static_cast<std::size_t>(cap-1),value.size());std::memcpy(out,value.data(),n);out[n]=0;}
}
static void need(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
API int nds_abi(){return 1;}
API void* nds_create(int m,const int* dims,const double* const* arrays,char* error,int cap){
    Model* model=nullptr;
    try {
        need(m>0 && dims && arrays,"invalid model arrays");model=new Model;int previous=1;
        for(int k=0;k<m;++k){int l=dims[3*k],d=dims[3*k+1],r=dims[3*k+2];need(l==previous && d>0 && r>0 && arrays[k],"invalid tensor dimensions");
            const std::uint64_t elements=std::uint64_t(l)*d*r;need(elements<=std::numeric_limits<std::size_t>::max()/sizeof(double)/2,"tensor size overflow");Site site{l,d,r,{}, {}};site.re.resize(elements);site.im.resize(elements);
            for(std::size_t j=0;j<elements;++j){double a=arrays[k][2*j],b=arrays[k][2*j+1];need(std::isfinite(a)&&std::isfinite(b),"nonfinite tensor");site.re[j]=a;site.im[j]=b;}
            model->bond=std::max({model->bond,l,r});model->sites.push_back(std::move(site));previous=r;
        }
        need(previous==1,"invalid right boundary");return model;
    }catch(const std::exception& e){delete model;error_text(error,cap,e.what());return nullptr;}
}
API void nds_destroy(void* handle){delete static_cast<Model*>(handle);}

// Coherent directions are site-major [sample,d,complex], and split masses
// site-major [sample,2]. Their preparation is timed separately by the caller.
API int nds_sample(void* handle,int batch,const double* const* coherent,
                   const double* const* split,const double* uniforms,int threads,int tile,
                   int* labels,double* selected,double* vacuum,double* residual,
                   char* error,int cap){
    try {
        need(handle && batch>=0 && threads>=1 && threads<=64 && tile>=1 && tile<=128,"invalid call shape");
        const auto& model=*static_cast<Model*>(handle);const int m=static_cast<int>(model.sites.size()),bond=model.bond;
        need(coherent && split && uniforms && labels && selected && vacuum && residual,"null call array");*residual=0.;if(!batch)return 0;
        std::vector<double> er(std::size_t(batch)*bond,0.),ei(er.size(),0.),nr(er.size(),0.),ni(er.size(),0.);
        for(int b=0;b<batch;++b)er[std::size_t(b)*bond]=1.;
        for(int site_index=0;site_index<m;++site_index){const Site& s=model.sites[site_index];const int width=s.d*s.r;need(coherent[site_index]&&split[site_index],"null site data");
            int failed=0;std::string failure;double site_residual=0.;
            #pragma omp parallel for schedule(static) num_threads(threads) reduction(max:site_residual)
            for(int first=0;first<batch;first+=tile){
                try {
                    const int count=std::min(tile,batch-first);std::vector<double> ar(std::size_t(count)*width,0.),ai(ar.size(),0.);
                    // A is scanned once per small trajectory tile; contiguous
                    // physical/right indices enable SIMD without complex ABI casts.
                    for(int l=0;l<s.l;++l){const double* tr=s.re.data()+std::size_t(l)*width;const double* ti=s.im.data()+std::size_t(l)*width;
                        for(int b=0;b<count;++b){double xr=er[std::size_t(first+b)*bond+l],xi=ei[std::size_t(first+b)*bond+l];double* yr=ar.data()+std::size_t(b)*width;double* yi=ai.data()+std::size_t(b)*width;
                            #pragma omp simd
                            for(int j=0;j<width;++j){yr[j]+=xr*tr[j]-xi*ti[j];yi[j]+=xr*ti[j]+xi*tr[j];}
                        }
                    }
                    std::vector<double> pr(s.r),pi(s.r),mass(s.d),weights(s.d+1);std::vector<int> order(s.d+1);
                    for(int b=0;b<count;++b){int sample=first+b;double* xr=ar.data()+std::size_t(b)*width;double* xi=ai.data()+std::size_t(b)*width;const double* c=coherent[site_index]+std::size_t(sample)*s.d*2;const double* partition=split[site_index]+std::size_t(sample)*2;
                        need(std::isfinite(partition[0])&&std::isfinite(partition[1])&&partition[0]>=0&&partition[1]>=0&&std::abs(partition[0]+partition[1]-1.)<2e-14,"invalid Poisson split");
                        double cnorm=0;for(int n=0;n<s.d;++n){need(std::isfinite(c[2*n])&&std::isfinite(c[2*n+1]),"nonfinite coherent vector");cnorm+=c[2*n]*c[2*n]+c[2*n+1]*c[2*n+1];}
                        need(std::abs(cnorm-1.)<1e-12&&c[0]>=0&&c[1]==0.,"invalid coherent normalization");const double denom=1.+c[0];
                        std::fill(pr.begin(),pr.end(),0.);std::fill(pi.begin(),pi.end(),0.);
                        for(int n=0;n<s.d;++n){double cr=c[2*n],ci=c[2*n+1];for(int r=0;r<s.r;++r){std::size_t j=std::size_t(n)*s.r+r;pr[r]+=cr*xr[j]+ci*xi[j];pi[r]+=cr*xi[j]-ci*xr[j];}}
                        std::fill(mass.begin(),mass.end(),0.);
                        // Test structural support before overwriting the input.
                        std::vector<unsigned char> physical_nonzero(s.d,0);for(int n=0;n<s.d;++n)for(int r=0;r<s.r;++r)if(xr[n*s.r+r]!=0.||xi[n*s.r+r]!=0.)physical_nonzero[n]=1;
                        for(int r=0;r<s.r;++r){double dr=(xr[r]+pr[r])/denom,di=(xi[r]+pi[r])/denom;
                            for(int n=1;n<s.d;++n){std::size_t j=std::size_t(n)*s.r+r;double cr=c[2*n],ci=c[2*n+1];xr[j]-=cr*dr-ci*di;xi[j]-=cr*di+ci*dr;}
                            xr[r]=-pr[r];xi[r]=-pi[r];
                        }
                        for(int n=0;n<s.d;++n){bool nonzero=false;for(int r=0;r<s.r;++r){std::size_t j=std::size_t(n)*s.r+r;mass[n]+=xr[j]*xr[j]+xi[j]*xi[j];nonzero|=(xr[j]!=0.||xi[j]!=0.);}need(std::isfinite(mass[n]),"nonfinite refined norm");
                            if(mass[n]==0.){need(!nonzero,"positive refined norm underflow");bool reachable=false;
                                for(int k=0;k<s.d;++k){double rr,ri;if(n==0){rr=-c[2*k];ri=c[2*k+1];}else if(k==0){rr=-c[2*n];ri=-c[2*n+1];}else{rr=-(c[2*n]*c[2*k]+c[2*n+1]*c[2*k+1])/denom;ri=-(c[2*n+1]*c[2*k]-c[2*n]*c[2*k+1])/denom;if(k==n)rr+=1.;}reachable|=physical_nonzero[k]&&(rr!=0.||ri!=0.);}
                                need(!reachable,"unresolved refined cancellation");
                            }
                        }
                        weights[0]=partition[0]*mass[0];weights[1]=partition[1]*mass[0];for(int n=1;n<s.d;++n)weights[n+1]=mass[n];
                        for(int k=0;k<2;++k)need(!(partition[k]>0.&&mass[0]>0.&&weights[k]==0.),"positive refined probability underflow");
                        std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return weights[a]<weights[b];});double total=0.;std::vector<double> cumulative(s.d+1);
                        for(int k=0;k<=s.d;++k){double w=weights[order[k]];need(w>=0.&&std::isfinite(w),"invalid weight");double next=total+w;need(!(w>0.&&next<=total),"positive categorical interval disappeared");cumulative[k]=next;total=next;}
                        need(total>0.&&std::isfinite(total),"empty categorical law");double drift=std::abs(total-1.);site_residual=std::max(site_residual,drift);need(drift<=1e-9,"conditional norm drift");
                        const std::size_t output=std::size_t(sample)*m+site_index;double u=uniforms[output];need(std::isfinite(u)&&u>=0.&&u<1.,"invalid uniform");double target=u*total;need(!(u>0.&&target==0.),"categorical coordinate underflow");int k=0;while(k<=s.d&&cumulative[k]<=target)++k;need(k<=s.d,"categorical endpoint outside support");int chosen=order[k],physical=chosen<=1?0:chosen-1;need(mass[physical]>0.,"selected zero norm");double scale=std::sqrt(mass[physical]);
                        for(int r=0;r<s.r;++r){nr[std::size_t(sample)*bond+r]=xr[physical*s.r+r]/scale;ni[std::size_t(sample)*bond+r]=xi[physical*s.r+r]/scale;}
                        labels[output]=chosen;selected[output]=weights[chosen]/total;vacuum[output]=weights[0]/total;
                    }
                }catch(const std::exception& e){
                    #pragma omp critical
                    {failed=1;if(failure.empty())failure="site "+std::to_string(site_index)+", tile "+std::to_string(first)+": "+e.what();}
                }
            }
            if(failed)throw std::runtime_error(failure);*residual=std::max(*residual,site_residual);er.swap(nr);ei.swap(ni);
        }
        return 0;
    }catch(const std::exception& e){error_text(error,cap,e.what());return 1;}
}
