// Small dense oracle adapter only. The production sampler never expands tensors.
#include "gaussian_mps_core94_v1.cpp"
static Mps93 from_dense93(int n,int d,const double* data){
 int total=1;for(int i=0;i<n;++i)total*=d;Mat93 current(1,total);for(int j=0;j<total;++j)current(0,j)={data[2*j],data[2*j+1]};double norm=current.norm();need93(norm>0&&current.allFinite(),"Dense probe input");current/=norm;Mps93 c(n);int rest=total;
 for(int i=0;i<n-1;++i){rest/=d;int dl=current.rows();Mat93 flat(d*dl,rest);for(int s=0;s<d;++s)for(int a=0;a<dl;++a)flat.row(s*dl+a)=current.row(a).segment(s*rest,rest);Eigen::JacobiSVD<Mat93> svd(flat,Eigen::ComputeThinU|Eigen::ComputeThinV);c[i].resize(d);for(int s=0;s<d;++s)c[i][s]=svd.matrixU().middleRows(s*dl,dl);current=svd.singularValues().asDiagonal()*svd.matrixV().adjoint();}
 c.back().resize(d);for(int s=0;s<d;++s)c.back()[s]=current.col(s);canonical93(c);return c;
}
static void dense93(const Mps93& c,double* output){int d=c[0].size(),n=c.size(),total=1;for(int i=0;i<n;++i)total*=d;for(int key=0;key<total;++key){std::vector<int> index(n);int rem=key;for(int i=n-1;i>=0;--i){index[i]=rem%d;rem/=d;}Mat93 x=Mat93::Ones(1,1);for(int i=0;i<n;++i)x=(x*c[i][index[i]]).eval();output[2*key]=x(0,0).real();output[2*key+1]=x(0,0).imag();}}
API93 int mps94_wait_probe(int n,int d,const double* core,const double* t,const double* alpha,double u,double* flow,double* next_t,double* next_alpha,double* stats,char* error,int cap){try{
 Guard93 guard;need93(n>=1&&n<=4&&d>=2&&d<=6&&u>0&&u<1,"Wait probe arguments");auto c=from_dense93(n,d,core);Gaussian93 g;g.t.resize(n);g.alpha.resize(n);for(int i=0;i<n;++i){g.t[i]=t[i];g.alpha[i]={alpha[2*i],alpha[2*i+1]};}std::fill(stats,stats+18,0);bool found=wait93(c,g,u,stats);stats[1]=found;dense93(c,flow);for(int i=0;i<n;++i){next_t[i]=g.t[i];next_alpha[2*i]=g.alpha[i].real();next_alpha[2*i+1]=g.alpha[i].imag();}return 0;
 }catch(const std::exception& e){message93(error,cap,e.what());return 1;}}
