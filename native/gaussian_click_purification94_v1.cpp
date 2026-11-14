// A conditional-state building block, not a complete scalable GBS sampler.
// A[(occupation,remaining),environment] purifies the premeasurement density.
// The click update retains ALL nonzero occupations coherently with their
// environment labels; it never samples a hidden photon number.
#include "gaussian_mps_core94_v1.cpp"
API93 int click94_update(int d,int rest,int environment,const double* data,
 int click,int maxrank,double* output,double* stats,char* error,int cap){try{
 Guard93 guard;
 need93(d>=2&&d<=32&&rest>=1&&rest<=4096&&environment>=1&&environment<=256
  &&data&&output&&stats&&(click==0||click==1)&&maxrank>=1&&maxrank<=rest,
  "Conditional purification arguments");
 int columns=environment*(click?d-1:1);
 Mat93 branch(rest,columns);long double total=0,selected=0;
 for(int s=0;s<d;++s)for(int r=0;r<rest;++r)for(int e=0;e<environment;++e){
  auto at=(std::size_t(s)*rest+r)*environment+e;
  Z93 z(data[2*at],data[2*at+1]);need93(std::isfinite(std::abs(z)),"Nonfinite purification");
  long double mass=(long double)z.real()*z.real()+(long double)z.imag()*z.imag();total+=mass;
  if((click&&s>0)||(!click&&s==0)){branch(r,(click?s-1:0)*environment+e)=z;selected+=mass;}
 }
 need93(total>0&&std::isfinite(total),"Zero purification norm");
 std::fill(output,output+std::size_t(rest)*maxrank*2,0);std::fill(stats,stats+6,0);
 stats[0]=double(selected/total);stats[3]=double(total);
 if(selected==0)return 0;
 Mat93 U,Vt;Vec93 singular;thin_svd93(branch,U,singular,Vt);
 int rank=std::min<int>(maxrank,singular.size());long double kept=0,discarded=0;
 for(int j=0;j<singular.size();++j){long double w=(long double)singular[j]*singular[j];if(j<rank)kept+=w;else discarded+=w;}
 need93(kept>0,"Zero retained conditional norm");
 Mat93 factor=U.leftCols(rank)*singular.head(rank).asDiagonal();factor/=std::sqrt(kept);
 for(int r=0;r<rest;++r)for(int j=0;j<rank;++j){auto z=factor(r,j);auto at=std::size_t(r)*maxrank+j;output[2*at]=z.real();output[2*at+1]=z.imag();}
 stats[1]=double(discarded/(kept+discarded));stats[2]=rank;
 stats[4]=double(std::abs(kept+discarded-selected)/selected);
 stats[5]=double(std::abs(norm93(factor)-1));return 0;
 }catch(const std::exception& e){message93(error,cap,e.what());return 1;}}
