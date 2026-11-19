// Read-only replay and decimal Gram export. Production approximation unchanged.
#include "gaussian_click_projected88_v2.cpp"
#include <sstream>
#include <iomanip>
#include <cstring>
API int tc88_diagnostic(void* handle,int mode,int budget,const double* draws,double* covariance,int* ports,char* gramtext,int textcap,double* stats,char* message,int cap){try{
 SharedRoundGuard86 rounding;need(handle&&draws&&covariance&&ports&&gramtext&&stats,"Diagnostic88 arguments");auto& c=*static_cast<ClickModel70*>(handle);need(mode>=0&&mode<c.model->m,"Diagnostic88 mode");SharedState86 state{c.model->initial.cast<long double>(),{}};ProjectionTrace88 trace;
 for(int j=0;j<mode;++j){auto step=guarded_step87(c,state,j,draws[j],budget,trace.decision);projected_update88(state,step.step,j,step.bit,budget,trace);}
 int d=state.x.rows(),k=state.clicks.size();Mat82 F(d,2*(k+1));for(int j=0;j<=k;++j){int port=j<k?state.clicks[j]:mode;ports[j]=port;F.col(2*j)=c.a.col(port).cast<long double>();F.col(2*j+1)=c.b.col(port).cast<long double>();}Mat82 products=state.x*F;Mat82 G=F.transpose()*products;G=((G+G.transpose())/2).eval();G.diagonal().array()+=1;
 std::ostringstream text;text<<std::setprecision(42)<<"[";for(int i=0;i<G.rows();++i){if(i)text<<",";text<<"[";for(int j=0;j<G.cols();++j){if(j)text<<",";text<<"\""<<G(i,j)<<"\"";}text<<"]";}text<<"]";auto s=text.str();need(int(s.size())<textcap,"Diagnostic88 text capacity");std::memcpy(gramtext,s.c_str(),s.size()+1);
 for(int i=0;i<d;++i)for(int j=0;j<d;++j)covariance[i*d+j]=double(state.x(i,j));auto sums=vacuum_sum86<Weight83>(G,k);stats[0]=k;stats[1]=double83(sums.mass);stats[2]=double83(sums.zero);stats[3]=double83((sums.mass-sums.zero)/sums.mass);stats[4]=trace.attempts;stats[5]=trace.successes;stats[6]=trace.rejected;stats[7]=trace.max_residual;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
