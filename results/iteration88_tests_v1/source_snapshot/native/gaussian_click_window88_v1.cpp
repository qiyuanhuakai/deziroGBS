// Retain a sliding click window. On overflow, moment-match only the oldest
// single click factor into the Gaussian base; keep all later factors explicit.
// This changes the approximation law. It is not an exactness claim.
#include "gaussian_click_guarded87_v1.cpp"
static int capacity88(int budget){need(budget>=1&&budget<=128&&(budget&(budget-1))==0,"Window88 power-of-two budget required");return 31-__builtin_clz(unsigned(budget));}
static void window_update88(SharedState86& state,SharedStep86& step,int mode,bool click,int capacity,SharedTrace86& trace){
 if(!click){shared_update86(state,step,mode,false,256,trace);return;}
 need(step.p1>Weight83(0),"Window88 impossible click");state.clicks.push_back(mode);trace.peak=std::max(trace.peak,double(state.clicks.size()));
 if(int(state.clicks.size())>capacity){
  // Column zero is the oldest retained click; with capacity0 it is the new one.
  Vec82 a=step.forms.col(0),b=step.forms.col(1),xa=step.products.col(0),xb=step.products.col(1);
  Q82 g00=Q82(a.dot(xa))+1,g11=Q82(b.dot(xb))+1,g01=(Q82(a.dot(xb))+Q82(b.dot(xa)))/2,det=g00*g11-g01*g01;
  need(g00>0&&g11>0&&finite82(det)&&det>1,"Window88 oldest-click marginal is not positive; no clipping");
  Q82 ratio=1/expm1q(logq(det)/2);need(ratio>0&&finite82(ratio),"Window88 oldest-click moment ratio invalid");long double i00=static_cast<long double>(ratio*g11/det),i01=static_cast<long double>(-ratio*g01/det),i11=static_cast<long double>(ratio*g00/det);
  state.x+=xa*(i00*xa+i01*xb).transpose()+xb*(i01*xa+i11*xb).transpose();state.clicks.erase(state.clicks.begin());++trace.merges;
 }
 state.x=((state.x+state.x.transpose())/2).eval();need(state.x.allFinite(),"Window88 covariance overflow");
}
API int tc88_window_prefix(void* handle,int length,const std::uint8_t* bits,int budget,double* probability,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;int capacity=capacity88(budget);need(handle&&bits&&probability&&stats,"Window88 prefix arguments");auto& c=*static_cast<ClickModel70*>(handle);need(length>=0&&length<=std::min(16,c.model->m),"Window88 prefix length");SharedState86 state{c.model->initial.cast<long double>(),{}};SharedTrace86 trace;Weight83 joint(1);std::fill(stats,stats+6,0);
 for(int mode=0;mode<length;++mode){need(bits[mode]<=1,"Window88 bit");auto step=shared_step86(c,state,mode,trace);joint=joint*(bits[mode]?step.p1:step.p0);if(!(joint>Weight83(0)))break;if(mode+1<length)window_update88(state,step,mode,bits[mode],capacity,trace);stats[5]=mode+1;}
 *probability=double83(joint);stats[0]=trace.evaluations;stats[1]=trace.merges;stats[2]=trace.peak;stats[3]=trace.condition;stats[4]=trace.wide;return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int tc88_window_sample(void* handle,int count,int threads,int budget,int use_guard,const double* draws,std::uint8_t* bits,double* midpoint,double* bounds,std::uint8_t* kind,int* errors,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;int capacity=capacity88(budget);need(handle&&count>=0&&count<=4096&&threads>=1&&threads<=64&&(use_guard==0||use_guard==1)&&draws&&bits&&midpoint&&bounds&&kind&&errors&&stats,"Window88 sample arguments");auto& c=*static_cast<ClickModel70*>(handle);int m=c.model->m;std::size_t entries=std::size_t(count)*m;std::fill(bits,bits+entries,0);std::fill(midpoint,midpoint+entries,std::numeric_limits<double>::quiet_NaN());std::fill(bounds,bounds+2*entries,std::numeric_limits<double>::quiet_NaN());std::fill(kind,kind+entries,255);std::fill(errors,errors+2*count,0);std::fill(stats,stats+12*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){SharedRoundGuard86 precision;for(;;){int row=next.fetch_add(1);if(row>=count)break;int mode=0,stage=1;GuardTrace87 trace;try{SharedState86 state{c.model->initial.cast<long double>(),{}};for(;mode<m;++mode){stage=1;auto at=std::size_t(row)*m+mode;double u=draws[at];need(std::isfinite(u)&&u>=0&&u<1,"Window88 explicit uniform");GuardStep87 step;
   if(use_guard)step=guarded_step87(c,state,mode,u,256,trace); // No joint-factor closure: only the oldest single factor is merged below.
   else{step.step=shared_step86(c,state,mode,trace.base);step.bit=Weight83(Q82(u))<step.step.p1;step.midpoint=double83(step.step.p1);step.lower=std::nextafter(step.midpoint,-std::numeric_limits<double>::infinity());step.upper=std::nextafter(step.midpoint,std::numeric_limits<double>::infinity());step.kind=step.step.wide?2:0;}
   bits[at]=step.bit;midpoint[at]=step.midpoint;bounds[2*at]=step.lower;bounds[2*at+1]=step.upper;kind[at]=step.kind;if(mode+1<m){stage=2;window_update88(state,step.step,mode,step.bit,capacity,trace.base);}
  }}catch(const std::exception& e){errors[2*row]=stage;errors[2*row+1]=mode;messages[row]=e.what();}auto* s=stats+12*row;s[0]=trace.base.evaluations;s[1]=trace.base.merges;s[2]=trace.base.peak;s[3]=trace.base.condition;s[4]=trace.base.wide;s[5]=mode;s[6]=trace.balls;s[7]=trace.certified;s[8]=trace.ambiguous;s[9]=trace.closure_wide;s[10]=trace.ball_failure;s[11]=trace.max_width;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" mode="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
