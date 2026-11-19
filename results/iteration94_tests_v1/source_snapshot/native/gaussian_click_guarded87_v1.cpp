// Decision-directed precision for the frozen86 shared-Gaussian representation.
// A ball filter is used only where86 would use paired-binary128. A closure or
// ambiguous uniform still uses the frozen wide calculation. Midpoints are NOT
// advertised as exact conditional probabilities; the decision brackets are saved.
struct Ball87;
static Ball87 sqrt86(Ball87);
static bool finite86(Ball87);
#include "gaussian_click_shared86_v2.cpp"
using Rad87=long double;
static constexpr Rad87 eps87=std::numeric_limits<Rad87>::epsilon();
static constexpr Rad87 tiny87=std::numeric_limits<Rad87>::denorm_min();
// Positive radius expressions below contain at most eight elementary operations.
// The32-epsilon inflation dominates their rounding, with a separate gradual-
// underflow allowance. Fast-math and contraction must remain disabled.
static Rad87 inflate87(Rad87 x){return x*(1+32*eps87)+32*tiny87;}
static Rad87 qhi87(Q82 x){Rad87 a=std::abs(static_cast<Rad87>(x));return a*(1+4*eps87)+tiny87;}
static Rad87 qlo87(Q82 x){Rad87 a=std::abs(static_cast<Rad87>(x));if(std::isinf(a))return std::numeric_limits<Rad87>::max();return std::max(Rad87(0),a*(1-4*eps87)-tiny87);}
static Rad87 ulp87(Q82 x){auto bits=decode84(x);Rad87 value=std::ldexp(Rad87(1),bits.exponent);return value>0?value:tiny87;}
struct Ball87 {
 Q82 value=0;Rad87 radius=0;
 Ball87()=default;Ball87(Q82 v):value(v){}
 Ball87(Q82 v,Rad87 r):value(v),radius(r){need(finite82(v)&&std::isfinite(r)&&r>=0,"Guard87 ball overflow");}
};
static Rad87 lower87(Ball87 a){return std::nextafter(std::nextafter(static_cast<Rad87>(a.value),-std::numeric_limits<Rad87>::infinity())-a.radius,-std::numeric_limits<Rad87>::infinity());}
static Rad87 upper87(Ball87 a){return std::nextafter(std::nextafter(static_cast<Rad87>(a.value),std::numeric_limits<Rad87>::infinity())+a.radius,std::numeric_limits<Rad87>::infinity());}
static Ball87 operator+(Ball87 a,Ball87 b){Q82 v=a.value+b.value;return {v,inflate87(a.radius+b.radius+ulp87(v))};}
static Ball87 operator-(Ball87 a){return {-a.value,a.radius};}
static Ball87 operator-(Ball87 a,Ball87 b){return a+(-b);}
static Ball87 operator*(Ball87 a,Ball87 b){Q82 v=a.value*b.value;return {v,inflate87(qhi87(a.value)*b.radius+qhi87(b.value)*a.radius+a.radius*b.radius+ulp87(v))};}
static Ball87 operator/(Ball87 a,Ball87 b){
 Rad87 denominator=(qlo87(b.value)-b.radius)*(1-4*eps87)-4*tiny87;need(denominator>0,"Guard87 ball denominator overlaps zero");Q82 v=a.value/b.value;Rad87 rounding=ulp87(v),numerator=inflate87(a.radius+inflate87(qhi87(v)+rounding)*b.radius);return {v,inflate87(numerator/denominator+rounding)};
}
static bool operator<(Ball87 a,Ball87 b){return upper87(a)<lower87(b);}
static bool operator>(Ball87 a,Ball87 b){return b<a;}
static bool operator>=(Ball87 a,Ball87 b){return lower87(a)>=upper87(b);}
static bool finite86(Ball87 a){return finite82(a.value)&&std::isfinite(a.radius);}
static Ball87 sqrt86(Ball87 a){
 need(lower87(a)>0,"Guard87 square root overlaps zero");Q82 v=sqrtq(a.value);need(v>0&&finite82(v),"Guard87 root candidate");Q82 product=v*v,residual=product-a.value;Rad87 numerator=inflate87(a.radius+qhi87(residual)+ulp87(product)+ulp87(residual));return {v,inflate87(numerator/qlo87(v))};
}
struct GuardTrace87 {SharedTrace86 base;double balls=0,certified=0,ambiguous=0,closure_wide=0,ball_failure=0,max_width=0;};
// condition records the evaluated numeric paths; a certified ball decision need
// not evaluate the frozen wide condition estimate. Do not compare this maximum
// as if all implementations evaluated the same arithmetic paths.
struct GuardStep87 {SharedStep86 step;bool bit=false;double midpoint=0,lower=0,upper=0;int kind=0;};
static double lower_double87(Rad87 x){return std::nextafter(static_cast<double>(x),-std::numeric_limits<double>::infinity());}
static double upper_double87(Rad87 x){return std::nextafter(static_cast<double>(x),std::numeric_limits<double>::infinity());}
static GuardStep87 guarded_step87(const ClickModel70& c,const SharedState86& state,int mode,double uniform,int budget,GuardTrace87& trace){
 GuardStep87 result;auto& step=result.step;int k=int(state.clicks.size()),d=state.x.rows();need(k<=7,"Guard87 retained click limit");step.forms.resize(d,2*(k+1));for(int j=0;j<k+1;++j){int port=j<k?state.clicks[j]:mode;step.forms.col(2*j)=c.a.col(port).cast<long double>();step.forms.col(2*j+1)=c.b.col(port).cast<long double>();}step.products=state.x*step.forms;Mat82 gram=step.forms.transpose()*step.products;gram=((gram+gram.transpose())/2).eval();gram.diagonal().array()+=1;step.xa=step.products.col(2*k);step.xb=step.products.col(2*k+1);Q82 b00=gram(2*k,2*k),b11=gram(2*k+1,2*k+1),b01=gram(2*k,2*k+1),det=b00*b11-b01*b01;need(b00>0&&b11>0&&det>0,"Guard87 new-port determinant invalid");step.i00=static_cast<long double>(b11/det);step.i01=static_cast<long double>(-b01/det);step.i11=static_cast<long double>(b00/det);
 bool wider=false;try{step.narrow=std::make_unique<VacuumSum86<Q82>>(vacuum_sum86<Q82>(gram,k));auto& s=*step.narrow;Q82 click=s.mass-s.zero;Q82 smallest=fminq(fabsq(s.mass),fminq(fabsq(s.zero),fabsq(click)));wider=!(s.mass>0&&s.zero>=0&&click>=0&&finite82(s.mass))||smallest==0||s.absolute/smallest>Q82(1e16L);if(!wider){step.p0=Weight83(s.zero/s.mass);step.p1=Weight83(click/s.mass);trace.base.condition=std::max(trace.base.condition,double(s.absolute/smallest));}}catch(const std::exception&){wider=true;}
 bool certified=false,force_wide=wider;Rad87 ball_low=0,ball_high=0;
 if(wider){++trace.balls;try{
  auto sums=vacuum_sum86<Ball87>(gram,k);Ball87 probability=(sums.mass-sums.zero)/sums.mass;need(probability.value>=0&&probability.value<=1,"Guard87 midpoint outside probability range");ball_low=lower87(probability);ball_high=upper87(probability);need(std::isfinite(ball_low)&&std::isfinite(ball_high),"Guard87 probability enclosure overflow");
  if(Rad87(uniform)<ball_low||Rad87(uniform)>ball_high){certified=true;++trace.certified;result.bit=Rad87(uniform)<ball_low;step.p1=Weight83(probability.value);step.p0=Weight83(1)-step.p1;result.kind=1;result.lower=lower_double87(ball_low);result.upper=upper_double87(ball_high);result.midpoint=double83(step.p1);trace.max_width=std::max(trace.max_width,result.upper-result.lower);force_wide=result.bit&&mode+1<c.model->m&&(1U<<(k+1))>unsigned(budget);if(force_wide)++trace.closure_wide;}
  else ++trace.ambiguous;
 }catch(const std::exception&){++trace.ball_failure;}}
 if(force_wide){step.wide=std::make_unique<VacuumSum86<Weight83>>(vacuum_sum86<Weight83>(gram,k));auto& s=*step.wide;Weight83 click=s.mass-s.zero;need(s.mass>Weight83(0)&&s.zero>=Weight83(0)&&click>=Weight83(0)&&finite83(s.mass),"Guard87 wide probability invalid; no clipping");step.p0=s.zero/s.mass;step.p1=click/s.mass;Rad87 small=std::min({std::abs(ld83(s.mass)),std::abs(ld83(s.zero)),std::abs(ld83(click))});if(small>0)trace.base.condition=std::max(trace.base.condition,double(ld83(s.absolute)/small));++trace.base.wide;
  if(certified){need(!(step.p1<Weight83(Q82(ball_low)))&&!(Weight83(Q82(ball_high))<step.p1),"Guard87 closure oracle outside decision interval");need(result.bit==(Weight83(Q82(uniform))<step.p1),"Guard87 closure decision differs from oracle");}
  result.kind=2;
 }
 if(!certified||force_wide){result.bit=Weight83(Q82(uniform))<step.p1;result.midpoint=double83(step.p1);result.lower=std::nextafter(result.midpoint,-std::numeric_limits<double>::infinity());result.upper=std::nextafter(result.midpoint,std::numeric_limits<double>::infinity());}
 trace.base.evaluations+=1<<(k+1);return result;
}
API int tc87_guarded_sample(void* handle,int count,int threads,int budget,const double* draws,std::uint8_t* bits,double* midpoint,double* bounds,std::uint8_t* kind,int* errors,double* stats,char* message,int cap){try{
 SharedRoundGuard86 guard;need(handle&&count>=0&&count<=4096&&threads>=1&&threads<=64&&budget>=1&&budget<=128&&draws&&bits&&midpoint&&bounds&&kind&&errors&&stats,"Guard87 explicit sample arguments");auto& c=*static_cast<ClickModel70*>(handle);int m=c.model->m;std::size_t entries=std::size_t(count)*m;std::fill(bits,bits+entries,0);std::fill(midpoint,midpoint+entries,std::numeric_limits<double>::quiet_NaN());std::fill(bounds,bounds+2*entries,std::numeric_limits<double>::quiet_NaN());std::fill(kind,kind+entries,255);std::fill(errors,errors+2*count,0);std::fill(stats,stats+12*count,0);std::atomic<int> next{0};std::vector<std::string> messages(count);std::vector<std::thread> pool;
 auto worker=[&](){SharedRoundGuard86 precision;for(;;){int row=next.fetch_add(1);if(row>=count)break;int mode=0,stage=1;GuardTrace87 trace;try{SharedState86 state{c.model->initial.cast<long double>(),{}};for(;mode<m;++mode){stage=1;auto at=std::size_t(row)*m+mode;double u=draws[at];need(std::isfinite(u)&&u>=0&&u<1,"Guard87 explicit uniform");auto step=guarded_step87(c,state,mode,u,budget,trace);bits[at]=step.bit;midpoint[at]=step.midpoint;bounds[2*at]=step.lower;bounds[2*at+1]=step.upper;kind[at]=step.kind;if(mode+1<m){stage=2;shared_update86(state,step.step,mode,step.bit,budget,trace.base);}}}catch(const std::exception& e){errors[2*row]=stage;errors[2*row+1]=mode;messages[row]=e.what();}auto* s=stats+12*row;s[0]=trace.base.evaluations;s[1]=trace.base.merges;s[2]=trace.base.peak;s[3]=trace.base.condition;s[4]=trace.base.wide;s[5]=mode;s[6]=trace.balls;s[7]=trace.certified;s[8]=trace.ambiguous;s[9]=trace.closure_wide;s[10]=trace.ball_failure;s[11]=trace.max_width;}};
 try{for(int i=0;i<std::min(count,threads);++i)pool.emplace_back(worker);}catch(...){for(auto& t:pool)t.join();throw;}for(auto& t:pool)t.join();for(int row=0;row<count;++row)if(errors[2*row]){error(message,cap,"row="+std::to_string(row)+" mode="+std::to_string(errors[2*row+1])+" "+messages[row]);return 1;}return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
API int ball87_probe(int op,const char* a,const char* ar,const char* b,const char* br,char* midpoint,char* radius,int size,char* message,int cap){try{
 SharedRoundGuard86 guard;need(op>=0&&op<=4&&a&&ar&&b&&br&&midpoint&&radius&&size>=128,"Guard87 scalar probe arguments");auto number=[](const char* p){char* end;Q82 q=strtoflt128(p,&end);need(end&&!*end&&finite82(q),"Guard87 quad input");return q;};auto uncertainty=[](const char* p){char* end;Rad87 r=strtold(p,&end);need(end&&!*end&&r>=0&&std::isfinite(r),"Guard87 radius input");return r;};Ball87 x(number(a),uncertainty(ar)),y(number(b),uncertainty(br)),z;
 if(op==0)z=x+y;else if(op==1)z=x-y;else if(op==2)z=x*y;else if(op==3)z=x/y;else z=sqrt86(x);int nm=quadmath_snprintf(midpoint,size,"%.85Qg",z.value),nr=std::snprintf(radius,size,"%.85Le",z.radius);need(nm>0&&nr>0&&nm<size&&nr<size,"Guard87 short scalar output");return 0;
 }catch(const std::exception& e){error(message,cap,e.what());return 1;}}
