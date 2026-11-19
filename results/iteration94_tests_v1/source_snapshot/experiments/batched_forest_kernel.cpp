/* ABI and scientific scope are documented in batched_forest_kernel.h.
 * This implements the standard Gaussian Schur/IE identities in
 * research/iteration7_batched_event_kernel.md.  It is not a new sampling law.
 * Error indicators are deliberately propagated with absolute dependencies;
 * they are NOT directed-rounding or verified-library interval bounds.
 */
#include "batched_forest_kernel.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <new>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <quadmath.h>

namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now()-start).count();
}
struct NumericError : std::runtime_error { using std::runtime_error::runtime_error; };

template<class T> struct Format;
template<> struct Format<long double> {
    static long double abs(long double x) { return fabsl(x); }
    static long double sqrt(long double x) { return sqrtl(x); }
    static bool finite(long double x) { return std::isfinite(x); }
    static bool subnormal(long double x) { return std::fpclassify(x)==FP_SUBNORMAL; }
    static long double eps() { return std::numeric_limits<long double>::epsilon(); }
};
template<> struct Format<__float128> {
    static __float128 abs(__float128 x) { return fabsq(x); }
    static __float128 sqrt(__float128 x) { return sqrtq(x); }
    static bool finite(__float128 x) { return !isnanq(x) && !isinfq(x); }
    static bool subnormal(__float128 x) { return x!=0 && fabsq(x)<ldexpq(1,-16382); }
    static __float128 eps() { return ldexpq(1,-112); }
};
template<class T> struct Number { T value=0, error=0; };

/* Each tracked operation includes its own fixed mu=8*format_epsilon term.
 * The error arithmetic is checked but is not itself interval arithmetic.
 * A nonzero product/division or indicator that underflows is unresolved.
 */
template<class T> class Arithmetic {
    uint64_t *counts;
    T mu;
    static T finite(T x) {
        if (!Format<T>::finite(x)) throw NumericError("nonfinite native arithmetic or indicator");
        if (Format<T>::subnormal(x)) throw NumericError("subnormal native arithmetic or indicator");
        return x;
    }
    static T product(T a,T b) {
        T v=finite(a*b);
        if (a!=0 && b!=0 && v==0) throw NumericError("positive native magnitude underflowed");
        return v;
    }
    static T quotient(T a,T b) {
        if (b==0) throw NumericError("unresolved native zero divisor");
        T v=finite(a/b);
        if (a!=0 && v==0) throw NumericError("positive native quotient magnitude underflowed");
        return v;
    }
    static Number<T> checked(T v,T e) {
        finite(v); finite(e);
        if (e<0) throw NumericError("negative native error indicator");
        return {v,e};
    }
public:
    explicit Arithmetic(uint64_t *c):counts(c),mu(T(8)*Format<T>::eps()) {}
    Number<T> literal(T v) { return checked(v,0); }
    Number<T> add(Number<T> a,Number<T> b) {
        ++counts[BF_TRACKED_ADD_SUB];
        T v=finite(a.value+b.value);
        T e=finite(a.error+b.error+product(mu,finite(Format<T>::abs(a.value)+Format<T>::abs(b.value)+Format<T>::abs(v))));
        return checked(v,e);
    }
    Number<T> sub(Number<T> a,Number<T> b) {
        b.value=-b.value;
        return add(a,b);
    }
    Number<T> mul(Number<T> a,Number<T> b) {
        ++counts[BF_TRACKED_MULTIPLY];
        T v=product(a.value,b.value);
        T e=finite(product(Format<T>::abs(a.value),b.error)
            +product(Format<T>::abs(b.value),a.error)+product(a.error,b.error)
            +product(mu,Format<T>::abs(v)));
        return checked(v,e);
    }
    Number<T> div(Number<T> a,Number<T> b) {
        ++counts[BF_TRACKED_DIVIDE];
        T lower=finite(Format<T>::abs(b.value)-b.error);
        if (!(lower>0)) throw NumericError("native denominator unresolved within indicator");
        T v=quotient(a.value,b.value);
        T e=finite(quotient(finite(a.error+product(Format<T>::abs(v),b.error)),lower)
                   +product(mu,Format<T>::abs(v)));
        return checked(v,e);
    }
    Number<T> sqrt(Number<T> a) {
        ++counts[BF_TRACKED_SQRT];
        T lower=finite(a.value-a.error);
        if (!(a.value>0) || !(lower>0)) throw NumericError("native Cholesky pivot unresolved within indicator");
        T v=finite(Format<T>::sqrt(a.value));
        T e=finite(quotient(a.error,finite(Format<T>::sqrt(lower)+v))+product(mu,Format<T>::abs(v)));
        return checked(v,e);
    }
};

template<class T> void cholesky(Number<T> *matrix,int n,Arithmetic<T>& a) {
    for (int row=0;row<n;++row) {
        for (int col=0;col<=row;++col) {
            auto value=matrix[row*n+col];
            for (int k=0;k<col;++k) value=a.sub(value,a.mul(matrix[row*n+k],matrix[col*n+k]));
            matrix[row*n+col]=(row==col) ? a.sqrt(value) : a.div(value,matrix[col*n+col]);
        }
    }
}
template<class T> void triangular_solve(const Number<T> *lower,Number<T> *rhs,
                                       int n,int columns,Arithmetic<T>& a) {
    for (int row=0;row<n;++row) for (int col=0;col<columns;++col) {
        auto value=rhs[row*columns+col];
        for (int k=0;k<row;++k) value=a.sub(value,a.mul(lower[row*n+k],rhs[k*columns+col]));
        rhs[row*columns+col]=a.div(value,lower[row*n+row]);
    }
}
template<class T> Number<T> reciprocal_diagonal_product(const Number<T> *lower,int n,Arithmetic<T>& a) {
    auto result=a.literal(1);
    for (int j=0;j<n;++j) result=a.div(result,lower[j*n+j]);
    return result;
}

template<class T> struct State {
    bool valid=false;
    std::vector<Number<T>> matrix;
    std::vector<int> modes,positions,zeros;
    uint64_t bytes() const {
        return matrix.capacity()*sizeof(Number<T>)+(modes.capacity()+positions.capacity()+zeros.capacity())*sizeof(int);
    }
};
struct QueryData {
    int mode;
    std::vector<int> zeros,retained,omitted,edges;
    int f() const { return static_cast<int>(omitted.size()); }
    int e() const { return static_cast<int>(edges.size()/2); }
    int group() const { return 1+f()+e(); }
    int count() const { return 1+2*group(); }
};
void error_text(char *error,size_t capacity,const std::string& value) {
    if (error && capacity) {
        const auto n=std::min(capacity-1,value.size());
        std::memcpy(error,value.data(),n); error[n]='\0';
    }
}
template<class F> int protected_call(char *error,size_t cap,F&& function) {
    error_text(error,cap,"");
    try { function(); return BF_OK; }
    catch (const NumericError& e) { error_text(error,cap,e.what()); return BF_NUMERICAL_ERROR; }
    catch (const std::invalid_argument& e) { error_text(error,cap,e.what()); return BF_INVALID_ARGUMENT; }
    catch (const std::bad_alloc& e) { error_text(error,cap,e.what()); return BF_RESOURCE_ERROR; }
    catch (const std::exception& e) { error_text(error,cap,e.what()); return BF_INTERNAL_ERROR; }
    catch (...) { error_text(error,cap,"unknown native program error"); return BF_INTERNAL_ERROR; }
}
} // namespace

struct BFHandle {
    int m;
    std::vector<double> original;
    std::vector<int> order,order_position;
    std::vector<uint8_t> vacuum;
    State<long double> ld;
    State<__float128> quad;
    BFDiagnostics stats{};
    void memory(int precision,uint64_t workspace=0) {
        stats.state_payload_bytes[0]=ld.bytes();
        stats.state_payload_bytes[1]=quad.bytes();
        for (int p=0;p<2;++p) stats.peak_state_payload_bytes[p]=
            std::max(stats.peak_state_payload_bytes[p],stats.state_payload_bytes[p]);
        stats.peak_workspace_payload_bytes[precision]=std::max(stats.peak_workspace_payload_bytes[precision],workspace);
        const uint64_t payload=stats.original_input_bytes+stats.state_payload_bytes[0]
            +stats.state_payload_bytes[1]+workspace;
        stats.peak_kernel_payload_bytes=std::max(stats.peak_kernel_payload_bytes,payload);
    }
};

namespace {
QueryData validate_query(BFHandle *h,const BFQuery *query,BFOutput *output) {
    if (!h || !query || !output) throw std::invalid_argument("null native query argument");
    const auto& q=*query;
    if (q.mode<0 || q.mode>=h->m || q.zero_count<0 || q.retained_count<0
        || q.omitted_count<0 || q.edge_count<0 || q.zero_count>h->m
        || q.retained_count>BF_MAX_RETAINED || q.omitted_count>h->m
        || q.edge_count>std::max(0,q.omitted_count-1))
        throw std::invalid_argument("invalid native mode, event size or retained budget");
    if ((q.zero_count && !q.zeros) || (q.retained_count && !q.retained)
        || (q.omitted_count && !q.omitted) || (q.edge_count && !q.edges))
        throw std::invalid_argument("missing native query index buffer");
    QueryData result; result.mode=q.mode;
    if (q.zero_count) result.zeros.assign(q.zeros,q.zeros+q.zero_count);
    if (q.retained_count) result.retained.assign(q.retained,q.retained+q.retained_count);
    if (q.omitted_count) result.omitted.assign(q.omitted,q.omitted+q.omitted_count);
    if (q.edge_count) result.edges.assign(q.edges,q.edges+2*q.edge_count);
    std::vector<uint8_t> used(h->m,0); used[q.mode]=1;
    for (const auto *list:{&result.zeros,&result.retained,&result.omitted}) for (int mode:*list) {
        if (mode<0 || mode>=h->m || used[mode]) throw std::invalid_argument("native event sets must be distinct and disjoint");
        used[mode]=1;
    }
    for (size_t j=1;j<result.zeros.size();++j)
        if (h->order_position[result.zeros[j-1]]>=h->order_position[result.zeros[j]])
            throw std::invalid_argument("zero modes must follow the fixed measurement order");
    // Only enumeration order is canonicalized; the selected R set is unchanged.
    std::sort(result.retained.begin(),result.retained.end());
    std::vector<int> local(h->m,-1),parent(result.f());
    std::iota(parent.begin(),parent.end(),0);
    for (int j=0;j<result.f();++j) local[result.omitted[j]]=j;
    auto find=[&](int vertex) { while (parent[vertex]!=vertex) vertex=parent[vertex]; return vertex; };
    for (int j=0;j<result.e();++j) {
        int u=result.edges[2*j],v=result.edges[2*j+1];
        if (u<0 || v<0 || u>=h->m || v>=h->m || local[u]<0 || local[v]<0)
            throw std::invalid_argument("forest endpoints must belong to omitted clicks");
        u=find(local[u]); v=find(local[v]);
        if (u==v) throw std::invalid_argument("native edge set must be an acyclic forest");
        parent[u]=v;
    }
    if (!output->values || !output->indicators || !output->exact_zero
        || output->capacity<static_cast<size_t>(result.count()))
        throw std::invalid_argument("native event output buffer is too small");
    return result;
}

template<class T> void rebuild(BFHandle *h,State<T>& state,int precision,Arithmetic<T>& a) {
    state.valid=false;
    const int n=2*h->m;
    state.matrix.resize(static_cast<size_t>(n)*n);
    state.modes.resize(h->m); std::iota(state.modes.begin(),state.modes.end(),0);
    state.positions.resize(h->m); std::iota(state.positions.begin(),state.positions.end(),0);
    state.zeros.clear();
    for (int row=0;row<n;++row) for (int col=0;col<n;++col) {
        int x=row/2+(row%2)*h->m,y=col/2+(col%2)*h->m;
        auto input=a.literal(T(h->original[x*n+y]));
        state.matrix[row*n+col]=a.div(a.add(input,a.literal(row==col ? 1 : 0)),a.literal(2));
    }
    state.valid=true;
    ++h->stats.counters[precision][BF_STATE_REBUILDS];
    h->memory(precision);
}

template<class T> void project_zero(BFHandle *h,State<T>& state,int mode,int precision,Arithmetic<T>& a) {
    int n=static_cast<int>(state.modes.size()),N=2*n,position=state.positions[mode];
    if (position<0) throw std::logic_error("zero state attempted a duplicate projection");
    std::array<Number<T>,4> lower;
    for (int i=0;i<2;++i) for (int j=0;j<2;++j) lower[2*i+j]=state.matrix[(2*position+i)*N+2*position+j];
    ++h->stats.counters[precision][BF_ZERO_FACTORIZATIONS];
    cholesky(lower.data(),2,a);
    std::vector<int> keep; keep.reserve(n-1);
    for (int j=0;j<n;++j) if (j!=position) keep.push_back(j);
    const int columns=2*(n-1);
    std::vector<Number<T>> rhs(2*columns),next(static_cast<size_t>(columns)*columns);
    for (int i=0;i<2;++i) for (int j=0;j<columns;++j)
        rhs[i*columns+j]=state.matrix[(2*position+i)*N+2*keep[j/2]+j%2];
    h->memory(precision,rhs.capacity()*sizeof(Number<T>)+next.capacity()*sizeof(Number<T>)
                        +keep.capacity()*sizeof(int)+sizeof(lower));
    triangular_solve(lower.data(),rhs.data(),2,columns,a);
    h->stats.counters[precision][BF_ZERO_SOLVE_COLUMNS]+=columns;
    for (int row=0;row<columns;++row) for (int col=row;col<columns;++col) {
        auto value=state.matrix[(2*keep[row/2]+row%2)*N+2*keep[col/2]+col%2];
        for (int k=0;k<2;++k) value=a.sub(value,a.mul(rhs[k*columns+row],rhs[k*columns+col]));
        next[row*columns+col]=next[col*columns+row]=value;
    }
    h->stats.counters[precision][BF_ZERO_SCHUR_BLOCKS]+=uint64_t(n-1)*n/2;
    state.matrix.swap(next);
    state.modes.erase(state.modes.begin()+position);
    std::fill(state.positions.begin(),state.positions.end(),-1);
    for (size_t j=0;j<state.modes.size();++j) state.positions[state.modes[j]]=static_cast<int>(j);
    state.zeros.push_back(mode);
    ++h->stats.counters[precision][BF_ZERO_UPDATES];
    h->memory(precision);
}

template<class T> void prepare_state(BFHandle *h,State<T>& state,const QueryData& q,
                                     int precision,bool force,Arithmetic<T>& a) {
    auto started=Clock::now();
    try {
        bool prefix=state.valid && state.zeros.size()<=q.zeros.size()
            && std::equal(state.zeros.begin(),state.zeros.end(),q.zeros.begin());
        if (force || !prefix) rebuild(h,state,precision,a);
        else if (state.zeros.size()==q.zeros.size()) ++h->stats.counters[precision][BF_STATE_REUSES];
        else ++h->stats.counters[precision][BF_STATE_APPENDS];
        while (state.zeros.size()<q.zeros.size())
            project_zero(h,state,q.zeros[state.zeros.size()],precision,a);
    } catch (...) {
        state.valid=false;
        ++h->stats.counters[precision][BF_STATE_INVALIDATIONS];
        h->stats.seconds[precision][BF_STATE_SECONDS]+=elapsed(started);
        throw;
    }
    h->stats.seconds[precision][BF_STATE_SECONDS]+=elapsed(started);
}

template<class T> using Block=std::array<Number<T>,4>;
template<class T> void insert_block(Number<T> *matrix,int n,int row,int col,const Block<T>& block,bool transpose=false) {
    for (int i=0;i<2;++i) for (int j=0;j<2;++j)
        matrix[(2*row+i)*n+2*col+j]=block[transpose ? 2*j+i : 2*i+j];
}
template<class T> Number<T> small_vacuum(std::array<Number<T>,36>& matrix,int n,
                                        Arithmetic<T>& a,uint64_t *counts) {
    ++counts[BF_LOCAL_DETERMINANTS];
    counts[BF_LOCAL_DIMENSION_SUM]+=n;
    counts[BF_LOCAL_CUBIC_WORK]+=uint64_t(n)*n*n;
    cholesky(matrix.data(),n,a);
    return reciprocal_diagonal_product(matrix.data(),n,a);
}
template<class T> Number<T> vacuum1(const Block<T>& diagonal,Arithmetic<T>& a,uint64_t *counts) {
    std::array<Number<T>,36> matrix{};
    insert_block(matrix.data(),2,0,0,diagonal);
    return small_vacuum(matrix,2,a,counts);
}
template<class T> Number<T> vacuum2(const Block<T>& first,const Block<T>& second,const Block<T>& cross,
                                   Arithmetic<T>& a,uint64_t *counts) {
    std::array<Number<T>,36> matrix{};
    insert_block(matrix.data(),4,0,0,first); insert_block(matrix.data(),4,1,1,second);
    insert_block(matrix.data(),4,0,1,cross); insert_block(matrix.data(),4,1,0,cross,true);
    return small_vacuum(matrix,4,a,counts);
}
template<class T> Number<T> vacuum3(const Block<T>& first,const Block<T>& second,const Block<T>& third,
    const Block<T>& ab,const Block<T>& ac,const Block<T>& bc,Arithmetic<T>& a,uint64_t *counts) {
    std::array<Number<T>,36> matrix{};
    insert_block(matrix.data(),6,0,0,first); insert_block(matrix.data(),6,1,1,second);
    insert_block(matrix.data(),6,2,2,third);
    insert_block(matrix.data(),6,0,1,ab); insert_block(matrix.data(),6,1,0,ab,true);
    insert_block(matrix.data(),6,0,2,ac); insert_block(matrix.data(),6,2,0,ac,true);
    insert_block(matrix.data(),6,1,2,bc); insert_block(matrix.data(),6,2,1,bc,true);
    return small_vacuum(matrix,6,a,counts);
}

std::vector<uint8_t> zero_certificates(BFHandle *h,const QueryData& q) {
    std::vector<uint8_t> flags(q.count(),0);
    bool baseline_zero=std::any_of(q.retained.begin(),q.retained.end(),[&](int j){ return h->vacuum[j]; });
    flags[0]=baseline_zero;
    for (int outcome=0;outcome<2;++outcome) {
        int offset=1+outcome*q.group();
        bool branch_zero=baseline_zero || (outcome==1 && h->vacuum[q.mode]);
        flags[offset]=branch_zero;
        for (int j=0;j<q.f();++j) flags[offset+1+j]=branch_zero || h->vacuum[q.omitted[j]];
        for (int j=0;j<q.e();++j)
            flags[offset+1+q.f()+j]=branch_zero || h->vacuum[q.edges[2*j]] || h->vacuum[q.edges[2*j+1]];
    }
    return flags;
}

template<class T> std::vector<Number<T>> events(BFHandle *h,State<T>& state,const QueryData& q,
                                              int precision,Arithmetic<T>& a) {
    auto started=Clock::now();
    uint64_t *counts=h->stats.counters[precision];
    const int r=static_cast<int>(q.retained.size()),f=q.f(),e=q.e(),d=f+1,N=2*static_cast<int>(state.modes.size());
    std::vector<Number<T>> sums(q.count());
    std::vector<int> D(1,q.mode); D.insert(D.end(),q.omitted.begin(),q.omitted.end());
    std::vector<int> f_position(h->m,-1);
    for (int j=0;j<f;++j) f_position[q.omitted[j]]=j+1;
    std::vector<int> subset; subset.reserve(r);
    std::vector<Number<T>> lower,columns;
    lower.reserve(4*r*r); columns.reserve(4*r*d);
    std::vector<Block<T>> diagonals(d),stars(f),edge_blocks(e);
    std::vector<Number<T>> node_vac(f),star_vac(f);
    uint64_t workspace=sums.capacity()*sizeof(Number<T>)+(D.capacity()+f_position.capacity()+subset.capacity())*sizeof(int)
        +(lower.capacity()+columns.capacity()+node_vac.capacity()+star_vac.capacity())*sizeof(Number<T>)
        +(diagonals.capacity()+stars.capacity()+edge_blocks.capacity())*sizeof(Block<T>)+36*sizeof(Number<T>);
    h->memory(precision,workspace);
    try {
        for (uint64_t mask=0;mask<(uint64_t(1)<<r);++mask) {
            ++counts[BF_SUBSETS]; subset.clear();
            for (int j=0;j<r;++j) if (mask&(uint64_t(1)<<j)) subset.push_back(q.retained[j]);
            const int s=static_cast<int>(subset.size()),n=2*s;
            auto beta=a.literal(1);
            lower.resize(n*n); columns.resize(n*2*d);
            for (int row=0;row<n;++row) for (int col=0;col<n;++col)
                lower[row*n+col]=state.matrix[(2*state.positions[subset[row/2]]+row%2)*N
                    +2*state.positions[subset[col/2]]+col%2];
            if (n) {
                ++counts[BF_SUBSET_FACTORIZATIONS]; counts[BF_SUBSET_DIMENSION_SUM]+=n;
                counts[BF_SUBSET_CUBIC_WORK]+=uint64_t(n)*n*n;
                cholesky(lower.data(),n,a);
                beta=reciprocal_diagonal_product(lower.data(),n,a);
                for (int row=0;row<n;++row) for (int col=0;col<2*d;++col)
                    columns[row*2*d+col]=state.matrix[(2*state.positions[subset[row/2]]+row%2)*N
                        +2*state.positions[D[col/2]]+col%2];
                triangular_solve(lower.data(),columns.data(),n,2*d,a);
                counts[BF_SHARED_SOLVE_COLUMNS]+=2*d;
                counts[BF_SHARED_SOLVE_DOT_TERMS]+=uint64_t(n)*(n-1)/2*(2*d);
            }
            auto block=[&](int u,int v) {
                ++counts[BF_SPARSE_SCHUR_BLOCKS]; counts[BF_SPARSE_SCHUR_DOT_TERMS]+=4*n;
                Block<T> result;
                for (int row=0;row<2;++row) for (int col=0;col<2;++col) {
                    auto value=state.matrix[(2*state.positions[D[u]]+row)*N+2*state.positions[D[v]]+col];
                    for (int k=0;k<n;++k) value=a.sub(value,a.mul(columns[k*2*d+2*u+row],columns[k*2*d+2*v+col]));
                    result[row*2+col]=value;
                }
                return result;
            };
            for (int j=0;j<d;++j) diagonals[j]=block(j,j);
            for (int j=0;j<f;++j) stars[j]=block(0,j+1);
            for (int j=0;j<e;++j) edge_blocks[j]=block(f_position[q.edges[2*j]],f_position[q.edges[2*j+1]]);
            const auto one=a.literal(1);
            auto ni=vacuum1(diagonals[0],a,counts);
            auto accumulate=[&](int index,Number<T> factor) {
                auto term=a.mul(beta,factor);
                sums[index]=(s&1) ? a.sub(sums[index],term) : a.add(sums[index],term);
                ++counts[BF_EVENT_ACCUMULATIONS];
            };
            accumulate(0,one);
            accumulate(1,ni); accumulate(1+q.group(),a.sub(one,ni));
            for (int j=0;j<f;++j) {
                node_vac[j]=vacuum1(diagonals[j+1],a,counts);
                star_vac[j]=vacuum2(diagonals[0],diagonals[j+1],stars[j],a,counts);
                auto h0=a.sub(ni,star_vac[j]);
                auto h1=a.add(a.sub(a.sub(one,ni),node_vac[j]),star_vac[j]);
                accumulate(2+j,h0); accumulate(2+q.group()+j,h1);
            }
            for (int j=0;j<e;++j) {
                int u=f_position[q.edges[2*j]],v=f_position[q.edges[2*j+1]];
                auto uv=vacuum2(diagonals[u],diagonals[v],edge_blocks[j],a,counts);
                auto iuv=vacuum3(diagonals[0],diagonals[u],diagonals[v],stars[u-1],stars[v-1],edge_blocks[j],a,counts);
                auto h0=a.add(a.sub(a.sub(ni,star_vac[u-1]),star_vac[v-1]),iuv);
                auto h1=a.sub(a.add(a.add(a.add(a.sub(a.sub(a.sub(one,ni),node_vac[u-1]),node_vac[v-1]),
                    star_vac[u-1]),star_vac[v-1]),uv),iuv);
                accumulate(2+f+j,h0); accumulate(2+q.group()+f+j,h1);
            }
        }
        // Empty retained event is the exact conditional identity P(empty|Z)=1.
        if (r==0) sums[0]=a.literal(1);
    } catch (...) { h->stats.seconds[precision][BF_EVENT_SECONDS]+=elapsed(started); throw; }
    h->stats.seconds[precision][BF_EVENT_SECONDS]+=elapsed(started);
    return sums;
}

template<class T> void convert_output(BFHandle *h,const std::vector<Number<T>>& values,
    const std::vector<uint8_t>& flags,int precision,BFOutput *output) {
    auto started=Clock::now();
    try {
        for (size_t j=0;j<values.size();++j) {
            ++h->stats.counters[precision][BF_OUTPUT_CONVERSIONS];
            output->exact_zero[j]=flags[j];
            if (flags[j]) {
                output->values[j]=output->indicators[j]=0;
                ++h->stats.counters[precision][BF_ANALYTIC_ZERO_EVENTS]; continue;
            }
            const auto& value=values[j];
            long double stored=static_cast<long double>(value.value);
            if (!std::isfinite(stored) || std::fpclassify(stored)==FP_SUBNORMAL
                || (value.value!=0 && stored==0))
                throw NumericError("native event loses finite positive magnitude in long-double output");
            T error=value.error;
            if (precision==BF_QUAD) {
                T conversion=Format<T>::abs(T(stored)-value.value);
                T rounding=T(8)*T(std::numeric_limits<long double>::epsilon())*Format<T>::abs(T(stored));
                if (stored!=0 && rounding==0) throw NumericError("quad output conversion indicator underflowed");
                error+=conversion+rounding;
            }
            if (!Format<T>::finite(error) || error<0) throw NumericError("invalid converted native indicator");
            long double indicator=static_cast<long double>(error);
            if (T(indicator)<error) indicator=nextafterl(indicator,std::numeric_limits<long double>::infinity());
            if (!std::isfinite(indicator) || std::fpclassify(indicator)==FP_SUBNORMAL
                || (error>0 && indicator==0)) throw NumericError("native indicator lost in long-double output");
            output->values[j]=stored; output->indicators[j]=indicator;
        }
    } catch (...) { h->stats.seconds[precision][BF_OUTPUT_SECONDS]+=elapsed(started); throw; }
    h->stats.seconds[precision][BF_OUTPUT_SECONDS]+=elapsed(started);
}

template<class T> void query_in_precision(BFHandle *h,State<T>& state,const QueryData& q,
    int precision,bool force,BFOutput *output) {
    const auto started=Clock::now(); uint64_t *counts=h->stats.counters[precision];
    ++counts[BF_QUERIES];
    counts[BF_MAXIMUM_RETAINED]=std::max(counts[BF_MAXIMUM_RETAINED],uint64_t(q.retained.size()));
    counts[BF_MAXIMUM_OMITTED]=std::max(counts[BF_MAXIMUM_OMITTED],uint64_t(q.f()));
    counts[BF_MAXIMUM_ZEROS]=std::max(counts[BF_MAXIMUM_ZEROS],uint64_t(q.zeros.size()));
    try {
        auto flags=zero_certificates(h,q);
        std::vector<Number<T>> values(q.count());
        Arithmetic<T> a(counts);
        if (!flags[0]) {
            prepare_state(h,state,q,precision,force,a);
            values=events(h,state,q,precision,a);
        }
        convert_output(h,values,flags,precision,output);
        ++counts[BF_QUERY_SUCCESSES];
    } catch (const NumericError&) {
        ++counts[BF_NUMERIC_FAILURES]; h->stats.seconds[precision][BF_QUERY_SECONDS]+=elapsed(started); throw;
    } catch (...) { h->stats.seconds[precision][BF_QUERY_SECONDS]+=elapsed(started); throw; }
    h->stats.seconds[precision][BF_QUERY_SECONDS]+=elapsed(started);
}
void raw_query(BFHandle *h,const QueryData& q,int precision,bool force,BFOutput *output) {
    if (precision==BF_LONG_DOUBLE) query_in_precision(h,h->ld,q,precision,force,output);
    else if (precision==BF_QUAD) query_in_precision(h,h->quad,q,precision,force,output);
    else throw std::invalid_argument("raw query requires long-double or quad precision");
}

void score(const QueryData& q,const BFOutput& out,long double guard,BFConditional& result) {
    auto check=[&](int index) {
        long double value=out.values[index],error=out.indicators[index];
        if (!std::isfinite(value) || !std::isfinite(error) || error<0 || value<0 || value>1)
            throw NumericError("invalid event mass or roundoff indicator");
        if (out.exact_zero[index] && (value!=0 || error!=0)) throw NumericError("inconsistent analytic-zero certificate");
        if (value==0 && !out.exact_zero[index]) throw NumericError("uncertified numerical zero for native event");
        if (value>0) {
            long double relative=error/value;
            if (!std::isfinite(relative) || relative>=1 || relative>guard) {
                std::ostringstream message; message<<"relative event indicator "<<std::setprecision(3)<<relative<<" exceeds guard "<<guard;
                throw NumericError(message.str());
            }
            result.maximum_event_relative_indicator=std::max(result.maximum_event_relative_indicator,relative);
        }
    };
    auto nested=[&](int numerator,int denominator,const char *label) {
        if (out.values[numerator]>out.values[denominator]) throw NumericError(std::string(label)+" violates event nesting; no clipping applied");
    };
    check(0);
    if (out.values[0]==0) throw NumericError("baseline conditioning event has zero support");
    int branch[2]={1,1+q.group()};
    for (int b:branch) { check(b); nested(b,0,"baseline branch"); }
    const long double eps=std::numeric_limits<long double>::epsilon();
    long double consistency=fabsl(out.values[branch[0]]+out.values[branch[1]]-out.values[0]);
    long double allowance=out.indicators[0]+out.indicators[branch[0]]+out.indicators[branch[1]]+8*eps*out.values[0];
    if (consistency>allowance) throw NumericError("branch masses do not sum to baseline event mass");
    std::vector<int> degree(q.f(),0),position;
    int maxmode=q.mode;
    for (int mode:q.omitted) maxmode=std::max(maxmode,mode);
    position.assign(maxmode+1,-1);
    for (int j=0;j<q.f();++j) position[q.omitted[j]]=j;
    for (int j=0;j<q.e();++j) { ++degree[position[q.edges[2*j]]]; ++degree[position[q.edges[2*j+1]]]; }
    result.maximum_log_score_indicator=0;
    for (int outcome=0;outcome<2;++outcome) {
        result.zero_score_reason[outcome]=BF_SCORE_POSITIVE;
        result.log_scores[outcome]=result.log_indicators[outcome]=0;
        int b=branch[outcome];
        if (out.values[b]==0) { result.zero_score_reason[outcome]=BF_ZERO_BRANCH; continue; }
        for (int j=0;j<q.f();++j) {
            int index=b+1+j; check(index); nested(index,b,"forest singleton");
            if (out.values[index]==0) { result.zero_score_reason[outcome]=BF_ZERO_NODE; break; }
        }
        if (result.zero_score_reason[outcome]) continue;
        for (int j=0;j<q.e();++j) {
            int index=b+1+q.f()+j; check(index);
            nested(index,b+1+position[q.edges[2*j]],"forest edge at first node");
            nested(index,b+1+position[q.edges[2*j+1]],"forest edge at second node");
            if (out.values[index]==0) { result.zero_score_reason[outcome]=BF_ZERO_EDGE; break; }
        }
        if (result.zero_score_reason[outcome]) continue;
        long double logscore=0,indicator=0,absolute_sum=0; int terms=0;
        auto term=[&](int coefficient,int index) {
            if (!coefficient) return;
            long double value=coefficient*logl(out.values[index]);
            logscore+=value; absolute_sum+=fabsl(value);
            indicator+=std::abs(coefficient)*(-log1pl(-out.indicators[index]/out.values[index]));
            ++terms;
        };
        term(1-q.f()+q.e(),b);
        for (int j=0;j<q.f();++j) term(1-degree[j],b+1+j);
        for (int j=0;j<q.e();++j) term(1,b+1+q.f()+j);
        indicator+=16*eps*(1+fabsl(logscore)+absolute_sum+terms);
        if (!std::isfinite(logscore) || !std::isfinite(indicator) || indicator>guard)
            throw NumericError("forest log-score roundoff guard failed");
        result.log_scores[outcome]=logscore; result.log_indicators[outcome]=indicator;
        result.maximum_log_score_indicator=std::max(result.maximum_log_score_indicator,indicator);
    }
    if (result.zero_score_reason[0] && result.zero_score_reason[1]) throw NumericError("both forest scores have zero support");
    if (result.zero_score_reason[0]) result.probability=1;
    else if (result.zero_score_reason[1]) result.probability=0;
    else {
        long double difference=result.log_scores[0]-result.log_scores[1];
        if (!std::isfinite(difference)) throw NumericError("nonfinite forest log-score difference");
        long double small=expl(difference>=0 ? -difference : difference);
        result.probability=static_cast<double>(difference>=0 ? small/(1+small) : 1/(1+small));
        if (!std::isfinite(result.probability) || !(result.probability>0 && result.probability<1))
            throw NumericError("strictly positive forest branches rounded to a Bernoulli endpoint");
    }
}
} // namespace

extern "C" int bf_formats(BFFormats *result) {
    if (!result) return BF_INVALID_ARGUMENT;
    *result={BF_ABI_VERSION,static_cast<int>(sizeof(long double)),std::numeric_limits<long double>::digits,
        static_cast<int>(sizeof(__float128)),113,BF_MAX_MODES,BF_MAX_RETAINED,
        sizeof(BFQuery),sizeof(BFOutput),sizeof(BFConditional),sizeof(BFDiagnostics)};
    return BF_OK;
}
extern "C" BFHandle *bf_create(int modes,const double *covariance,const int *order,char *error,size_t cap) {
    BFHandle *handle=nullptr;
    int status=protected_call(error,cap,[&] {
        const auto started=Clock::now();
        if (modes<1 || modes>BF_MAX_MODES || !covariance || !order) throw std::invalid_argument("invalid native constructor arguments");
        auto candidate=new BFHandle;
        try {
            candidate->m=modes; const int n=2*modes;
            candidate->original.assign(covariance,covariance+static_cast<size_t>(n)*n);
            candidate->order.assign(order,order+modes); candidate->order_position.assign(modes,-1);
            for (int j=0;j<modes;++j) {
                int mode=order[j];
                if (mode<0 || mode>=modes || candidate->order_position[mode]>=0) throw std::invalid_argument("measurement order must be a permutation");
                candidate->order_position[mode]=j;
            }
            for (int row=0;row<n;++row) for (int col=0;col<n;++col)
                if (!std::isfinite(covariance[row*n+col]) || covariance[row*n+col]!=covariance[col*n+row])
                    throw std::invalid_argument("native covariance must be finite and exactly symmetric binary64");
            candidate->vacuum.assign(modes,1);
            for (int j=0;j<modes;++j) for (int coordinate:{j,j+modes}) for (int col=0;col<n;++col)
                if (covariance[coordinate*n+col]!=(coordinate==col ? 1.0 : 0.0)
                    || covariance[col*n+coordinate]!=(coordinate==col ? 1.0 : 0.0)) candidate->vacuum[j]=0;
            candidate->stats.original_input_bytes=candidate->original.capacity()*sizeof(double)
                +(candidate->order.capacity()+candidate->order_position.capacity())*sizeof(int)+candidate->vacuum.capacity();
            candidate->stats.setup_seconds=elapsed(started); candidate->memory(0);
            handle=candidate;
        } catch (...) { delete candidate; throw; }
    });
    if (status!=BF_OK) return nullptr;
    return handle;
}
extern "C" void bf_destroy(BFHandle *handle) { delete handle; }
extern "C" void bf_reset(BFHandle *handle) {
    if (!handle) return;
    ++handle->stats.reset_requests;
    handle->ld.valid=handle->quad.valid=false;
    for (int p=0;p<2;++p) ++handle->stats.counters[p][BF_STATE_INVALIDATIONS];
}
extern "C" int bf_diagnostics(BFHandle *handle,BFDiagnostics *result) {
    if (!handle || !result) return BF_INVALID_ARGUMENT;
    handle->memory(0); *result=handle->stats; return BF_OK;
}
extern "C" int bf_query(BFHandle *handle,const BFQuery *query,int precision,int force,
    BFOutput *output,char *error,size_t cap) {
    return protected_call(error,cap,[&] {
        QueryData q=validate_query(handle,query,output);
        raw_query(handle,q,precision,force!=0,output);
    });
}
extern "C" int bf_probability(BFHandle *handle,const BFQuery *query,int policy,long double guard,
    BFOutput *output,BFConditional *conditional,char *error,size_t cap) {
    return protected_call(error,cap,[&] {
        QueryData q=validate_query(handle,query,output);
        if (!conditional || policy<BF_LONG_DOUBLE || policy>BF_AUTO || !std::isfinite(guard) || !(guard>0 && guard<1))
            throw std::invalid_argument("invalid native precision policy or relative guard");
        *conditional=BFConditional{}; ++handle->stats.probability_calls;
        int precision=(policy==BF_QUAD) ? BF_QUAD : BF_LONG_DOUBLE;
        for (int attempt=0;attempt<2;++attempt) {
            bool batch_complete=false;
            conditional->precision_used=precision;
            conditional->maximum_event_relative_indicator=0;
            try {
                raw_query(handle,q,precision,false,output); batch_complete=true;
                score(q,*output,guard,*conditional);
                if (conditional->refinement_attempted) ++handle->stats.auto_refinement_successes;
                return;
            } catch (const NumericError& e) {
                if (batch_complete) ++handle->stats.counters[precision][BF_NUMERIC_FAILURES];
                if (policy==BF_AUTO && precision==BF_LONG_DOUBLE) {
                    conditional->refinement_attempted=1;
                    ++handle->stats.auto_refinements;
                    error_text(conditional->first_error,BF_ERROR_CAPACITY,e.what());
                    precision=BF_QUAD;
                } else {
                    if (conditional->refinement_attempted) ++handle->stats.auto_refinement_failures;
                    throw;
                }
            }
        }
        throw std::logic_error("native precision state machine exceeded two attempts");
    });
}
