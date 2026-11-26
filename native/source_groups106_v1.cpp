#include "two_mode_source106_v1.hpp"
#include <numeric>

// Each node is either a single-source leaf or a coherent set of sources.
// Children precede their parent. All offsets count elements, not bytes.
struct Node106 {
    int left,right,source,modes,cutoff,rows,port_offset,matrix_offset,level_offset,state_offset,first_source;
};
struct Level106 { int size,parent_offset; };
struct Row106 { double upper,correction; };

static int coherent106(const Node106& node,int photons,int physical,int pnr,
    const Row106* rows,const C* transfer,const Level106* levels,const int* parent,
    const double* ladder,const int* states,const double* initial,
    DomainRng84& port,DomainRng84& accept,std::vector<C>& psi,std::vector<C>& next,unsigned* record) {
    const Level106* level=levels+node.level_offset;
    const int dim=node.modes;
    const double* wave=initial+states[node.state_offset+photons/2];
    psi.assign(wave,wave+level[photons].size);
    const auto* proposal=rows+node.port_offset;
    while(photons>0) {
        double u=uniform84(port);
        const int j=int(std::upper_bound(proposal,proposal+node.rows,u,
            [](double v,const Row106& r){return v<r.upper;})-proposal);
        if(j>=node.rows)return 4;
        const auto* direction=transfer+node.matrix_offset+std::size_t(j)*dim;
        const auto& block=level[photons];
        const int size=level[photons-1].size;
        const auto* src=parent+block.parent_offset;
        const auto* weights=ladder+block.parent_offset;
        next.resize(size);
        double norm=0.;
        for(int i=0;i<size;++i) {
            C value{};
            for(int h=0;h<dim;++h) value+=direction[h]*weights[i*dim+h]*psi[src[i*dim+h]];
            next[i]=value; norm+=std::norm(value);
        }
        // The row is normalized. The norm squared is at most the number
        // of remaining photons. Correct the finite proposal probabilities.
        if(!std::isfinite(norm))return 5;
        if(uniform84(accept)*photons>=norm*proposal[j].correction)continue;
        if(j<physical){if(pnr)++record[j];else record[j]=1;}
        if(--photons) {
            const double inv=1./std::sqrt(norm);
            for(auto& a:next)a*=inv;
            psi.swap(next);
        }
    }
    return 0;
}

extern "C" __declspec(dllexport) int group106_draw(int m,int sectors,int shots,int pnr,int workers,
    U seed,unsigned offset,int port_bits,int max_n,const int* meta,const U* totals,const unsigned* total_alt,
    const U* cuts,const unsigned* alt,const Direction105* dirs,const double* roots,
    int node_count,const Node106* nodes,const Row106* rows,const C* transfer,
    const Level106* levels,const int* parent,const double* ladder,const int* states,const double* initial,
    unsigned* output) {
    if(m<1||sectors<1||sectors>65536||node_count<sectors||node_count>65536||shots<1||workers<1||U(offset)+shots>(U(1)<<32))return 1;
    int error=0;
    #pragma omp parallel num_threads(workers) reduction(max:error)
    {
        std::vector<int> pairs(node_count);
        std::vector<unsigned char> skip(node_count);
        std::vector<C> psi,next;
        #pragma omp for schedule(static)
        for(int row=0;row<shots;++row) {
            unsigned* record=output+std::size_t(row)*m;
            std::fill_n(record,m,0U); std::fill(skip.begin(),skip.end(),0);
            for(int i=0;i<node_count;++i) {
                const auto& node=nodes[i];
                if(node.source>=0) {
                    const int s=node.source;const int* info=meta+4*s;
                    DomainRng84 rng(seed,s,201,U(offset)+row);
                    pairs[i]=int(alias105(rng(),info[0],totals+info[1],total_alt+info[1]));
                }else pairs[i]=pairs[node.left]+pairs[node.right];
            }
            for(int i=node_count-1;i>=0;--i) {
                const auto& node=nodes[i];const int k=pairs[i];
                if(node.source<0) {
                    if(skip[i]||k<=node.cutoff)skip[node.left]=skip[node.right]=1;
                    if(skip[i]||!k||k>node.cutoff)continue;
                    DomainRng84 pick(seed,i,211,U(offset)+row),accept(seed,i,212,U(offset)+row);
                    int rc=coherent106(node,2*k,m,pnr,rows,transfer,levels,parent,ladder,states,initial,pick,accept,psi,next,record);
                    if(rc){error=rc;break;}
                }else if(!skip[i]&&k) {
                    int s=node.source;
                    if(2*k>max_n){error=2;break;}
                    psi.resize(2*k+1);
                    DomainRng84 pick(seed,s,202,U(offset)+row),accept(seed,s,203,U(offset)+row);
                    int rc=route_source105(m,pnr,2*k,port_bits,meta+4*s,nullptr,cuts,alt,std::size_t(s)<<port_bits,
                        dirs+std::size_t(s)*(m+2),roots,psi,pick,accept,record);
                    if(rc){error=rc;break;}
                }
            }
        }
    }
    return error;
}
