"""Gaussian block selection, count marks, and physically indexed tree tables."""
from concurrent.futures import ThreadPoolExecutor
import ctypes as ct
from functools import lru_cache
import math
import os
from pathlib import Path
import shutil
import subprocess
from time import perf_counter

import networkx as nx
import numpy as np
from iteration92_common_v2 import ROOT,H,save,read,manifest

SCALE=2**53
D,B,W,T,J=[np.ctypeslib.ndpointer(dtype=t,flags="C_CONTIGUOUS") for t in (np.float64,np.uint8,np.uint64,np.uint32,np.int32)]
I,P,U,F=ct.c_int,ct.c_void_p,ct.c_uint64,ct.c_double
def bind(lib,name,args,result=I):
    f=getattr(lib,name);f.argtypes=args;f.restype=result;return f
class Native:
    def __init__(self,path):
        self.runtime=os.add_dll_directory(str(Path(shutil.which("g++")).parent));dll=ct.CDLL(str(path));self.dll=dll
        self.click=bind(dll,"group96_click",[I,I,D,D,I,D,D])
        self.ternary=bind(dll,"group96_ternary",[I,I,D,D,D,D])
        self.radial=bind(dll,"alloc95_radial",[I,I,I,I,I,D,D,D])
        self.opt=bind(dll,"window95_moments",[I,I,D,D,D,D,D,D])
        self.prepare=bind(dll,"group96_prepare",[I,I,I,I,I,I,J,J,D,D,W,D],P)
        self.target=bind(dll,"group96_target",[P,D,F,D]);self.free=bind(dll,"group96_destroy",[P],None)
        self.draw=bind(dll,"group96_draw",[P,I,U,U,I,I,B,T,B,B])
        self.fock=bind(dll,"group96_fock",[I,I,I,D,F,W,D])
def build(out):
    cmd=read(ROOT/"results/iteration95_loop_alloc_v1/build.json")["command"]
    cmd[-3]=str(ROOT/"native/gaussian_group_tree96_v1.cpp");cmd[-1]=str(out/"tree.dll")
    run=subprocess.run(cmd,capture_output=True,text=True)
    save(out,"build.json",dict(command=cmd,code=run.returncode,stderr=run.stderr))
    run.check_returncode()
    return Native(out/"tree.dll")
def entropy(p):
    x=np.asarray(p);v=x[x>0];return float(-v@np.log(v))
def boson_entropy(nu):
    n=np.maximum(0,(nu-1)/2);return np.log1p(n)+n*np.log1p(1/np.maximum(n,1e-300))
def gaussian_entropy(q):
    ans=0.
    for s in q:
        m=len(s)//2;v=2*s-np.eye(2*m);l=np.linalg.cholesky((v+v.T)/2)
        omega_l=np.vstack([l[m:],-l[:m]])
        nu=np.linalg.eigvalsh(1j*(l.T@omega_l))[m:]
        assert nu.min()>1-1e-8
        ans+=boson_entropy(nu).sum()
    return float(ans)

def pair_information(q):
    m=q.shape[1]//2
    info=np.zeros((m,m));single=np.zeros(m);i,j=np.triu_indices(m,1)
    for s in q:
        v=2*s-np.eye(2*m)
        a=np.arange(m);det=v[a,a]*v[a+m,a+m]-v[a,a+m]**2
        hs=boson_entropy(np.sqrt(det));single+=hs
        for offset in range(0,len(i),16384):
            x=i[offset:offset+16384];y=j[offset:offset+16384]
            ix=np.stack([x,x+m,y,y+m],axis=1);sub=v[ix[:,:,None],ix[:,None,:]]
            dc=sub[:,0,2]*sub[:,1,3]-sub[:,0,3]*sub[:,1,2]
            delta=det[x]+det[y]+2*dc;dv=np.linalg.det(sub)
            hi=(delta+np.sqrt(np.maximum(0,delta**2-4*dv)))/2
            lo=dv/hi
            value=hs[x]+hs[y]-boson_entropy(np.sqrt(lo))-boson_entropy(np.sqrt(hi))
            # A repeated symplectic eigenvalue near vacuum can lose double
            # precision after subtracting the discriminant. Recompute when needed.
            suspect=(abs(delta**2-4*dv)<1e-12*delta**2)&(np.linalg.norm(sub[:,:2,2:],axis=(1,2))>1e-11)
            if suspect.any():
                omega=np.kron(np.eye(2),[[0,1],[-1,0]])
                l=np.linalg.cholesky(sub[suspect]);nu=np.linalg.eigvalsh(1j*(l.transpose(0,2,1)@(omega@l)))[:,2:]
                value[suspect]=hs[x[suspect]]+hs[y[suspect]]-boson_entropy(nu).sum(1)
            value[np.linalg.norm(sub[:,:2,2:],axis=(1,2))<1e-14]=0
            info[x,y]+=np.maximum(0,value)
    return info+info.T,single
def pair_up(weights):
    n=len(weights)
    if n==1:return [[0]]
    if n<=32:
        graph=nx.from_numpy_array(weights)
        # Zero-weight edges allow a maximum-cardinality partition even when independent.
        for a in range(n):
            for b in range(a):graph.add_edge(a,b,weight=weights[a,b])
        pairs=[sorted(x) for x in nx.max_weight_matching(graph,maxcardinality=True)]
    else:
        i,j=np.triu_indices(n,1);free=np.ones(n,bool);pairs=[]
        for k in np.argsort(-weights[i,j],kind="stable"):
            a,b=i[k],j[k]
            if free[a] and free[b]:
                pairs.append([int(a),int(b)]);free[[a,b]]=False
                if len(pairs)==n//2:break
        # Strict score improvements; isolated zero-weight ties keep the original order.
        for _ in range(4):
            changed=False
            for t,(a,b) in enumerate(pairs):
                rest=np.asarray(pairs[t+1:],int)
                if not len(rest):continue
                c,e=rest.T
                old=weights[a,b]+weights[c,e]
                gain=np.stack([weights[a,c]+weights[b,e]-old,weights[a,e]+weights[b,c]-old])
                option,index=np.unravel_index(np.argmax(gain),gain.shape)
                if gain[option,index]>1e-13:
                    v,w=rest[index]
                    if option:v,w=w,v
                    pairs[t]=[a,int(v)];pairs[t+1+index]=[b,int(w)];changed=True
            if not changed:break
    used={i for pair in pairs for i in pair}
    return sorted(pairs+[[i] for i in range(n) if i not in used],key=min)
def block_weights(groups,info):
    # Sums over cross-block pair mutual information are candidate-edge scores.
    out=np.zeros((len(groups),len(groups)))
    for a in range(max(map(len,groups))):
        ids=np.array([g[a] if len(g)>a else -1 for g in groups]);valid=ids>=0
        for b in range(max(map(len,groups))):
            js=np.array([g[b] if len(g)>b else -1 for g in groups]);active=js>=0
            out[np.ix_(valid,active)]+=info[np.ix_(ids[valid],js[active])]
    np.fill_diagonal(out,0);return out
def partition(m,width,adaptive,info):
    if not adaptive:return [list(range(i,min(m,i+width))) for i in range(0,m,width)]
    groups=[[i] for i in range(m)]
    while max(map(len,groups))<width and len(groups)>1:
        weight=info if len(groups)==m else block_weights(groups,info)
        groups=[sorted(v for old in pair for v in groups[old]) for pair in pair_up(weight)]
    return groups
def spanning(w):
    # Dense deterministic Prim, including zero-weight links between independent components.
    n=len(w);seen=np.zeros(n,bool);best=np.full(n,-np.inf);par=np.zeros(n,int);best[0]=0
    edges=[]
    for _ in range(n):
        i=int(np.argmax(np.where(seen,-np.inf,best)));seen[i]=True
        if i:edges.append(tuple(sorted((i,int(par[i])))))
        better=(w[i]>best)&~seen;best[better]=w[i,better];par[better]=i
    return edges
def codes_cdf(v):
    x=np.maximum(v,0);y=np.rint(SCALE*x.cumsum()/x.sum()).clip(0,SCALE).astype(np.uint64);y[-1]=SCALE
    return y
def integer_prob(v):
    cd=codes_cdf(v);return np.diff(np.r_[np.uint64(0),cd]).astype(float)/SCALE

class Optical:
    def __init__(self,lib,q,mu):
        self.lib=lib;self.q=q;self.mu=mu;self.m=q.shape[1]//2
        self.max_negative=0.
    def sub(self,ids):
        ids=np.asarray(ids,int);ix=np.r_[ids,self.m+ids]
        return np.ascontiguousarray(self.q[:,ix][:,:,ix]),np.ascontiguousarray(self.mu[:,ix])
    @lru_cache(None)
    def law(self,ids,base):
        q,mu=self.sub(ids);n=len(ids);v=np.empty(base**n);d=np.zeros(2)
        rc=(self.lib.click(n,len(q),q,mu,12 if n>10 else 1,v,d) if base==2
            else self.lib.ternary(n,len(q),q,mu,v,d))
        assert rc==0,(ids,base,rc,d)
        self.max_negative=max(self.max_negative,d[0])
        return v
    @lru_cache(None)
    def local_entropy(self,ids):
        return gaussian_entropy(self.sub(ids)[0])
    @lru_cache(None)
    def photon_split(self,ids,cap):
        size=len(ids);q,mu=self.sub(ids);stride=cap+2;freq=1<<(stride-1).bit_length()
        values=np.empty((freq,stride),complex)
        assert self.lib.radial(size,1,len(q),cap+1,freq,q,mu,values.view(float))==0
        coeff=np.fft.fft(values,axis=0).real/freq
        return coeff

def construct(opt,groups,weight,base,cap,connected=True):
    groups=[tuple(g) for g in groups];ng=len(groups);k=base**max(map(len,groups));support=cap+2
    singles=[opt.law(g,base) for g in groups];joint={};mi={}
    if connected and ng>1:
        # Exact Chow-Liu on small graphs; optical candidate pool on large graphs.
        if ng<=32:edges=[(a,b) for a in range(ng) for b in range(a+1,ng)]
        else:
            edges=set(spanning(weight))
            for a in range(ng):
                for b in np.argsort(-weight[a],kind="stable")[:3]:
                    if a!=b:edges.add(tuple(sorted((a,int(b)))))
            edges=sorted(edges)
        def edge(e):
            a,b=e;na=len(singles[a])
            if weight[a,b]<=1e-15:
                # Only factor structural zeros; tiny nonzero correlations are evaluated.
                qa=opt.sub(groups[a]+groups[b])[0];d=len(groups[a])+len(groups[b])
                ix=np.r_[np.arange(len(groups[a])),d+np.arange(len(groups[a]))]
                jx=np.setdiff1d(np.arange(2*d),ix)
                independent=np.count_nonzero(qa[:,ix][:,:,jx])==0
            else:independent=False
            v=np.outer(singles[a],singles[b]) if independent else opt.law(groups[a]+groups[b],base).reshape(-1,na).T
            assert max(abs(v.sum(1)-singles[a]).max(),abs(v.sum(0)-singles[b]).max())<2e-10
            return e,v,max(0.,entropy(singles[a])+entropy(singles[b])-entropy(v))
        with ThreadPoolExecutor(max_workers=12) as pool:
            for e,v,w in pool.map(edge,edges):joint[e]=v;mi[e]=w
        graph=nx.Graph();graph.add_nodes_from(range(ng));graph.add_weighted_edges_from((a,b,mi[a,b]) for a,b in edges)
        picked=nx.maximum_spanning_tree(graph)
        root=max(range(ng),key=lambda j:picked.degree[j]);order=list(nx.bfs_tree(picked,root));previous=dict(nx.bfs_predecessors(picked,root))
    else:
        root=0;order=list(range(ng));previous={i:0 for i in range(1,ng)}
    trans=np.zeros((ng,k,k));em=np.zeros((ng,k,support));splits=np.zeros((ng,k,support,support),np.uint64) if base==3 else np.zeros(1,np.uint64)
    parents=np.full(ng,-1,np.int32);ports=np.full((ng,8),-1,np.int32);opt_tail=0.;edge_info=0.
    maps={v:i for i,v in enumerate(order)}
    for i,g in enumerate(order):
        ids=groups[g];size=len(ids);ports[i,:size]=ids;s=singles[g]
        trans[i,:,:len(s)]=s[None,:]
        if i:
            parent=previous[g];parents[i]=maps[parent]
            if connected:
                e=tuple(sorted((g,parent)));v=joint[e] if parent<g else joint[e].T
                edge_info+=mi[e]
                for a,row in enumerate(v):
                    if row.sum()>1e-17:trans[i,a,:len(s)]=row/row.sum()
        for a in range(k): trans[i,a]=integer_prob(trans[i,a])
        if base==2:
            for a in range(k):em[i,a,a.bit_count()]=1.
            continue
        coeff=opt.photon_split(ids,cap)
        for n in range(cap+1):
            for a in ([n] if size==1 else range(n+1)):
                state=min(a,2)+(3*min(n-a,2) if size==2 else 0)
                em[i,state,n]+=max(0.,coeff[a,n])
        for state in range(k):
            if state>=len(s) or s[state]<1e-21:
                em[i,state]=0;em[i,state,0]=1
                continue
            ca,cb=state%3,state//3
            lost=s[state]-em[i,state].sum()
            assert lost>=-3e-10,(ids,state,lost)
            if ca==2 or cb==2:em[i,state,cap+1]=max(0.,lost);opt_tail+=max(0.,lost)
            em[i,state]=integer_prob(em[i,state])
            if size==2 and ca==2 and cb==2:
                for n in range(4,cap+2):
                    v=np.zeros(n+1)
                    if n<=cap:v[2:n-1]=np.maximum(0,coeff[2:n-1,n])
                    if v.sum()<=0:v[n//2]=1
                    splits[i,state,n,:n+1]=codes_cdf(v)
    return dict(parents=parents,ports=ports,transition=trans,emission=em,split=splits),dict(
        candidate_edges=len(joint),tree_mi=edge_info,tail_union=opt_tail,
        block_entropy=sum(opt.local_entropy(x) for x in groups),
        block_classical=sum(entropy(x) for x in singles),groups=ng,negative_mass=opt.max_negative)
def joint_values(table,base,states):
    par=table["parents"];ports=table["ports"];tr=table["transition"];em=table["emission"];split=table["split"]
    p=np.ones(len(states));zlist=[]
    for i,ix in enumerate(ports):
        ids=ix[ix>=0]
        a=states[:,ids[0]].astype(int);v=np.minimum(states[:,ids],base-1).astype(int)
        z=v@(base**np.arange(len(ids)));n=states[:,ids].sum(1).astype(int)
        parent=0 if par[i]<0 else zlist[par[i]]
        p*=tr[i,parent,z];zlist.append(z)
        if base==3:
            p*=em[i,z,n]
            if len(ids)==2:
                sel=(z==8)
                high=split[i,8,n[sel],a[sel]]
                low=split[i,8,n[sel],np.maximum(0,a[sel]-1)]
                p[sel]*=(high-low).astype(float)/SCALE
    return p
