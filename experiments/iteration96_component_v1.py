"""Identify exact optical components and sample a complete click word per component."""
import argparse
import ctypes as ct
from time import perf_counter
from scipy.sparse.csgraph import connected_components
from scipy.linalg import block_diag
from iteration96_group_common_v1 import ROOT,Path,np,shutil,subprocess,D,J,B,W,P,I,U,bind,Native,Optical,H,save,manifest
from iteration96_group_benchmark_v1 import BUILD,fixture
def components(q):
    d=q.shape[1]//2
    edges=np.any(q.reshape(len(q),2,d,2,d)!=0,axis=(0,1,3))
    _,comp=connected_components(edges,directed=False)
    return [np.flatnonzero(comp==i).tolist() for i in range(int(comp.max())+1)]

def repeat(q,mu,copies):
    d=len(mu[0])//2
    cov=np.array([block_diag(*([s]*copies)) for s in q])
    ix=np.arange(copies*2*d).reshape(copies,2,d).transpose(1,0,2).ravel()
    return np.ascontiguousarray(cov[:,ix][:,:,ix]),np.ascontiguousarray(np.tile(mu,copies)[:,ix])

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    raw=ROOT/'results/iteration96_optical_v1';source=ROOT/'native/component_click96_v1.cpp'
    files=[Path(__file__),source,BUILD,ROOT/'native/gaussian_rng_domain84_v1.hpp',raw/'manifest.json']
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},max_component=18,
        connectivity='exact nonzero inter-port block in any Gaussian spectral covariance',seed=96040000))
    cmd=[shutil.which('g++'),'-shared','-O3','-std=c++20','-fopenmp','-ffp-contract=off','-fno-fast-math',str(source),'-o',str(out/'component.dll')]
    proc=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=proc.returncode,stderr=proc.stderr));proc.check_returncode()
    lib=Native(BUILD);rawlib=ct.CDLL(cmd[-1])
    prep=bind(rawlib,'component96_prepare',[I,I,J,J,D],P);free=bind(rawlib,'component96_free',[P],None)
    draw=bind(rawlib,'component96_draw',[P,I,U,U,I,B]);cuts=bind(rawlib,'component96_cuts',[P,I,W])
    rows=[]
    for case,name in enumerate(['twins','weak','mixed','actual16','repeat128','repeat1152','actual1152']):
        repeated=name.startswith('repeat')
        if name=='actual1152':q,mu,_,_=fixture(name)
        else:
            ref=np.load(raw/f'{"mixed" if repeated else name}_reference.npz');q,mu=ref['q'],ref['mu']
        if repeated:q,mu=repeat(q,mu,int(name[6:])//16)
        m=q.shape[1]//2;clock=perf_counter();groups=components(q)
        row=dict(case=name,m=m,sizes=[len(g) for g in groups],graph_s=perf_counter()-clock)
        if max(map(len,groups))>18:
            row['eligible']=False;rows.append(row);continue
        opt=Optical(lib,q,mu);cache={};laws=[];start=perf_counter()
        for g in groups:
            qc,md=opt.sub(g);key=(qc.tobytes(),md.tobytes())
            if key not in cache:cache[key]=opt.law(tuple(g),2)
            laws.append(cache[key])
        h=prep(m,len(groups),np.array(list(map(len,groups)),np.int32),np.array(sum(groups,[]),np.int32),np.concatenate(laws))
        assert h;row.update(eligible=True,setup_s=perf_counter()-start+row['graph_s'],unique_optical_tables=len(cache),quantization_tv_union=0.)
        try:
            realized=[]
            for j,law in enumerate(laws):
                c=np.zeros(len(law),np.uint64);assert cuts(h,j,c)==0
                assert np.all(c[1:]>=c[:-1]) and c[-1]==2**53
                p=np.diff(np.r_[np.uint64(0),c]).astype(float)/2**53
                row['quantization_tv_union']+=float(abs(p-law).sum()/2);realized.append(p)
            if not repeated:
                ix=np.arange(1<<m);model=np.ones(len(ix))
                for g,p in zip(groups,realized):
                    code=sum(((ix>>a)&1)<<j for j,a in enumerate(g));model*=p[code]
                row['full_click_tv']=float(abs(model-ref['click']).sum()/2)
            batch=2048 if m==16 else 8192;domain=96040000+case
            x=np.empty((batch,m),np.uint8);times=[]
            for threads in [1,16]:
                stamps=[]
                for rep in range(5):
                    start=perf_counter();x=np.empty((batch,m),np.uint8)
                    assert draw(h,batch,domain,(threads*5+rep)*batch,threads,x)==0
                    stamps.append((perf_counter()-start)*1e6/batch)
                times.append(dict(threads=threads,batch=batch,us=stamps,median=float(np.median(stamps))))
            row['timing']=times;row['report_negative_mass']=opt.max_negative
            a=np.empty((64,m),np.uint8);b=np.empty((48,m),np.uint8)
            assert draw(h,64,domain,0,1,a)==draw(h,48,domain,16,4,b)==0 and np.array_equal(a[16:],b)
            np.savez_compressed(out/f'{name}.npz',first=a,sizes=np.array(row['sizes']),ports=np.array(sum(groups,[])),**({'law':model} if not repeated else {}))
            print(name,row.get('full_click_tv'),row['setup_s'],times[0]['median'],flush=True)
        finally:free(h)
        rows.append(row);Optical.law.cache_clear()
    save(out,'summary.json',rows);manifest(out)
if __name__=='__main__':main()
