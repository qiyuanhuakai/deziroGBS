"""Exact one-arm passive pair-source sampler, full-sector and scaling controls."""
import argparse
import ctypes as ct
from math import factorial
from time import perf_counter
from scipy.stats import nbinom
from iteration96_group_common_v1 import ROOT,H,save,manifest,np,shutil,subprocess,Path,os,D,I,U,T,B,J,P,bind
from iteration96_group_benchmark_v1 import fock,Native,BUILD
p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
source=ROOT/'native/heralded_fock96_v1.cpp'
ref=ROOT/'results/iteration96_optical_v1'
cases=[('twins',8,0,.25),('weak',8,.025,.24**2),('mixed',8,.30,.24**2),('sparse128',64,.30,.24**2),('sparse1152',576,.30,.24**2)]
sources=[source,Path(__file__),ROOT/'native/gaussian_rng_domain84_v1.hpp',BUILD,ROOT/'experiments/iteration96_group_benchmark_v1.py',
         *(ref/f'{n}_reference.npz' for n,*_ in cases[:3])]
save(out,'plan.json',dict(cases=cases,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},domain=96030000,
    audit='all balanced occupations through three pairs; optical Fock amplitudes independent of ordered annihilations; 32 higher patterns'))
cmd=[shutil.which('g++'),'-std=c++20','-shared','-O3','-fopenmp','-ffp-contract=off','-fno-fast-math',str(source),'-o',str(out/'herald.dll')]
res=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=res.returncode,stderr=res.stderr))
res.check_returncode()
native=Native(BUILD);lib=ct.CDLL(cmd[-1]);prep=bind(lib,'herald96_prepare',[I,D,D,T],P)
free=bind(lib,'herald96_free',[P],None);draw=bind(lib,'herald96_draw',[P,I,U,U,I,T,B])
ordered=bind(lib,'herald96_ordered',[P,T,J,I],ct.c_double)
rows=[]
for case,(name,d,angle,prob) in enumerate(cases):
    start=perf_counter();v=np.eye(d,dtype=complex)
    for step in range(2):
        for a in range(step,d,2):
            b=(a+1)%d;phase=np.exp(1j*(a+1)*.317)
            v[[a,b]]=np.array([[np.cos(angle),np.sin(angle)*phase],[-np.sin(angle)*phase.conjugate(),np.cos(angle)]])@v[[a,b]]
    probv=np.full(d,prob);err=np.ones(1,np.uint32)
    h=prep(d,v.view(float),probv,err);assert h and not err[0]
    row=dict(case=name,ports=2*d,preparation_s=perf_counter()-start,mean_pairs=d*prob/(1-prob))
    def batch(n,exp,off,workers):
        x=np.empty((n,2*d),np.uint32);flags=np.empty(n,np.uint8)
        before=perf_counter();assert draw(h,n,exp,off,workers,x,flags)==0
        elapsed=perf_counter()-before
        return x,flags,elapsed
    try:
        if d==8:
            data=np.load(ref/f'{name}_reference.npz');barg=data['barg']
            assert np.max(abs(barg[:d,d:]-v*np.sqrt(prob)))<1e-12
            occ,fp=fock(native,barg,data['q'],8)
            balanced=(occ[:,:d].sum(1)==occ[:,d:].sum(1))
            assert fp[~balanced].sum()<1e-23
            num=occ[:,:d].sum(1);rng=np.random.default_rng(96030)
            checked=np.r_[np.flatnonzero(balanced&(num<=3)),rng.choice(np.flatnonzero(balanced&(num==4)),32,replace=False)]
            errors=[];orders=[]
            for k in checked:
                signal=occ[k,:d];idler=occ[k,d:].astype(np.uint32)
                seq=np.repeat(np.arange(d),signal).astype(np.int32);length=len(seq)
                if not length:seq=np.zeros(1,np.int32)
                mult=factorial(length)
                for vj in signal:mult//=factorial(vj)
                conditional=ordered(h,idler,seq,length)
                rev=ordered(h,idler,seq[::-1].copy(),length)
                assert conditional>=0 and rev>=0
                predicted=(1-prob)**d*prob**length*mult*conditional
                errors.append(abs(predicted-fp[k]));orders.append(abs(rev-conditional))
            row.update(checked=len(checked),max_fock_error=max(errors),l1_checked=sum(errors),order_reverse_max=max(orders),
                       failure_bound=float(nbinom.sf(24,d,1-prob)))
            assert row['max_fock_error']<1e-13 and max(orders)<1e-13
        domain=96030000+case
        x,flags,_=batch(256,domain,0,1);part,partflags,_=batch(224,domain,32,4)
        assert np.array_equal(part,x[32:]) and np.array_equal(partflags,flags[32:])
        n=2048 if d<=64 else 4096
        samples=0;failures=0;imbalances=0;hist=np.zeros(512,int);count=np.zeros(1<<(2*d) if d==8 else 1,int)
        row['timing']=[];record_hash=[]
        import hashlib
        for workers in [1,16]:
            stamps=[]
            for rep in range(5):
                x,flags,elapsed=batch(n,domain,1024+(workers*5+rep)*n,workers);stamps.append(elapsed*1e6/n)
                failures+=int(np.count_nonzero(flags));samples+=n
                good=x[flags==0];tot=good[:,:d].sum(1).astype(int)
                imbalances+=int(np.count_nonzero(tot!=good[:,d:].sum(1)))
                hist+=np.bincount(tot,minlength=len(hist))
                if d==8:count+=np.bincount((good>0)@(2**np.arange(2*d)),minlength=len(count))
                record_hash.append(hashlib.sha256(x).hexdigest())
            row['timing'].append(dict(workers=workers,batch=n,us=stamps,median=float(np.median(stamps))))
        row.update(attempts=samples,failures=failures,imbalances=imbalances,record_hash=record_hash,
            count_ks=float(np.max(abs(hist.cumsum()/samples-nbinom.cdf(np.arange(len(hist)),d,1-prob)))))
        assert failures==0 and imbalances==0
        if d==8:row['empirical_click_tv']=float(abs(count/samples-data['click']).sum()/2)
        np.savez_compressed(out/f'{name}.npz',click_hist=count,pair_count_hist=hist,first=x[:32],source_p=probv,unitary=v)
        rows.append(row);save(out,'summary.json',rows)
        print(name,row['timing'][0]['median'],row['timing'][1]['median'],row.get('max_fock_error'),flush=True)
    finally:free(h)
manifest(out)
