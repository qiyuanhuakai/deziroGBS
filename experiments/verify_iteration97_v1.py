"""Independent mixed-alphabet, optical-state, first-batch and publication checks."""
import argparse,ctypes as ct,subprocess,sys
from pathlib import Path
from time import perf_counter
from scipy.linalg import sqrtm
from threadpoolctl import threadpool_limits
from iteration97_conditionals_v1 import ROOT,Kernel,Optical,BUILD,PREVIOUS,np,H,save,manifest,I,D,B,W,P,bind
from iteration97_sample_v1 import prepare
from iteration97_field_v1 import field_factor
from iteration96_group_benchmark_v1 import ACTUAL,HIST
from iteration96_group_common_v1 import read
DATA=ROOT/'results'
def fidelity(a,b):
    # Inputs V use vacuum I. The fidelity formula uses vacuum I/2.
    a=a/2;b=b/2;n=len(a)//2;eye=np.eye(2*n)
    omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
    aux=omega.T@np.linalg.solve(a+b,.25*omega+b@omega@a)
    arg=eye+.25*np.linalg.inv(aux@omega@aux@omega)
    root=sqrtm(arg);sign,det=np.linalg.slogdet(2*(eye+root)@aux)
    assert abs(sign-1)<1e-7 and abs(root@root-arg).max()<1e-8
    ldet=np.linalg.slogdet(a+b)[1]
    f=float(np.exp(.5*(det-ldet)))
    eig=np.linalg.eigvals(2j*aux@omega)
    assert abs(eig.imag).max()<1e-7
    w=np.sort(eig.real)[n:]
    assert w.min()>1-1e-9
    f2=float(np.exp(np.arccosh(np.maximum(1.,w)).sum()-.5*ldet))
    assert abs(f-f2)<3e-7,(f,f2)
    return f,f2

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
    dirs=[DATA/f'iteration97_{x}_v1' for x in ['build','panel','field']]
    dll=DATA/'iteration96_frugal_build_v1/frugal.dll';ref=DATA/'iteration96_optical_v1/actual16_reference.npz'
    oracle=DATA/'iteration96_quad_v1/actual16.npz';src=ROOT/'experiments/iteration97_sample_v1.py'
    comp=DATA/'iteration96_component_v1/component.dll';files=[Path(__file__),src,comp,dll,ref,oracle,BUILD,PREVIOUS,*(d/'manifest.json' for d in dirs)]
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},rng=97090000,
        cold='both APIs loaded; identical in-memory q; single thread incl both preparations and OR, randomized order, 7 reps',small_checks='all child ternary and parent bit patterns for five physically permuted displaced controls'))
    count=0
    for d in dirs:
        m=read(d/'manifest.json')
        assert set(m)=={p.name for p in d.iterdir() if p.name!='manifest.json'}
        for k,v in m.items():assert H(d/k)==v;count+=1
        for k,v in read(d/'plan.json')['hashes'].items():assert H(ROOT/k)==v
    kernel=Kernel();rng=np.random.default_rng(97090000);z=np.load(ref);q,mu=z['q'],z['mu'];m=16
    err=0.;cases=0
    for h,w in [(0,1),(1,1),(2,2),(3,2),(4,1)]:
        ids=rng.permutation(m)[:h+w];ix=np.r_[ids,ids+m];qc=q[:,ix][:,:,ix].copy()
        md=np.ascontiguousarray(rng.normal(0,.17,(2,2*(w+h))))
        opt=Optical(kernel.prior,qc,md);full=opt.law(tuple(range(h+w)),3)
        pos=np.arange(len(full));parent=sum((((pos//3**i)%3)>0).astype(int)<<i for i in range(h));child=pos//3**h
        expected=np.bincount(parent*3**w+child,weights=full,minlength=2**h*3**w).reshape(2**h,-1)
        actual=kernel.joint(opt,list(range(h)),list(range(h,h+w)),3,2)[0]
        err=max(err,float(np.max(abs(expected-actual))));cases+=expected.size
    assert err<1e-11
    # Read back the integer CDF actually used to draw records, in physical bit order.
    law=kernel.joint(Optical(kernel.prior,q,mu),[],list(range(m)),2,1)[0].ravel()
    lib=ct.CDLL(str(comp));J=np.ctypeslib.ndpointer(np.int32,flags='C_CONTIGUOUS')
    cp=bind(lib,'component96_prepare',[I,I,J,J,D],P);fetch=bind(lib,'component96_cuts',[P,I,W]);free=bind(lib,'component96_free',[P],None)
    h=cp(16,1,np.array([16],np.int32),np.arange(16,dtype=np.int32),law);assert h
    cuts=np.empty(65536,np.uint64)
    try:assert fetch(h,0,cuts)==0
    finally:free(h)
    actual=np.diff(cuts,prepend=np.uint64(0))/float(2**53);truth=np.load(oracle)['reference']
    tv=float(abs(actual-truth).sum()/2);assert tv<1e-10
    flib=ct.CDLL(str(dll));fsample=bind(flib,'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I]);msg=ct.create_string_buffer(4096)
    results={name:dict(cold=[],warm=[]) for name in ['schur','K2','K4']}
    for rep in range(8):
        for name in rng.permutation(list(results)):
            t0=perf_counter()
            if name=='schur':
                emit,close,_=prepare(kernel,q,mu,'click',8,1)
                try:
                    t1=perf_counter();x,_=emit(2048,97091100+rep,0,1);warm=perf_counter()-t1
                finally:close()
            else:
                covs=2*q-np.eye(32);xs=[];warm=0.
                for sector,cov in enumerate(covs):
                    x=np.empty((2048,16),np.uint8);st=np.zeros(7)
                    assert fsample(int(name[1]),2048,97091200+100*int(name[1])+2*rep+sector,cov,x,st,msg,len(msg))==0,msg.value
                    assert st[5]==64;warm+=st[3]+st[4];xs.append(x)
                t1=perf_counter();x=np.bitwise_or(*xs);warm+=perf_counter()-t1
            cold=perf_counter()-t0
            if rep:results[name]['cold'].append(cold*1e6/2048);results[name]['warm'].append(warm*1e6/2048)
    for row in results.values():
        row.update(cold_median=float(np.median(row['cold'])),warm_median=float(np.median(row['warm'])))
    source=[];fid={}
    for s in range(2):
        zz=np.load(ACTUAL/f'input_{s}.npz');t,L=zz['t'],zz['noise']
        v=np.diag(1+2*np.r_[t/(1-t),-t/(1+t)])+4*L@L.T
        for method in ['displaced_vacuum','normal_preserving']:
            g,log_lower,_=field_factor(t,L,method);cl=np.eye(len(v))+2*g@g.T
            f,f2=fidelity(v,cl)
            assert f>np.exp(log_lower)-1e-10 and f<=1+1e-8
            fid.setdefault(method,[]).append([f,f2])
        source.append(v)
    for route in ('click','pnr'):
        path=out/f'{route}_cli.npz'
        cmd=[sys.executable,str(src),str(ref),str(path),'--route',route,'--seed','97092000','--shots','256','--workers','2']
        subprocess.run(cmd,check=True)
        result=np.load(path);emit,close,_=prepare(kernel,q,mu,route,8,1)
        try:x,tail=emit(224,97092000,32,1)
        finally:close()
        assert np.array_equal(result['records'][32:],x) and np.array_equal(result['tail'][32:],tail)
    save(out,'validation.json',dict(passed=True,archived_files=count,conditional_cells=cases,conditional_max_error=err,
        complete_16_tv=tv,benchmark=results,fidelity_by_spectrum=fid,full_tv_from_fidelity={k:float(np.sqrt(1-np.prod(np.array(v)[:,0]))) for k,v in fid.items()}))
    np.savez_compressed(out/'direct16.npz',law=actual);manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
