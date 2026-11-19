"""Verify full-record charges and a selected jump against dense Fock vectors."""
import argparse
import subprocess
import math
from scipy.stats import beta
from iteration94_common_v2 import ROOT,Path,np,C,H,save,manifest,read,shutil,os
from iteration92_quality_v1 import confidence

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
base=ROOT/'results/iteration95_adaptive_pnr_v1'
source=ROOT/'native/gaussian_mps_adaptive95_control.cpp'
targets=ROOT/'results/iteration93_reference_v2'
save(out,'plan.json',dict(dense_cases=12,seed=9507,small_family_events=10,small_pilot=4096,
    small_intervals='Paired outputs permitted across arms; fixed numerical reference and iid records within arm. Per arm: full PNR lower/upper, click lower/upper, failure upper.',
    inputs={f.relative_to(ROOT).as_posix():H(f) for f in [Path(__file__),source,base/'plan.json',
       targets/'law_0.npz',targets/'row_0.json']}))
compiler=Path(shutil.which('g++'))
cmd=[str(compiler),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE',
     '-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(out/'control.dll')]
b=subprocess.run(cmd,capture_output=True,text=True)
save(out,'build.json',dict(command=cmd,code=b.returncode,stderr=b.stderr));b.check_returncode()
runtime=os.add_dll_directory(str(compiler.parent));lib=C.CDLL(str(out/'control.dll'))
D,J=[np.ctypeslib.ndpointer(dtype=t,flags='C_CONTIGUOUS') for t in (float,np.int32)]
fn=lib.adaptive95_control
fn.argtypes=[C.c_int,C.c_int,J,D,D,D,D,C.c_int,C.c_int,C.c_int,C.c_double,D,D]
rng=np.random.default_rng(9507)
dense=[]
for case in range(12):
    n,d=4,3;bonds=np.array([1,3,4,3,1],np.int32)
    sites=[rng.normal(size=(d,l,r))+1j*rng.normal(size=(d,l,r)) for l,r in zip(bonds,bonds[1:])]
    vector=sites[0][:,0,:]
    for a in sites[1:]:vector=np.einsum('...l,slr->...sr',vector,a)
    vector=vector[...,0];vector/=np.linalg.norm(vector)
    bg=rng.uniform(-.75,.75,n);alpha=rng.normal(size=n)+1j*rng.normal(size=n)
    w=rng.normal(size=n)+1j*rng.normal(size=n);w/=np.linalg.norm(w)
    extended=np.pad(vector,[(0,1)]*n)
    exact=np.zeros_like(extended)
    annihilate=np.diag(np.sqrt(np.arange(1,d+1)),1)
    for i in range(n):
        op=alpha[i]*np.eye(d+1)+(annihilate+bg[i]*annihilate.T)/np.sqrt(1-bg[i]**2)
        exact+=w[i]*np.moveaxis(np.tensordot(op,extended,axes=(1,i)),0,i)
    exact/=np.linalg.norm(exact)
    output=np.empty((d+1)**n,complex);trace=np.zeros(14)
    caps=[(2,3),(4,3),(2,4),(6,4)][case%4]
    payload=np.concatenate([a.ravel() for a in sites])
    assert fn(n,d,bonds,payload.view(float),bg,alpha.view(float),w.view(float),2,*caps,.025,
              output.view(float),trace)==0
    loss=1-abs(np.vdot(exact.ravel(),output))**2
    assert abs(np.linalg.norm(output)-1)<1e-12 and abs(loss-trace[10]**2)<1e-11
    dense.append(dict(case=case,true_loss=loss,charged_squared=trace[10]**2,
                      sites=int(trace[9]),cuts=int(trace[7])))
save(out,'dense.json',dense)
assert max(r['sites'] for r in dense)>0 and max(r['cuts'] for r in dense)>0

verified=read(base/'manifest.json')
for name,h in verified.items():assert H(base/name)==h
rows=[];failures=[]
timings=read(base/'timings.json')
for group,n in [('replay',12),('fresh',8)]:
    for arm in (0,1):
        ok=np.ones(n,bool);main_photons=[];charged=[]
        for off in range(0,n,4):
            for sector in (0,1):
                z=np.load(base/f'{group}_{off}_{sector}_{arm}.npz')
                ok[off:off+4]&=z['errors']==0
                for i,(stats,e,log) in enumerate(zip(z['statistics'],z['errors'],z['trace'])):
                    k=int(stats[0]);tr=log[:k+int(e==2)]
                    assert np.max(abs(np.cumsum(np.r_[0.,tr[:,10]])[:len(tr)]-tr[:,1]),initial=0)<1e-11
                    assert abs(sum(tr[:k,10])-stats[5])<1e-11
                    assert np.all(tr[:,13]<1e-9) and np.all(tr[:,3]<=tr[:,2]+1e-10)
                    if sector==0:
                        main_photons.append(k);charged.append(stats[5])
                        if e==2:
                            failures.append(dict(group=group,arm=arm,record=off+i,accepted=k,
                                attempted_port=int(tr[-1,0]),prior=tr[-1,1],increment=tr[-1,10],
                                fock=tr[-1,3],schmidt=tr[-1,4],worst_cut=int(tr[-1,6]),
                                sum_fock=float(tr[:,3].sum()),sum_schmidt=float(tr[:,4].sum())))
        seconds=sum(r['seconds'] for r in timings if r['group']==group and r['adaptive']==arm)
        rows.append(dict(group=group,adaptive=arm,completed=int(ok.sum()),records=n,ok=ok.tolist(),
                         seconds=seconds,main_photons=main_photons,main_charges=charged))
save(out,'actual.json',dict(arms=rows,failures=failures))
for tag in ('site','bond'):
    z=np.load(base/f'ablation_{tag}.npz')
    assert np.isin(z['errors'],[0,2]).all() and z['statistics'][0,5]<=.1
law=np.load(targets/'law_0.npz');meta=read(targets/'row_0.json')
keys=law['keys'];full=np.r_[law['probabilities'],float(meta['total_tail']),0.]
small=[]
for arm in (0,1):
    raw=np.load(base/f'small_{arm}.npz');c=raw['counts'];failed=raw['errors']!=0
    n=len(c);assert n==16384 and c.max()<256
    packed=sum(c[:,i].astype(np.uint64)<<(8*i) for i in range(4))
    pos=np.searchsorted(keys,packed);inside=pos<len(keys);inside[inside]&=keys[pos[inside]]==packed[inside]
    labels=np.where(inside,pos,len(keys));labels[failed]=len(keys)+1
    click=sum((c[:,i]>0).astype(int)<<i for i in range(4));click[failed]=16
    delta=.001;tail=float(meta['tail_chernoff'])
    a,_=confidence(labels,full,n,4096,delta,tail)
    b,_=confidence(click,np.r_[law['click'],0.],n,4096,delta)
    pf=float(beta.ppf(1-delta,failed.sum()+1,n-failed.sum())) if not failed.all() else 1.
    small.append(dict(adaptive=arm,n=n,failed=int(failed.sum()),failure_upper=pf,pnr=a,click=b))
save(out,'quality.json',small)
save(out,'validation.json',dict(manifest_entries=len(verified),dense_cases=len(dense),
     max_squared_charge_error=max(abs(r['true_loss']-r['charged_squared']) for r in dense)))
manifest(out)
