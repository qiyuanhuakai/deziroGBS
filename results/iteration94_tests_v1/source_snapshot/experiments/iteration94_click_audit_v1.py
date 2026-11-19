"""Independent density-matrix oracle for positive conditional click compression.
Dense diagnostic inputs only: this is not a scalable Gaussian input preparation
or a head-to-head sampler benchmark. No photon occupation is sampled.
"""
import argparse
from iteration94_common_v1 import *

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
loaded,runtime=load93();dll=ROOT/'results/iteration94_click_build_v1/click.dll';lib=C.CDLL(str(dll))
setter=lib.mps93_set_lapack;setter.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p]
assert setter(*[C.cast(getattr(loaded._blas,name),C.c_void_p) for name in ('scipy_LAPACKE_zgesdd','scipy_LAPACKE_zgeqrf','scipy_LAPACKE_zungqr')])==0
fn=lib.click94_update;fn.argtypes=[C.c_int,C.c_int,C.c_int,DP,C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int]
files=[Path(__file__),ROOT/'native/gaussian_click_purification94_v1.cpp',ROOT/'native/gaussian_mps_core94_v1.cpp',dll]
save(out,'plan.json',dict(scope=__doc__,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},seed=940030))
def update(factor,d,click,maxrank):
    rows,environment=factor.shape;assert rows%d==0;rest=rows//d;rank=min(maxrank,rest)
    factor=np.ascontiguousarray(factor);output=np.zeros((rest,rank),complex);stats=np.zeros(6);err=C.create_string_buffer(4096)
    rc=fn(d,rest,environment,ptr(factor.view(float)),click,rank,ptr(output.view(float)),ptr(stats),err,len(err));assert rc==0,err.value
    return output,stats

rng=np.random.default_rng(940030);rows=[]
for d,rest,env in ((2,3,1),(3,8,2),(4,9,3),(5,16,4)):
    for trial in range(5):
        factor=rng.normal(size=(d*rest,env))+1j*rng.normal(size=(d*rest,env));factor/=np.linalg.norm(factor)
        if trial==0:factor[rest:]=0;factor/=np.linalg.norm(factor)
        if trial==1:factor[:rest]=0;factor/=np.linalg.norm(factor)
        block=factor.reshape(d,rest,env)
        for click in (0,1):
            density=sum(x@x.conj().T for x in (block[1:] if click else block[:1]));prob=float(np.trace(density).real)
            for rank in (1,2,rest):
                result,stats=update(factor,d,click,rank);assert abs(stats[0]-prob)<3e-14
                if prob==0:assert not result.any() and stats[1]==0;distance=0.;mineig=0.
                else:
                    density/=prob;actual=result@result.conj().T;distance=float(np.sum(abs(np.linalg.eigvalsh(actual-density)))/2);mineig=float(np.linalg.eigvalsh(actual).min())
                    assert mineig>-2e-14 and abs(np.trace(actual)-1)<3e-14
                    assert abs(distance-stats[1])<8e-14,(d,rest,env,trial,click,rank,distance,stats)
                    assert max(stats[4:])<3e-14
                rows.append(dict(d=d,rest=rest,environment=env,trial=trial,click=click,rank=rank,probability=prob,trace_distance=distance,charged=float(stats[1]),minimum_eigenvalue=mineig))
trees=[]
for modes,d,env in ((3,3,2),(4,3,3),(4,4,1)):
    factor=rng.normal(size=(d**modes,env))+1j*rng.normal(size=(d**modes,env));factor/=np.linalg.norm(factor)
    reference=np.zeros(2**modes)
    for occupations in np.ndindex((d,)*modes):
        address=np.ravel_multi_index(occupations,(d,)*modes);key=0
        for occupation in occupations:key=2*key+int(occupation>0)
        reference[key]+=float(np.sum(abs(factor[address])**2))
    for rank in (1,2,d**(modes-1)):
        nodes=[(0,1.,factor)];bound=0.
        for level in range(modes):
            children=[]
            for prefix,weight,state in nodes:
                for click in (0,1):
                    child,stats=update(state,d,click,rank);mass=weight*stats[0]
                    if mass>0:children.append((prefix*2+click,mass,child));bound+=mass*stats[1]
            nodes=children
        actual=np.zeros_like(reference)
        for prefix,weight,_ in nodes:actual[prefix]=weight
        distance=float(np.sum(abs(actual-reference))/2);assert abs(actual.sum()-1)<3e-14 and distance<=bound+1e-13
        if rank>=d**(modes-1):assert distance<3e-14
        trees.append(dict(modes=modes,d=d,environment=env,rank=rank,full_click_tv=distance,weighted_compression_bound=bound))
        np.savez_compressed(out/f'tree_{modes}_{d}_{rank}.npz',input_factor=factor,reference=reference,actual=actual)
report=dict(passed=True,branch_checks=len(rows),complete_tree_checks=len(trees),maximum_trace_identity_error=max(abs(r['trace_distance']-r['charged']) for r in rows),rows=rows,trees=trees)
save(out,'validation.json',report);manifest(out);print(json.dumps({k:v for k,v in report.items() if k not in ('rows','trees')}),flush=True)
