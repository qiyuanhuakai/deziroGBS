"""Independent dense local-map/Brent oracle for the changed waiting solver.
This is an arithmetic consistency gate, not a whole-distribution certificate.
"""
import argparse,math
from scipy.optimize import brentq
from iteration94_common_v2 import *
from iteration93_identity_v2 import local_map

def act(tensor,op,axis):
    return np.moveaxis(np.tensordot(op,tensor,axes=(1,axis)),0,axis)

def flow(core,t,alpha,eta):
    state=core.copy();weight=1.;nt=[];na=[]
    for axis in range(len(t)):
        m,f,tp,ap=local_map(core.shape[axis],t[axis],alpha[axis],eta)
        state=act(state,m,axis);weight*=abs(f)**2;nt.append(tp);na.append(ap)
    norm=float(np.vdot(state,state).real)
    return norm*weight,(state/np.sqrt(norm) if norm>0 else np.zeros_like(state)),np.array(nt),np.array(na)

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
loaded,runtime=load93();dll=ROOT/'results/iteration94_wait_probe_build_v2/probe.dll';lib=C.CDLL(str(dll))
setter=lib.mps93_set_lapack;setter.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p]
assert setter(*[C.cast(getattr(loaded._blas,name),C.c_void_p) for name in ('scipy_LAPACKE_zgesdd','scipy_LAPACKE_zgeqrf','scipy_LAPACKE_zungqr')])==0
fn=lib.mps94_wait_probe;fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_double,DP,DP,DP,DP,C.c_char_p,C.c_int]
files=[Path(__file__),ROOT/'native/gaussian_wait_probe94_v2.cpp',ROOT/'native/gaussian_mps_core94_v2.cpp',ROOT/'experiments/iteration93_identity_v2.py',dll]
save(out,'plan.json',dict(scope=__doc__,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},seed=940020))
rng=np.random.default_rng(940020);cases=[]
for n,d in ((1,6),(2,4),(3,3),(4,3)):
    for k in range(3):
        core=rng.normal(size=(d,)*n)+1j*rng.normal(size=(d,)*n);core=np.ascontiguousarray(core/np.linalg.norm(core))
        t=np.ascontiguousarray(rng.uniform(-.6,.6,n));alpha=np.ascontiguousarray(.3*(rng.normal(size=n)+1j*rng.normal(size=n)))
        cases.append((f'entangled_{n}_{k}',core,t,alpha))
for amp in (.01,.5,4.,15.):
    core=np.zeros(6,complex);core[0]=1;cases.append((f'coherent_{amp}',core,np.zeros(1),np.array([amp+.1j])))
core=np.zeros(6,complex);core[5]=1;cases.append(('fock5',core,np.zeros(1),np.zeros(1,complex)))
rows=[]
for index,(name,core,t,alpha) in enumerate(cases):
    n=len(t);d=core.shape[0]
    with np.errstate(invalid='ignore',divide='ignore'):vac=flow(core,t,alpha,0.)[0]
    uniforms=[vac+(1-vac)*fraction for fraction in (1e-5,.1,.5,.99,1-1e-8)]
    if vac>0:uniforms.insert(0,.5*vac)
    for j,u in enumerate(uniforms):
        if u==0 or u==1:continue
        actual=np.zeros_like(core);nt=np.zeros(n);na=np.zeros(n,complex);stats=np.zeros(18);err=C.create_string_buffer(4096)
        rc=fn(n,d,ptr(core.view(float)),ptr(t),ptr(alpha.view(float)),u,ptr(actual.view(float)),ptr(nt),ptr(na.view(float)),ptr(stats),err,len(err));assert rc==0,(name,u,err.value)
        found=bool(u>vac);assert bool(stats[1])==found,(name,u,vac,stats)
        if found:
            eta=brentq(lambda x:flow(core,t,alpha,x)[0]-u,0.,1.,xtol=1e-15,rtol=1e-14)
            _,expected,expected_t,expected_a=flow(core,t,alpha,eta)
            state_error=float(np.linalg.norm(actual-expected));parameter_error=float(max(np.max(abs(nt-expected_t)),np.max(abs(na-expected_a))))
            assert state_error<2e-7 and parameter_error<2e-7,(name,u,state_error,parameter_error)
            if name.startswith('coherent'):
                analytic=1+math.log(u)/float(np.vdot(alpha,alpha).real);assert abs(eta-analytic)<2e-12,(name,u,eta,analytic)
            elif name=='fock5':assert abs(eta-u**.2)<2e-12
        else:
            eta=None;state_error=float(np.linalg.norm(actual-core));parameter_error=float(max(np.max(abs(nt-t)),np.max(abs(na-alpha))));assert max(state_error,parameter_error)<1e-12
        rows.append(dict(case=name,uniform=float(u),vacuum=float(vac),found=found,eta_reference=eta,state_error=state_error,parameter_error=parameter_error,iterations=int(stats[2]),residual=float(stats[4])))
        np.savez_compressed(out/f'case_{index}_{j}.npz',core=core,t=t,alpha=alpha,uniform=u,actual=actual,next_t=nt,next_alpha=na,statistics=stats)
report=dict(passed=True,checks=len(rows),maximum_state_error=max(r['state_error'] for r in rows),maximum_parameter_error=max(r['parameter_error'] for r in rows),rows=rows)
save(out,'validation.json',report);manifest(out);print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
