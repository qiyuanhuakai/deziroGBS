"""Independent dense audit of the positive centered detector proposal.
Check mean, variance, every port rate, exact-rate decomposition and rejection
envelope. The posterior remains the full coherent jump in production.
This is a deterministic numerical audit, not an independent sampling batch.
"""
import argparse,math
from iteration93_common_v4 import *

def act(tensor,op,axis):return np.moveaxis(np.tensordot(op,tensor,axes=(1,axis)),0,axis)
def physical(d,t,alpha):
    a=np.diag(np.sqrt(np.arange(1,d)),1);return (a+t*a.T)/math.sqrt(1-t*t)+alpha*np.eye(d)

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
loaded,runtime=load93();dll=ROOT/'results/iteration93_probe_build_v5/probe.dll';lib=C.CDLL(str(dll));setter=lib.mps93_set_lapack;setter.argtypes=[C.c_void_p]*3;assert setter(*[C.cast(getattr(loaded._blas,name),C.c_void_p) for name in ('scipy_LAPACKE_zgesdd','scipy_LAPACKE_zgeqrf','scipy_LAPACKE_zungqr')])==0
fn=lib.mps93_center_probe;fn.argtypes=[C.c_int]*3+[DP]*10+[C.c_char_p,C.c_int]
save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),ROOT/'native/gaussian_mps_probe93_v5.cpp',ROOT/'native/gaussian_mps_core93_v4.cpp',dll)},cases=40));save(out,'backend.json',loaded._backend)
rng=np.random.default_rng(930050);rows=[]
for n,d in ((1,4),(2,4),(3,3),(4,3)):
    for trial in range(10):
        m=n+2;core=rng.normal(size=(d,)*n)+1j*rng.normal(size=(d,)*n)
        if trial==0:core.fill(0);core.flat[0]=1
        core=np.ascontiguousarray(core/np.linalg.norm(core));full=np.zeros((d+1,)*n,complex);full[(slice(0,d),)*n]=core
        t=np.ascontiguousarray(rng.uniform(-.6,.6,n));alpha=np.ascontiguousarray((rng.normal(size=n)+1j*rng.normal(size=n))*(0 if trial%3==0 else (5 if trial%3==1 else .2)));u=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0])
        acted=[act(full,physical(d+1,t[i],alpha[i]),i) for i in range(n)];refmean=np.array([np.vdot(full,x) for x in acted]);centered=[x-refmean[i]*full for i,x in enumerate(acted)];refdiag=np.array([np.vdot(x,x).real for x in centered]);refcoherent=abs(u@refmean)**2;refproposal=abs(u)**2@refdiag;refvariance=np.array([np.linalg.norm(sum(u[j,i]*centered[i] for i in range(n)))**2 for j in range(m)]);refport=np.array([np.linalg.norm(sum(u[j,i]*acted[i] for i in range(n)))**2 for j in range(m)])
        mean=np.zeros(n,complex);diag=np.zeros(n);rates=np.zeros((4,m));err=C.create_string_buffer(4096)
        rc=fn(n,d,m,ptr(core.view(float)),ptr(t),ptr(alpha.view(float)),ptr(u.view(float)),ptr(mean.view(float)),ptr(diag),*[ptr(x) for x in rates],err,len(err));assert rc==0,err.value
        errors=[float(np.max(abs(x-y))) for x,y in ((mean,refmean),(diag,refdiag),(rates[0],refport),(rates[1],refcoherent),(rates[2],refvariance),(rates[3],refproposal))];decomposition=float(np.max(abs(rates[0]-rates[1]-rates[2])));envelope=float(np.min(n*rates[3]-rates[2]));isometry=float(abs(rates[2].sum()-diag.sum()));assert max(errors+[decomposition,isometry])<3e-11,(n,trial,errors,decomposition,isometry);assert envelope>-3e-11
        row=dict(modes=n,ports=m,trial=trial,errors=errors,decomposition_error=decomposition,envelope_min_slack=envelope,isometry_error=isometry,coherent_fraction=float(refcoherent.sum()/refport.sum()));rows.append(row);np.savez_compressed(out/f'case_{n}_{trial}.npz',core=core,t=t,alpha=alpha,u=u,mean=mean,diag=diag,rates=rates,reference_mean=refmean,reference_diag=refdiag,reference_rates=np.array([refport,refcoherent,refvariance,refproposal]))
save(out,'validation.json',dict(passed=True,cases=len(rows),max_error=max(max(r['errors']+[r['decomposition_error'],r['isometry_error']]) for r in rows),rows=rows));manifest(out);print(json.dumps(dict(passed=True,cases=len(rows),max_error=max(max(r['errors']) for r in rows))))
