"""Dense tensor oracle for core norm, all pair rates and charged trace distances.
This verifier forms full small Hilbert spaces using independent local matrices.
The production sampler never performs this expansion.
"""
import argparse,math
from iteration93_common_v2 import *
from iteration93_identity_v2 import local_map

def act(tensor,op,axis):return np.moveaxis(np.tensordot(op,tensor,axes=(1,axis)),0,axis)
def pad(core):
    shape=tuple(d+1 for d in core.shape);out=np.zeros(shape,complex);out[tuple(slice(0,d) for d in core.shape)]=core;return out
def physical(d,t,alpha):
    a=np.diag(np.sqrt(np.arange(1,d)),1);return (a+t*a.T)/math.sqrt(1-t*t)+alpha*np.eye(d)

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));dll=ROOT/'results/iteration93_probe_build_v3/probe.dll';lib=C.CDLL(str(dll));fn=lib.mps93_probe;fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_double,DP,C.c_int,C.c_double,DP,DP,DP,DP,C.c_char_p,C.c_int];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),ROOT/'native/gaussian_mps_probe93_v3.cpp',ROOT/'native/gaussian_mps_core93_v2.cpp',dll)}));rng=np.random.default_rng(930020);rows=[]
for n,d in ((1,4),(2,4),(3,3),(4,3)):
    for trial in range(3):
        core=rng.normal(size=(d,)*n)+1j*rng.normal(size=(d,)*n);core=np.ascontiguousarray(core/np.linalg.norm(core));t=np.ascontiguousarray(rng.uniform(-.5,.5,n));alpha=np.ascontiguousarray((rng.normal(size=n)+1j*rng.normal(size=n))*.15);w=np.ascontiguousarray(rng.normal(size=n)+1j*rng.normal(size=n));w/=np.linalg.norm(w);eta=(.13,.57,.89)[trial];full=pad(core);acted=[act(full,physical(d+1,t[i],alpha[i]),i) for i in range(n)];refgram=np.array([[np.vdot(x,y) for y in acted] for x in acted]);refjump=sum(w[i]*acted[i] for i in range(n));jumpnorm=float(np.vdot(refjump,refjump).real);inside=refjump[tuple(slice(0,d) for _ in range(n))];inside_norm=float(np.vdot(inside,inside).real);ref_flow=core.copy();factor=1+0j
        for i in range(n):
            M,f,_,_=local_map(d,t[i],alpha[i],eta);ref_flow=act(ref_flow,M,i);factor*=f
        refnorm=float(np.vdot(ref_flow,ref_flow).real);survival=refnorm*abs(factor)**2;ref_flow/=math.sqrt(refnorm)
        for bond in (1,16):
            flowed=np.zeros_like(core);jumped=np.zeros_like(core);gram=np.zeros((n,n),complex);stats=np.zeros(7);err=C.create_string_buffer(4096);rc=fn(n,d,ptr(core.view(float)),ptr(t),ptr(alpha.view(float)),eta,ptr(w.view(float)),bond,8.,ptr(flowed.view(float)),ptr(jumped.view(float)),ptr(gram.view(float)),ptr(stats),err,len(err));assert rc==0,err.value
            normerr=abs(stats[0]-survival);flowerr=float(np.linalg.norm(flowed-ref_flow));gramerr=float(np.max(abs(gram-refgram)));insideerr=abs(stats[1]-inside_norm);outsideerr=abs(stats[2]-(jumpnorm-inside_norm));delta=math.sqrt(max(0.,1-abs(np.vdot(refjump/math.sqrt(jumpnorm),pad(jumped)))**2));assert max(normerr,flowerr,gramerr,insideerr,outsideerr)<3e-12,(n,d,trial,bond,stats,normerr,flowerr,gramerr,insideerr,outsideerr);assert abs(delta-stats[3])<3e-12,(delta,stats)
            if bond==16:assert np.linalg.norm(jumped-inside/math.sqrt(inside_norm))<3e-12
            row=dict(modes=n,local_dimension=d,trial=trial,bond=bond,norm_error=normerr,flow_error=flowerr,gram_error=gramerr,inside_error=insideerr,outside_error=outsideerr,actual_trace_distance=delta,charged_trace_bound=float(stats[3]),canonical_residual=float(stats[4]));rows.append(row);np.savez_compressed(out/f'case_{n}_{trial}_{bond}.npz',core=core,t=t,alpha=alpha,weights=w,eta=eta,flow=flowed,jump=jumped,gram=gram,stats=stats,reference_gram=refgram,reference_flow=ref_flow,reference_jump=refjump);print(row,flush=True)
save(out,'validation.json',dict(passed=True,cases=len(rows),rows=rows,scope=__doc__));manifest(out)
