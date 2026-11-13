"""Independent complete Fock probabilities verify conditional ordered-pair matrices."""
from pathlib import Path
import os,json,argparse,hashlib,shutil,math,functools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def formula(v,N):
    a=v.shape[1]/2;j=N/2;ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;L=np.outer(ell,ell)+abs(v@v.conj().T)**2;return ((a+j)*A+(j-1)*L)/(2*a*(a+1)*(2*j-1))
def check(v,N,states,probs,label):
    mass=float(probs.sum());probs=probs/mass;x=states.astype(float);reference=(x.T@(x*probs[:,None])-np.diag(probs@x))/(N*(N-1));expected=formula(v,N);delta=float(abs(reference-expected).max());assert delta<2e-11 and abs(expected.sum()-1)<2e-12;np.savez_compressed(OUT/f'{label}.npz',v=v,total=N,states=states,probabilities=probs,reference=reference,formula=expected);return dict(label=label,inputs=v.shape[1],outputs=len(v),total=N,states=len(states),maximum_difference=delta,original_mass=mass)
rows=[]
for case in (0,1):
    d=np.load(ROOT/f'results/iteration74_pure_quality_v1/input_{case}.npz');base=int(d['cutoff'])+1;states=np.array([(d['codes']//base**j)%base for j in range(len(d['v']))]).T;totals=states.sum(axis=1)
    for N in range(2,65,2):mask=totals==N;rows.append(check(d['v'],N,states[mask],d['target'][:-1][mask],f'archive_{case}_{N}'))
def compositions(N,m):
    if m==1:yield (N,);return
    for k in range(N+1):
        for rest in compositions(N-k,m-1):yield (k,)+rest
def hafnian(a):
    @functools.lru_cache(None)
    def f(mask):
        if not mask:return 1.+0j
        i=(mask&-mask).bit_length()-1;rest=mask^(1<<i);return sum(a[i,j]*f(rest^(1<<j)) for j in range(len(a)) if rest>>j&1)
    return f((1<<len(a))-1)
rng=np.random.default_rng(782001)
for n,m in ((3,5),(8,12)):
    v=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0];A=v@v.T
    for N in (2,4,6):
        states=np.array(list(compositions(N,m)),np.int32);probs=[];j=N//2;normal=math.prod((n/2+k)/(k+1) for k in range(j))
        for state in states:
            ids=np.repeat(np.arange(m),state);probs.append(abs(hafnian(A[np.ix_(ids,ids)]))**2/math.prod(math.factorial(int(k)) for k in state)/normal)
        probs=np.array(probs);assert abs(probs.sum()-1)<1e-11;rows.append(check(v,N,states,probs,f'random_{n}_{N}'))
save('validation.json',dict(passed=True,rows=rows,checks=len(rows)));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}));print('complete pair-matrix checks',len(rows),flush=True)
