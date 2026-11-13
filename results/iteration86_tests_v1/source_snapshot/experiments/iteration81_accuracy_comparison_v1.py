"""Accuracy supplement on exactly the iteration80 timing inputs.
Enumerate full small click laws and the same first-eight-port marginal at 144
outputs. Marginal TV is a full-TV lower bound, not a global accuracy upper bound.
Frugal thresholds use the preserved upstream SIMD arithmetic and stated RNG law.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,ctypes as C,math,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.native_philox_click import NativePhiloxClickSampler
from gbslab.frugal_law import full_law
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
runtime=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent];handles=[os.add_dll_directory(str(p)) for p in runtime]
bins={k:ROOT/v for k,v in dict(frugal='results/iteration81_prefix_build_v1/prefix.dll',branch='results/iteration71_branch_pilot_v1/branches.dll',cpu='results/iteration72_cuda_pilot_v1/cpu.dll',gpu='results/iteration74_fused_pilot_v2/philox.dll').items()}
lib=C.CDLL(str(bins['frugal']));DP=C.POINTER(C.c_double);ptr=lambda a:a.ctypes.data_as(DP);fn=lib.fc81_tree;fn.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
save('plan.json',dict(scope=__doc__,cases=[0,1,2],prefixes=[4,8,8],frugal_cutoffs=[[1,2,3,4],[1,2,3,4,5],[1,2,3]],branch_budgets=[1,2,4,8,16,32,128],workers=16,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*bins.values(),ROOT/'src/gbslab/frugal_law.py']}))
def mobius(vac):
    law=vac[np.arange(len(vac))^(len(vac)-1)].copy()
    for bit in range((len(vac)-1).bit_length()):
        for mask in range(len(vac)):
            if mask>>bit&1:law[mask]-=law[mask^(1<<bit)]
    assert law.min()>=0 and abs(law.sum()-1)<1e-10
    return law
def coefficients(eigen,exponent,degree):
    traces=np.array([exponent*np.sum(eigen**k) for k in range(1,degree+1)],np.longdouble);h=np.zeros(degree+1,np.longdouble);h[0]=1
    for j in range(1,degree+1):h[j]=np.dot(traces[:j],h[j-1::-1])/j
    return h
def normalizer(shape,degree):
    x=np.ones(degree+1,np.longdouble)
    for j in range(1,degree+1):x[j]=x[j-1]*(np.longdouble(shape)+j-1)/j
    return x
rows=[]
for case in (0,1,2):
    d=np.load(ROOT/f'results/iteration80_timing_v3/input_{case}.npz');v,x,cov=d['v'],d['x'],d['cov'];m,n=v.shape;K=min(m,8);patterns=np.array([[(s>>j)&1 for j in range(K)] for s in range(1<<K)],np.uint8);vac=np.ones(1<<K,np.longdouble)
    for mask in range(1,1<<K):
        ids=[j for j in range(K) if mask>>j&1];quad=ids+[m+j for j in ids];matrix=(cov[np.ix_(quad,quad)]+np.eye(len(quad)))/2;sign,logdet=np.linalg.slogdet(matrix);assert sign>0;vac[mask]=np.exp(-np.longdouble(logdet)/2)
    target=mobius(vac);np.savez_compressed(OUT/f'input_{case}.npz',v=v,x=x,cov=cov,target=target,vacuum=vac,patterns=patterns)
    def record(method,budget,law,**extra):
        tv=float(abs(law-target).sum()/2);row=dict(case=case,inputs=n,outputs=m,ports=list(range(K)),scope='complete_click_law' if K==m else 'first_eight_click_marginal',method=method,budget=budget,numerical_tv=tv,normalization_error=float(abs(law.sum()-1)),**extra);rows.append(row);save(f'row_{case}_{method}_{budget}.json',row);print(row,flush=True)
    for cutoff in range(1,(4 if case==0 else 5 if case==1 else 3)+1):
        q=np.zeros((1<<K)-1);previous=q.copy();proposed=q.copy();stats=np.zeros(3);err=C.create_string_buffer(4096);start=time.perf_counter();rc=fn(m,cutoff,K,16,ptr(cov),ptr(q),ptr(previous),ptr(proposed),ptr(stats),err,len(err));assert rc==0,err.value
        tree=[dict(prefix=''.join('1' if bits>>i&1 else '0' for i in range(length)),threshold_hex=float(q[(1<<length)-1+bits]).hex()) for length in range(K) for bits in range(1<<length)];discrete=full_law(K,tree,model='uint64');continuous=full_law(K,tree,model='continuous');law=np.array([float(v) for v in discrete['distribution']],np.longdouble);claw=np.array([float(v) for v in continuous['distribution']],np.longdouble)
        np.savez_compressed(OUT/f'frugal_{case}_{cutoff}.npz',thresholds=q,previous=previous,proposed=proposed,law=law,continuous=claw,stats=stats);save(f'frugal_tree_{case}_{cutoff}.json',dict(tree=tree,reachable_nonfinite=list(discrete['reachable_nonfinite_prefixes']),reachable_saturated=list(discrete['reachable_saturated_prefixes'])));record('frugal_cpu',cutoff,law,continuous_tv=float(abs(claw-target).sum()/2),random_model_difference=float(abs(claw-law).sum()/2),official_sample_replay=True,prefix_thresholds_checked=int(stats[2]),reachable_nonfinite=len(discrete['reachable_nonfinite_prefixes']),reachable_saturated=len(discrete['reachable_saturated_prefixes']),analysis_seconds=time.perf_counter()-start)
    for budget in (1,2,4,8,16,32,128):
        with NativeBranchClickSampler(v,x,bins['branch'],branch_budget=budget,runtime_dirs=runtime) as model:law=np.array([model.prefix_probability(bits)['probability'] for bits in patterns],np.longdouble)
        assert law.min()>=0 and abs(law.sum()-1)<1e-9;np.savez_compressed(OUT/f'direct_{case}_{budget}.npz',law=law);record('direct_cpu_law',budget,law)
        if budget in (4,8):
            # Replay full native GPU paths; only the shared physical prefix enters TV.
            draws=np.full((len(patterns),m),.5);draws[:,:K]=np.where(patterns,0.,np.nextafter(1.,0.))
            with NativePhiloxClickSampler(v,x,bins['cpu'],gpu_library=bins['gpu'],branch_budget=budget,runtime_dirs=runtime) as model:r=model.replay(np.zeros((len(patterns),2*n)),draws)
            assert np.array_equal(r['clicks'][:,:K],patterns);prob=r['conditional_click_probabilities'][:,:K];glaw=np.prod(np.where(patterns,prob,1-prob),axis=1);assert np.max(abs(glaw-law))<1e-9;np.savez_compressed(OUT/f'gpu_{case}_{budget}.npz',law=glaw,conditional=prob,clicks=r['clicks']);record('direct_gpu_law',budget,glaw,maximum_cpu_difference=float(abs(glaw-law).max()))
    if case in (0,2):
        t=d['t'];z=np.longdouble(t[0])**2;shape=np.longdouble(n)/2;nb=normalizer(shape,256)*(1-z)**shape*z**np.arange(257);ratio=z*(shape+256)/257;tail=float(nb[-1]*ratio/(1-ratio));assert ratio<1
        if case==0:
            table=np.load(ROOT/'results/iteration80_ipf_pilot_v1/strong4.npz');w=table['weights'];P=table['couplings'].copy()
        else:
            base=ROOT/'results/iteration80_wide_pilot_v1';w=np.load(base/'input_2.npz')['weights'];P=np.concatenate([np.load(base/f'table_2_{first}.npz')['couplings'] for first in range(0,len(w),64)])
        P/=P.sum(axis=(1,2))[:,None,None];w=w/w.sum();marg=P.sum(axis=2);vp=np.ones(1<<K,np.longdouble);vs=vp.copy()
        for mask in range(1,1<<K):
            ids=[j for j in range(K) if mask>>j&1];remaining=v[[j for j in range(m) if j not in ids]]
            if not len(remaining):vp[mask]=(1-z)**shape;vs[mask]=vp[mask];continue
            _,singular,Vh=np.linalg.svd(remaining,full_matrices=False);f=singular[:,None]*Vh;ep=np.linalg.svd(f@f.T,compute_uv=False)**2;hp=coefficients(ep,.5,4);exact_low=(1-z)**shape*np.dot(z**np.arange(5),hp);s=1-2*marg[:,ids].sum(axis=1)+P[:,ids][:,:,ids].sum(axis=(1,2));assert s.min()>=0 and s.max()<=1
            full=np.exp(shape*(np.log1p(-z)-np.log1p(-z*s.astype(np.longdouble))));low=np.sum(nb[:5,None]*s.astype(np.longdouble)[None,:]**np.arange(5)[:,None],axis=0);vp[mask]=exact_low+np.dot(w,full-low)
            hc=coefficients(singular**2,1,512)/normalizer(n,512);vs[mask]=exact_low+np.dot(nb[5:],hc[10::2])
        for kind,vacuum in [('transport',vp),('sphere',vs)]:
            law=mobius(vacuum);np.savez_compressed(OUT/f'pnr_{case}_{kind}.npz',law=law,vacuum=vacuum);record(kind,8,law,omitted_total_tail_upper=tail if kind=='sphere' else 0.)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
