"""80-digit vacuum oracle for Gaussian purification Gram matrices.
Includes actual 1152-output inputs with sparse projection sets; no complete
1152-output sample is generated and no sampler-throughput claim is made.
"""
import argparse
from iteration94_common_v2 import *
import mpmath as mp

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));dll=ROOT/'results/iteration94_gram_build_v1/gram.dll';lib=C.CDLL(str(dll));fn=lib.gaussian94_gram
fn.argtypes=[C.c_int,DP,DP,C.c_int,C.POINTER(C.c_uint64),C.c_int,DP,DP,DP,C.c_char_p,C.c_int]
sources=[ROOT/f'results/iteration93_reference_v2/input_{j}.npz' for j in range(4)]+[ROOT/f'results/iteration93_actual_quality_v1/input_{j}.npz' for j in (0,1)]
files=[Path(__file__),ROOT/'native/gaussian_click_gram94_v1.cpp',ROOT/'native/gaussian_mps_core94_v2.cpp',dll,*sources]
save(out,'plan.json',dict(scope=__doc__,precision_digits=80,seed=940050,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files}))
mp.mp.dps=80;rng=np.random.default_rng(940050);rows=[]
for case,source in enumerate(sources):
    z=np.load(source);u=z['u'];t=z['t'];alpha=z['alpha'];noise=z['noise'];m,n=u.shape
    real=np.block([[u.real,-u.imag],[u.imag,u.real]])
    x=np.diag(np.r_[t/(1-t),-t/(1+t)])+2*noise@noise.T
    q=np.ascontiguousarray(np.eye(2*m)+real@x@real.T);mean=np.ascontiguousarray(np.sqrt(2)*np.r_[(u@alpha).real,(u@alpha).imag])
    active=list(range(m)) if m<=8 else [0,63,64,127,128,m-1]
    codes=list(range(2**m)) if m<=4 else [0]+[int(v) for v in rng.integers(1,2**len(active),15)]
    sets=[frozenset(active[b] for b in range(len(active)) if (code>>b)&1) for code in codes]
    masks=np.zeros((len(sets),(m+63)//64),np.uint64)
    for row,selected in enumerate(sets):
        for mode in selected:masks[row,mode//64]|=np.uint64(1)<<np.uint64(mode%64)
    next_mode=active[-1];G=np.zeros((len(sets),len(sets)));G0=np.zeros_like(G);stats=np.zeros(2);error=C.create_string_buffer(4096)
    start=time.perf_counter();rc=fn(m,ptr(q),ptr(mean),len(sets),masks.ctypes.data_as(C.POINTER(C.c_uint64)),next_mode,ptr(G),ptr(G0),ptr(stats),error,len(error));seconds=time.perf_counter()-start;assert rc==0,error.value
    cache={}
    def vacuum(selected):
        if selected not in cache:
            modes=sorted(selected);indices=modes+[j+m for j in modes]
            if not modes:value=mp.mpf(1)
            else:
                matrix=mp.matrix([[mp.mpf(float(q[i,j])) for j in indices] for i in indices]);mu=mp.matrix([mp.mpf(float(mean[i])) for i in indices]);value=mp.exp(-(mu.T*mp.lu_solve(matrix,mu))[0]/2)/mp.sqrt(mp.det(matrix))
            cache[selected]=float(value)
        return cache[selected]
    reference=np.array([[vacuum(a|b) for b in sets] for a in sets]);reference0=np.array([[vacuum(a|b|{next_mode}) for b in sets] for a in sets])
    difference=float(max(np.max(abs(reference-G)),np.max(abs(reference0-G0))))
    eigenvalues=[float(np.linalg.eigvalsh(matrix).min()) for matrix in (G,G0,G-G0)]
    assert difference<3e-14 and min(eigenvalues)>-2e-13,(case,difference,eigenvalues)
    previous_error=None
    if case<4:
        original=np.load(ROOT/f'results/iteration93_reference_v2/law_{case}.npz')['vacuum']
        previous_error=max(abs(v-original[sum(1<<j for j in selected)]) for selected,v in cache.items());assert previous_error<3e-14,(case,previous_error)
    indices=active+[j+m for j in active]
    np.savez_compressed(out/f'case_{case}.npz',masks=masks,active=active,q_active=q[np.ix_(indices,indices)],mean_active=mean[indices],gram=G,zero_gram=G0,reference=reference,reference_zero=reference0,stats=stats)
    row=dict(case=case,modes=m,branches=len(sets),next_mode=next_mode,native_seconds=seconds,vacuum_evaluations=int(stats[0]),largest_projection_set=int(stats[1]),gram_entries_checked=int(2*G.size),maximum_80digit_error=difference,minimum_eigenvalues=eigenvalues,prior_reference_error=None if previous_error is None else float(previous_error),dense_q_sha256=hashlib.sha256(q.tobytes()).hexdigest())
    rows.append(row);save(out,f'row_{case}.json',row);print(json.dumps(row),flush=True)
save(out,'validation.json',dict(passed=True,gram_entries_checked=sum(r['gram_entries_checked'] for r in rows),maximum_80digit_error=max(r['maximum_80digit_error'] for r in rows),scope=__doc__,rows=rows));manifest(out)
