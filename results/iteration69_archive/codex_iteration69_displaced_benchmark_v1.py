from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,time,platform,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
OUT=ROOT/'.git/iteration69_displaced_benchmark_v1_20261110';OUT.mkdir(exist_ok=False)
BUILD=ROOT/'.git/iteration69_anchor_v8_20261110'
assert json.loads((BUILD/'summary.json').read_text())['passed']
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
dll=BUILD/'native.dll';gcc=Path(shutil.which('g++'));runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
def ptr(a):return a.ctypes.data_as(DP)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p]
lib.ds_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
lib.ds_sample_anchor.argtypes=[C.c_void_p,C.c_int,C.c_uint64,DP,DP,UP,DP,C.c_char_p,C.c_int]
lib.gr_set_gemm.argtypes=[C.c_void_p,C.c_void_p]
blas_path=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));blas=C.CDLL(str(blas_path));blas.scipy_openblas_set_num_threads64_.argtypes=[C.c_int];blas.scipy_openblas_set_num_threads64_(1);blas.scipy_openblas_get_num_threads64_.restype=C.c_int;assert blas.scipy_openblas_get_num_threads64_()==1
cases=[(144,50,1.4,.43,32),(1152,50,1.4,.43,32),(1024,8,.7,.65,128),(8176,8,.7,.65,128),(8176,32,.7,.65,64),(8176,64,.7,.65,32)]
sources=[Path(__file__)]+[ROOT/'.git'/name for name in ('iteration69_reset_blas_v5.cpp','iteration69_displaced_v6.cpp','iteration69_anchor_v8.cpp')]
for p in sources:shutil.copy2(p,OUT/p.name)
save('plan.json',dict(cases=cases,repetitions=3,methods={1:'Eigen full detector scan',3:'total-occupation diagonal rejection',4:'OpenBLAS ILP64 full detector scan',5:'coherent-reference plus covariance diagonal rejection'},input_seed=691001,ordering_seed=691002,sampling_seed_base=691100,source_hashes={p.name:H(p) for p in sources},dependencies={str(dll.relative_to(ROOT)):H(dll),str((BUILD/'summary.json').relative_to(ROOT)):H(BUILD/'summary.json')},blas=dict(path=str(blas_path),sha256=H(blas_path),threads=1),environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),scope='Three interleaved repetitions with fresh input seed and common per-repetition sampling seed. Same displaced reset model; selectors generally have different paths. All use identical native binary/compiler flags and one BLAS thread. Timing includes per-trajectory noise, coherent-anchor setup, waiting inversions and output writing; QR/model preparation recorded separately. First two fixtures match Jiuzhang3 mode/input count and representative r/eta, NOT its measured matrix, source nonuniformity or imperfect detector model. 1152-mode fixture uses ideal balanced 1-to-8 fanout of the SAME 144-port isometry. No equal-quality quantum advantage claim.'))
rng=np.random.default_rng(691001);order_rng=np.random.default_rng(691002);rows=[];error=C.create_string_buffer(4096);base_v=None
for case,(m,n,r,eta,count) in enumerate(cases):
    start=time.perf_counter()
    if case==1:v=np.ascontiguousarray(np.repeat(base_v,8,axis=0)/np.sqrt(8))
    else:
        q,rr=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)),mode='reduced');v=np.ascontiguousarray(q*(np.diag(rr)/abs(np.diag(rr)))[None,:])
    qr_seconds=time.perf_counter()-start
    if case==0:base_v=v.copy()
    vm=.5*(eta*np.exp(-2*r)+1-eta);vp=.5*(eta*np.exp(2*r)+1-eta);u=-.5*np.log(2*vm);x0=np.ascontiguousarray(np.diag(np.r_[np.repeat(.5*np.expm1(2*u),n),np.repeat(.5*np.expm1(-2*u),n)]));mean=np.zeros(2*n);variance=np.r_[np.repeat(vp-.5*np.exp(2*u),n),np.zeros(n)]
    start=time.perf_counter();handle=lib.gr_create(m,n,ptr(v),ptr(x0),error,len(error));model_seconds=time.perf_counter()-start;assert handle,error.value;lib.gr_set_gemm(handle,C.cast(blas.scipy_cblas_zgemm64_,C.c_void_p))
    np.savez_compressed(OUT/f'input_{case}.npz',v=v,x0=x0,mean=mean,variance=variance);save(f'preparation_{case}.json',dict(qr_or_fanout_seconds=qr_seconds,model_seconds=model_seconds,target_mean_photons=float(n*eta*np.sinh(r)**2)))
    def run(method,samples,seed):
        output=np.zeros((samples,m),np.uint32);stats=np.zeros(8);start=time.perf_counter()
        if method==5:rc=lib.ds_sample_anchor(handle,samples,seed,ptr(mean),ptr(variance),output.ctypes.data_as(UP),ptr(stats),error,len(error))
        else:rc=lib.ds_sample(handle,samples,seed,method,ptr(mean),ptr(variance),output.ctypes.data_as(UP),ptr(stats),error,len(error))
        elapsed=time.perf_counter()-start;return rc,output,stats,elapsed
    for method in (1,3,4,5):
        rc,output,stats,elapsed=run(method,1,691050+case);save(f'warm_{case}_{method}.json',dict(returncode=rc,error=error.value.decode(),seconds=elapsed,stats=stats.tolist()));assert rc==0,error.value
    for repetition in range(3):
        order=order_rng.permutation([1,3,4,5]).tolist();save(f'order_{case}_{repetition}.json',order)
        for position,method in enumerate(order):
            seed=691100+case*100+repetition;rc,output,stats,elapsed=run(method,count,seed);row=dict(case=case,modes=m,inputs=n,r=r,eta=eta,samples=count,repetition=repetition,position=position,method=method,seed=seed,seconds=elapsed,returncode=rc,error=error.value.decode(),stats=stats.tolist(),samples_per_second=count/elapsed,proposals_per_photon=float(stats[1]/stats[0]) if stats[0] else 0.);rows.append(row);save(f'case_{case}_rep_{repetition}_method_{method}.json',row);np.savez_compressed(OUT/f'case_{case}_rep_{repetition}_method_{method}.npz',counts=output,stats=stats);print(row,flush=True);assert rc==0,error.value
    lib.gr_destroy(handle)
summaries=[]
for case in range(len(cases)):
    medians={method:float(np.median([row['seconds'] for row in rows if row['case']==case and row['method']==method])) for method in (1,3,4,5)}
    summaries.append(dict(case=case,median_seconds=medians,openblas_over_diagonal=medians[4]/medians[3],openblas_over_anchor=medians[4]/medians[5],diagonal_over_anchor=medians[3]/medians[5]))
save('summary.json',dict(passed=all(row['returncode']==0 for row in rows),rows=rows,comparisons=summaries))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
