"""Verify checked SVD recovery and replay the one exceptional PNR record without redraw."""
import argparse
import subprocess
from iteration94_common_v2 import ROOT,Path,np,C,DP,IP,UP,ptr,H,save,manifest,load93,shutil,os,time

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
base=ROOT/'results/iteration95_adaptive_pnr_v1';floor=ROOT/'results/iteration95_rank_floor_v2'
source=ROOT/'native/gaussian_svd_checked95_v1.cpp'
input_path=ROOT/'results/iteration94_actual_quality_v1/input_0.npz'
save(out,'plan.json',dict(scope='Replay record 7 from the actual main spectrum and successful record 6; identical source inputs, addresses and 0.1 budget. Recover only positive LAPACK status on finite matrices; validate scaled reconstruction and orthogonality.',
    hashes={f.relative_to(ROOT).as_posix():H(f) for f in [Path(__file__),source,base/'adaptive.dll',input_path,
        floor/'prior_4_0_adaptive4.npz',floor/'diagnostic_prior_7_0_adaptive4.json']}))
cmd=[shutil.which('g++'),'-O3','-shared','-std=c++20','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE',
     '-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(out/'checked.dll')]
b=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=b.returncode,stderr=b.stderr))
b.check_returncode()
old,runtime=load93();backend=C.CDLL(str(base/'adaptive.dll'));shim=C.CDLL(str(out/'checked.dll'))
I,P=C.c_int,C.c_void_p
shim.svd95_setup.argtypes=[P]
shim.svd95_report.argtypes=[IP,DP];shim.svd95_report.restype=None
signature=[I,C.c_char,I,I,P,I,P,P,I,P,I]
shim.svd95_checked.argtypes=signature
def report():
    info=np.empty(3,np.int32);r=np.empty(1)
    shim.svd95_report(info.ctypes.data_as(IP),ptr(r))
    return dict(attempts=int(info[0]),recovered=int(info[1]),last_info=int(info[2]),worst=float(r[0]))
def test_matrix(matrix):
    a=np.array(matrix,complex,order='F');m,n=a.shape;k=min(m,n)
    u=np.empty((m,k),complex,order='F');vt=np.empty((k,n),complex,order='F');s=np.empty(k)
    rc=shim.svd95_checked(102,b'S',m,n,a.ctypes.data,m,s.ctypes.data,u.ctypes.data,m,vt.ctypes.data,k)
    if rc==0:assert np.linalg.norm((u*s)@vt-matrix)<1e-10*np.linalg.norm(matrix)
    return rc
@C.CFUNCTYPE(I,*signature)
def partial_overwrite(layout,job,m,n,a,lda,values,u,ldu,vt,ldvt):
    C.memset(a,0,m*n*16)
    return 1
assert shim.svd95_setup(C.cast(partial_overwrite,P))==0
rng=np.random.default_rng(9601)
for m,n in ((3,7),(11,4),(5,5)):
    assert test_matrix(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))==0
assert test_matrix(np.array([[np.nan,1],[0,1]]))==1
assert shim.svd95_setup(C.cast(old._blas.scipy_LAPACKE_zgesdd,P))==0
setter=backend.mps93_set_lapack;setter.argtypes=[P]*3
assert setter(C.cast(shim.svd95_checked,P),
    C.cast(old._blas.scipy_LAPACKE_zgeqrf,P),C.cast(old._blas.scipy_LAPACKE_zungqr,P))==0
fn=backend.mps95_sample;fn.argtypes=[I,I,DP,DP,DP,DP,I,I,C.c_double,I,I,C.c_uint64,C.c_uint,C.c_uint64,I,
    UP,DP,IP,C.c_char_p,I,I,I,C.c_double,DP]
z=np.load(input_path);u,t,alpha,noise=[np.ascontiguousarray(z[k]) for k in ('u','t','alpha','noise')]
original=np.load(floor/'prior_4_0_adaptive4.npz');rows=[]
for idx in (6,7):
    counts=np.empty((1,len(u)),np.uint32);s=np.empty((1,18));e=np.empty(1,np.int32);tr=np.empty((1,256,14))
    message=C.create_string_buffer(4096);before=report();start=time.perf_counter()
    rc=fn(len(u),len(t),ptr(u.view(float)),ptr(t),ptr(alpha.view(float)),ptr(noise),6,4,.1,256,1,0x95a00000,0,idx,1,
          counts.ctypes.data_as(UP),ptr(s),e.ctypes.data_as(IP),message,len(message),8,48,.0006,ptr(tr))
    elapsed=time.perf_counter()-start;after=report()
    assert rc==int(e[0]!=0) and s[0,5]<=.1 and int(counts.sum())==s[0,0]
    if idx==6:
        assert np.array_equal(counts[0],original['counts'][2]) and np.array_equal(e,original['errors'][2:3])
    k=int(original['statistics'][idx-4,0])
    assert np.array_equal(tr[0,:k],original['trace'][idx-4,:k])
    rows.append(dict(record=idx,old_code=int(original['errors'][idx-4]),new_code=int(e[0]),seconds=elapsed,
        photons=int(s[0,0]),charge=s[0,5],new_attempts=after['attempts']-before['attempts'],
        new_recoveries=after['recovered']-before['recovered'],last_info=after['last_info'],worst_including_controls=after['worst'],
        message=message.value.decode()))
    np.savez_compressed(out/f'record_{idx}.npz',counts=counts,errors=e,statistics=s,trace=tr)
    save(out,'summary.json',rows)
manifest(out)
