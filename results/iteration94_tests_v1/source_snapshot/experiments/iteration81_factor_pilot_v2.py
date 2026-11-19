"""Native factorized IPF versus frozen dense tables; no L*m*m allocation.
Pilot timings are exploratory. Preserve infeasible cases and resource refusals.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';source=ROOT/'native/gaussian_pair_factor81_v1.cpp';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),source)},threads=16,tolerance=1e-10,large_input_seed=810001))
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-I',str(eigen),str(source),'-o',str(OUT/'factor.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('built factor IPF',flush=True)
directory=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'factor.dll'));DP=C.POINTER(C.c_double);ptr=lambda a:a.ctypes.data_as(DP)
fn=lib.factor81_ipf;fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_int,C.c_double,C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
design=lib.factor81_design;design.argtypes=[C.c_int,C.c_int,C.c_int,DP,DP,C.c_int,DP,DP,DP,C.c_char_p,C.c_int];design.restype=C.c_int
rows=[]
def calibrate(label,v,weights,p,Q,maximum,dense=None,wide_case=None,directions=None):
    m,n=v.shape;L=len(weights);K=np.zeros((m,m));A=np.zeros((L,m));B=np.zeros((L,m));stats=np.zeros(6);message=C.create_string_buffer(4096);weights=np.ascontiguousarray(weights);p=np.ascontiguousarray(p);Q=np.ascontiguousarray(Q)
    save('started_'+label+'.json',dict(inputs=n,outputs=m,latent=L,maximum=maximum));start=time.perf_counter();rc=fn(m,L,ptr(weights),ptr(p),ptr(Q),maximum,1e-10,16,ptr(K),ptr(A),ptr(B),ptr(stats),message,len(message));elapsed=time.perf_counter()-start
    row=dict(label=label,inputs=n,outputs=m,latent=L,success=rc==0,error=message.value.decode() if rc else '',seconds=elapsed,iterations=int(stats[0]),residual=stats[1],symmetric_residual=stats[2],factor_payload_bytes=int(stats[3]),base_arrays_bytes=int(stats[4]),dense_equivalent_bytes=L*m*m*8,cell_iterations=stats[5])
    save('calibration_'+label+'.json',row)
    if L>8192:
        np.savez_compressed(OUT/('input_'+label+'.npz'),v=v,weights=weights,Q=Q,K=K)
        for first in range(0,L,256):np.savez_compressed(OUT/f'inputchunk_{label}_{first}.npz',marginals=p[first:first+256],directions=directions[first:first+256])
    else:np.savez_compressed(OUT/('input_'+label+'.npz'),v=v,weights=weights,marginals=p,Q=Q,K=K,**({} if directions is None else dict(directions=directions)))
    max_difference=0.;maximum_row_error=0.;averaged=np.zeros((m,m));symmetric_marginal=np.zeros((L,m));minimum=float('inf')
    if L>8192 and rc==0:
        forward=A*(B@K.T);backward=B*(A@K);symmetric_marginal=.5*(forward+backward);average=K*(A.T@(B*weights[:,None]));averaged=.5*(average+average.T);maximum_row_error=float(abs(symmetric_marginal-p).max());minimum=0.
        assert K.min()>=0 and A.min()>=0 and B.min()>=0
    for first in range(0,L,256):
        last=min(L,first+256);aa=A[first:last];bb=B[first:last];np.savez_compressed(OUT/f'factors_{label}_{first}.npz',left=aa,right=bb)
        if rc==0 and L<=8192:
            # Independent bounded chunks, never the full 3D table.
            P=K[None,:,:]*aa[:,:,None]*bb[:,None,:];P=.5*(P+P.transpose(0,2,1));minimum=min(minimum,float(P.min()));averaged+=np.einsum('l,lij->ij',weights[first:last],P);marginal=P.sum(axis=2);symmetric_marginal[first:last]=marginal;maximum_row_error=max(maximum_row_error,float(abs(marginal-p[first:last]).max()))
            if dense is not None:max_difference=max(max_difference,float(abs(P-dense[first:last]).max()))
            if wide_case is not None:
                original=np.concatenate([np.load(ROOT/f'results/iteration80_wide_pilot_v1/table_{wide_case}_{part}.npz')['couplings'] for part in range(first,last,64)]);max_difference=max(max_difference,float(abs(P-original).max()))
    if rc==0:
        ell=(abs(v)**2).sum(axis=1);R=(np.outer(ell,ell)+abs(v@v.conj().T)**2+abs(v@v.T)**2)/(n*(n+2));cross=symmetric_marginal.T@(symmetric_marginal*weights[:,None]);pair_error=float(abs(averaged-Q).max());cross_error=float(abs(cross-R).max());assert minimum>=0 and max(maximum_row_error,pair_error,cross_error)<1e-8
        row.update(minimum_probability=minimum,independent_row_error=maximum_row_error,independent_pair_error=pair_error,independent_cross_error=cross_error,maximum_difference_from_dense=max_difference if dense is not None or wide_case is not None else None);np.savez_compressed(OUT/('moments_'+label+'.npz'),averaged_pair=averaged,cross_pair=cross,target_cross=R,**({} if L>8192 else dict(symmetric_marginals=symmetric_marginal)))
        if dense is not None or wide_case is not None:assert max_difference<1e-8
    rows.append(row);save('row_'+label+'.json',row);print(row,flush=True)
base=ROOT/'results/iteration80_ipf_pilot_v1'
for old in json.loads((base/'summary.json').read_text())['rows']:
    label=old['label'];d=np.load(base/(label+'.npz'));calibrate(label,d['v'],d['weights'],d['marginals'],d['Q'],256 if old['success'] else 1000,dense=d['couplings'] if old['success'] else None)
for case in range(4):
    d=np.load(ROOT/f'results/iteration80_wide_pilot_v1/input_{case}.npz');calibrate('wide'+str(case),d['v'],d['weights'],d['marginals'],d['Q'],64,wide_case=case)
n,m=128,384;L=n+256**2;rng=np.random.default_rng(810001);v=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0];u=np.zeros((L,n));weights=np.zeros(L);marginals=np.zeros((L,m));vr=np.ascontiguousarray(v.real);vi=np.ascontiguousarray(v.imag);message=C.create_string_buffer(4096);start=time.perf_counter();rc=design(n,m,L,ptr(vr),ptr(vi),16,ptr(u),ptr(weights),ptr(marginals),message,len(message));save('large_design.json',dict(returncode=rc,error=message.value.decode(),seconds=time.perf_counter()-start));assert rc==0,message.value
Q=abs(v@v.T)**2/n;calibrate('large128',v,weights,marginals,Q,32,directions=u)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
