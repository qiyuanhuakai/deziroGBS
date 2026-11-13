"""Native coupling calibration beyond four modes using finite real 4-designs.
Binary-field signs avoid exponential hypercube enumeration at n>=16.
All support/workspace/calibration failures remain explicit and archived.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time,ctypes as C,itertools,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_wide_transport_alias_pnr import NativeWideTransportAliasPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__),ROOT/'src/gbslab/native_wide_transport_alias_pnr.py',*sorted((ROOT/'native').glob('*.cpp')),*sorted((ROOT/'native').glob('*.hpp'))];cases=[(8,16),(16,32),(16,144),(32,64),(128,384)];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},cases=cases,seed=800801,records=100000,fixed_total=34,workers=16,max_cell_iterations=2000000000,max_cells=32000000,tolerance=1e-10));shutil.copy2(__file__,OUT/Path(__file__).name)
for kind,source in [('ipf','gaussian_pair_ipf80_v2.cpp'),('transport','gaussian_pnr_transport80_v3.cpp')]:
    command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native'/source),'-o',str(OUT/f'{kind}.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save(f'build_{kind}.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('built',kind,flush=True)
handle=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'ipf.dll'));DP=C.POINTER(C.c_double);fn=lib.pair80_ipf;fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_int,C.c_double,DP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int;ptr=lambda x:x.ctypes.data_as(DP)
def design(n):
    if n<=8:signs=np.array([[1]+[1 if mask&(1<<j) else -1 for j in range(n-1)] for mask in range(1<<(n-1))],float)
    else:
        bits=n.bit_length();poly={5:0x25,6:0x43,8:0x11b}[bits];q=1<<bits
        def multiply(a,b):
            result=0
            while b:
                if b&1:result^=a
                b>>=1;a<<=1
                if a&q:a^=poly
            return result
        cubes=[multiply(multiply(x,x),x) for x in range(1,n+1)];labels=[x|(cube<<bits) for x,cube in zip(range(1,n+1),cubes)]
        for size in (1,2,3,4):
            for group in itertools.combinations(labels,size):
                value=0
                for item in group:value^=item
                assert value!=0,'4-wise sign design lost independence'
        signs=np.array([[1-2*(((a&x)^(b&cube)).bit_count()%2) for x,cube in zip(range(1,n+1),cubes)] for a in range(q) for b in range(q)],float)
    L=len(signs);return np.ascontiguousarray(np.r_[np.eye(n),signs/np.sqrt(n)]),np.r_[np.full(n,2/(n*(n+2))),np.full(L,n/((n+2)*L))]
rows=[];rng=np.random.default_rng(800801)
for case,(n,m) in enumerate(cases):
    L=n+(2**(n-1) if n<=8 else 2**(2*n.bit_length()));cells=L*m*m;upper=L*m*(m+1)//2
    if L>8192 or cells>32000000 or upper>16000000:
        row=dict(case=case,inputs=n,outputs=m,success=False,stage='preflight',error='Explicit dense calibration/table resource bound',latent=L,cells=cells,full_table_bytes=8*cells,upper_entries=upper);rows.append(row);save(f'case_{case}.json',row);print(row,flush=True);continue
    v=np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0];u,weights=design(n);p=np.ascontiguousarray(abs(u@v.T)**2);ell=(abs(v)**2).sum(axis=1);Q=np.ascontiguousarray(abs(v@v.T)**2/n);R=(np.outer(ell,ell)+abs(v@v.conj().T)**2+abs(v@v.T)**2)/(n*(n+2));cross=p.T@(p*weights[:,None]);design_error=float(abs(cross-R).max());assert design_error<1e-12 and np.max(abs(weights@p-ell/n))<1e-12
    np.savez_compressed(OUT/f'input_{case}.npz',v=v,directions=u,weights=weights,marginals=p,Q=Q,R=R);table=np.zeros((L,m,m));stats=np.zeros(4);message=C.create_string_buffer(4096);maximum=min(256,2000000000//cells);save(f'started_{case}.json',dict(case=case,maximum=maximum,cells=cells,design_error=design_error));start=time.perf_counter();rc=fn(m,L,ptr(weights),ptr(p),ptr(Q),maximum,1e-10,ptr(table),ptr(stats),message,len(message));elapsed=time.perf_counter()-start
    row=dict(case=case,inputs=n,outputs=m,success=rc==0,stage='calibration',error=message.value.decode() if rc else '',seconds=elapsed,latent=L,cells=cells,maximum_iterations=maximum,iterations=int(stats[0]),residual=stats[1],symmetric_residual=stats[2],design_error=design_error)
    for first in range(0,L,64):np.savez_compressed(OUT/f'table_{case}_{first}.npz',first=first,couplings=table[first:first+64])
    if rc==0:
        try:
            start=time.perf_counter()
            with NativeWideTransportAliasPNRSampler(v,.5,OUT/'transport.dll',couplings=table,latent_weights=weights,exact_total_budget=0,workers=16,runtime_dirs=runtime) as model:
                built=time.perf_counter()-start;r=model.sample(100000,0xae000000000000+case*2**40,fixed_total=34);alias_bound=model.alias_latent_tv+17*model.alias_maximum_pair_tv
            x=r['counts'].astype(float);pair=(x.T@x-np.diag(x.sum(axis=0)))/(100000*34*33);target=(Q+32*R)/33;row.update(stage='sample',constructor_seconds=built,sampling_seconds=r['wall_seconds'],empirical_pair_max_error=float(abs(pair-target).max()),empirical_pair_l1_half=float(abs(pair-target).sum()/2),ideal_uint64_alias_tv_bound=alias_bound,statistics=r['statistics']);assert np.all(x.sum(axis=1)==34) and row['empirical_pair_max_error']<.006;np.savez_compressed(OUT/f'counts_{case}.npz',counts=r['counts'],pair=pair,target_pair=target)
        except Exception as e:row.update(success=False,stage='sample',error=str(e))
    rows.append(row);save(f'case_{case}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
