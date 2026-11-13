"""Serial finite-core cold/warm timings; table construction and sampling separated.
These 3/4-mode conditional tasks do not represent full1152-mode GBS throughput.
Prepared explicit uniforms are excluded; cold builds one whole cutoff table per
call, warm uses the same fixed table. Python allocation and C++ work included.
"""
import argparse,statistics
from iteration91_fock_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration91_fock_build_v1/fock.dll';base=ROOT/'results/iteration91_fock_quality_v1';files=[Path(__file__),dll,base/'summary.json',*[base/f'input_{j}.npz' for j in (0,1)]];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},records=[1,64,65536],repetitions=3));runtime=os.add_dll_directory(str(runtime_dirs()[0]));lib=C.CDLL(str(dll));rows=[];setup=[]
for case,(r,cut) in enumerate(((.6,64),(.3,32))):
 u=np.load(base/f'input_{case}.npz')['u']
 for k in (4,6):
  keys,p0,meta=core_law(lib,u,k,r,cut);assert meta['returncode']==0;setup.append(dict(case=case,k=k,patterns=len(keys),**meta));draws=uniforms(0x91005001+case,0,0,65536,1).ravel()
  for rep in range(3):
   for N in (1,64,65536):
    for cache in (('cold','warm') if rep%2==0 else ('warm','cold')):
     start=time.perf_counter()
     if cache=='cold':kk,pp,mm=core_law(lib,u,k,r,cut)
     else:kk,pp=keys,p0;mm=meta
     values,fails,sm=core_draw(lib,kk,pp,draws[:N]);seconds=time.perf_counter()-start;assert mm['returncode']==sm['returncode']==0;row=dict(case=case,k=k,cache=cache,repetition=rep,records=N,seconds=seconds,failed=int(fails.sum()));rows.append(row)
medians=[dict(case=case,k=k,cache=cache,records=N,median_seconds=statistics.median(r['seconds'] for r in rows if (r['case'],r['k'],r['cache'],r['records'])==(case,k,cache,N))) for case in (0,1) for k in (4,6) for cache in ('cold','warm') for N in (1,64,65536)]
save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,setup=setup,scope=__doc__));manifest(out);print(json.dumps(medians,indent=2),flush=True)
