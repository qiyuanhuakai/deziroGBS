"""Rank-two output-mode scaling, no output Fock table. Fixed4-photon core.
Three serial repeats share control streams; these are timings, not quality N.
The 1152-output task still has ONLY TWO squeezed latent modes.
"""
import argparse,statistics
from iteration92_common_v2 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);lib,runtime=load();N=4096;r=.3;k=4;modes=[2,4,6,16,64,1152];save(out,'plan.json',dict(scope=__doc__,records=N,r=r,k=k,modes=modes,threads=[1,16],repetitions=3,source_sha256=H(Path(__file__))));rng=np.random.default_rng(920070);inputs={m:np.linalg.qr(rng.normal(size=(m,2))+1j*rng.normal(size=(m,2)))[0] for m in modes};rows=[]
for m,u in inputs.items():np.savez_compressed(out/f'input_{m}.npz',u=u)
for rep in range(3):
    for m in modes[rep:]+modes[:rep]:
        for threads in (1,16) if rep%2==0 else (16,1):
            start=time.perf_counter();raw,meta=sample(lib,inputs[m],fock_core(k),r,N,0x92007000+m,threads=threads);bits=(raw['counts']>0).astype(np.uint8);elapsed=time.perf_counter()-start;assert meta['returncode']==0,meta;np.savez_compressed(out/f'raw_{m}_{threads}_{rep}.npz',**raw);row=dict(modes=m,threads=threads,rep=rep,seconds=elapsed,records=N,maximum_core_degree=float(raw['statistics'][:,2].max()),**{k:v for k,v in meta.items() if k!='seconds'});rows.append(row);save(out,f'row_{m}_{threads}_{rep}.json',row);print(row,flush=True)
medians=[dict(modes=m,threads=t,seconds=statistics.median(row['seconds'] for row in rows if row['modes']==m and row['threads']==t)) for m in modes for t in (1,16)];save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,scope=__doc__));manifest(out)
