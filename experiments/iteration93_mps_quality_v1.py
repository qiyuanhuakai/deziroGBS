"""Fresh fixed-size general-Gaussian MPS-core quality; all failures retained."""
import argparse
from iteration93_quality_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration93_quality_plan_v1.json';plan=read(planpath);lib,runtime=load93();save(out,'backend.json',lib._backend);save(out,'plan.json',dict(quality=plan,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),planpath,ROOT/'experiments/iteration93_quality_common_v1.py',ROOT/'experiments/iteration93_common_v5.py',ROOT/'native/gaussian_mps_core93_v5.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'results/iteration93_build_v5/mps.dll')}));rows=[]
for job in [j for j in plan['jobs'] if j['family']=='mps']:
    folder=out/job['name'];folder.mkdir();q=np.load(ROOT/f"results/iteration93_reference_v2/input_{job['case']}.npz");parts=[];metadata=[];save(folder,'started.json',job)
    for offset in range(0,job['records'],job['batch_size']):
        raw,meta=sample93(lib,q['u'],q['t'],q['alpha'],min(job['batch_size'],job['records']-offset),job['experiment'],offset=offset,threads=job['threads'],d=job['d'],bond=job['bond'],budget=job['budget'],cap=plan['photon_cap'],noise=q['noise']);raw['failed']=raw['errors']!=0;np.savez_compressed(folder/f'raw_{offset}.npz',**raw);save(folder,f'row_{offset}.json',meta);parts.append(raw);metadata.append(meta)
        if offset%4096==0:print(job['name'],offset,meta,flush=True)
    raw={k:np.concatenate([part[k] for part in parts]) for k in parts[0]};result=analyze(job,raw,plan,folder);result['seconds']=sum(x['seconds'] for x in metadata);save(folder,'quality.json',result);manifest(folder);rows.append(result);save(out,'progress.json',dict(completed_jobs=len(rows),rows=rows));print('finished',job['name'],result,flush=True)
save(out,'summary.json',dict(completed=True,rows=rows));manifest(out)
