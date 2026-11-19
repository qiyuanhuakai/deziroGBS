"""Execute a selected fixed baseline family, strictly serial across jobs.
PE/Frugal batch seeds = 0x93040000 + job.id*1024 + batch index, unique within
the frozen 33-job plan. Native methods use the plan's Philox domains directly.
Third-party seeded PRNGs retain their declared nominal iid modeling assumption.
"""
import argparse
from iteration93_baseline_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--family',choices=['old','direct','frugal','pe'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration93_quality_plan_v1.json';plan=read(planpath);assert read(ROOT/'results/iteration93_baseline_pilot_v1/validation.json')['passed'];files=[Path(__file__),planpath,ROOT/'experiments/iteration93_baseline_common_v1.py',ROOT/'experiments/iteration93_quality_common_v1.py'];save(out,'plan.json',dict(scope=__doc__,family=a.family,quality=plan,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));rows=[]
for job in [j for j in plan['jobs'] if j['family']==a.family]:
    folder=out/job['name'];folder.mkdir();q=np.load(ROOT/f"results/iteration93_reference_v2/input_{job['case']}.npz");model=Baseline93(job,q);parts=[];metadata=[];save(folder,'setup.json',model.meta)
    try:
        for offset in range(0,job['records'],job['batch_size']):
            seed=0x93040000+job['id']*1024+offset//job['batch_size'];raw,meta=model.run(min(job['batch_size'],job['records']-offset),offset,job['experiment'],seed);np.savez_compressed(folder/f'raw_{offset}.npz',**raw);save(folder,f'row_{offset}.json',meta);parts.append(raw);metadata.append(meta)
            if offset%8192==0:print(job['name'],offset,meta,flush=True)
    finally:model.close()
    keys=('counts','failed') if 'counts' in parts[0] else ('clicks','failed');raw={k:np.concatenate([part[k] for part in parts]) for k in keys};result=analyze(job,raw,plan,folder);result['seconds']=sum(x['wall_seconds_including_rng'] for x in metadata);save(folder,'quality.json',result);manifest(folder);rows.append(result);save(out,'progress.json',dict(completed_jobs=len(rows),rows=rows));print('finished',result,flush=True)
save(out,'summary.json',dict(completed=True,rows=rows));manifest(out)
