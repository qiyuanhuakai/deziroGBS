"""Frozen fresh quality batches for PNR v2 and direct Gaussian-branch trajectories."""
import argparse
from iteration93_quality_common_v1 import *
from iteration94_common_v2 import load93,sample93
from iteration94_branch_common_v1 import load_branch94,branch_sample94,gaussian94
p=argparse.ArgumentParser();p.add_argument('--family',choices=['mps','branch'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
planpath=ROOT/'research/iteration94_quality_plan_v1.json';plan=read(planpath);lib,runtime=load93() if a.family=='mps' else load_branch94()
files=[Path(__file__),planpath,ROOT/'experiments/iteration93_quality_common_v1.py',ROOT/'experiments/iteration94_common_v2.py',ROOT/'experiments/iteration94_branch_common_v1.py',ROOT/'native/gaussian_mps_core94_v2.cpp',ROOT/'native/gaussian_click_branch94_v1.cpp',ROOT/'results/iteration94_build_v2/mps.dll',ROOT/'results/iteration94_branch_build_v1/branch.dll']
save(out,'plan.json',dict(scope=__doc__,quality=plan,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files}));rows=[]
for job in [j for j in plan['jobs'] if j['family']==a.family]:
    folder=out/job['name'];folder.mkdir();z=np.load(ROOT/f"results/iteration93_reference_v2/input_{job['case']}.npz");parts=[];metadata=[];start=time.perf_counter();q,mean=gaussian94(z);save(folder,'setup.json',dict(gaussian_covariance_seconds=time.perf_counter()-start));save(folder,'started.json',job)
    for offset in range(0,job['records'],job['batch_size']):
        N=min(job['batch_size'],job['records']-offset)
        if a.family=='mps':raw,meta=sample93(lib,z['u'],z['t'],z['alpha'],N,job['experiment'],offset=offset,threads=job['threads'],d=job['d'],bond=job['bond'],budget=job['budget'],cap=256,noise=z['noise'])
        else:raw,meta=branch_sample94(lib,q,mean,N,job['experiment'],limit=job['limit'],budget=job['budget'],offset=offset,threads=job['threads'])
        raw['failed']=raw['errors']!=0;np.savez_compressed(folder/f'raw_{offset}.npz',**raw);save(folder,f'row_{offset}.json',meta);parts.append(raw);metadata.append(meta)
        if offset%8192==0:print(job['name'],offset,meta,flush=True)
    raw={k:np.concatenate([part[k] for part in parts]) for k in parts[0]};result=analyze(job,raw,plan,folder)
    if a.family=='branch':
        delta=plan['confidence']['failure_budget']/plan['confidence']['events'];pf=failure_upper(int(raw['failed'].sum()),job['records'],delta);result.update(failure_probability_upper=pf,ideal_arithmetic_compression_tv_upper=min(1.,job['budget']+pf),maximum_charged=float(raw['statistics'][:,4].max()),scope='Gaussian projection subspace truncation model; floating point is independently tested but not interval-certified.')
    result['seconds']=sum(x['seconds'] for x in metadata);save(folder,'quality.json',result);manifest(folder);rows.append(result);save(out,'progress.json',dict(completed_jobs=len(rows),rows=rows));print('finished',job['name'],result,flush=True)
save(out,'summary.json',dict(completed=True,rows=rows));manifest(out)
