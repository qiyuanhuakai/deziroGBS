"""Frozen64 actual dual-spectrum attempts with explicit early failure outcomes.
Main-spectrum failures terminate that record; secondary IDs are never repacked.
Failure probability bounds concern all64 attempts, not successful subsets.
"""
import argparse
from scipy.stats import beta
from iteration94_branch_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration94_quality_plan_v1.json';plan=read(planpath);job=plan['actual_branch'];lib,runtime=load_branch94()
files=[Path(__file__),planpath,ROOT/'experiments/iteration94_branch_common_v1.py',ROOT/'native/gaussian_click_branch94_v1.cpp',ROOT/'results/iteration94_branch_build_v1/branch.dll'];save(out,'plan.json',dict(job=job,quality_family=plan['confidence'],scope=__doc__,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files}))
inputs=[]
for sector in (0,1):
    source=ROOT/f'results/iteration93_actual_quality_v1/input_{sector}.npz';start=time.perf_counter();q,mean=gaussian94(np.load(source));inputs.append((q,mean));save(out,f'setup_{sector}.json',dict(source=source.relative_to(ROOT).as_posix(),source_sha256=H(source),seconds=time.perf_counter()-start,q_sha256=hashlib.sha256(q.tobytes()).hexdigest()))
parts=[];metadata=[]
for offset in range(0,job['records'],job['batch_size']):
    print('starting main',offset,flush=True);q,mean=inputs[0];raw,meta=branch_sample94(lib,q,mean,min(job['batch_size'],job['records']-offset),job['experiment'],limit=job['limit'],budget=job['budgets'][0],sector=0,offset=offset,threads=job['threads']);np.savez_compressed(out/f'raw_0_{offset}.npz',**raw);save(out,f'row_0_{offset}.json',meta);parts.append(raw);metadata.append(meta);print(meta,flush=True)
main={k:np.concatenate([part[k] for part in parts]) for k in parts[0]};combined=main['clicks'].copy();failed=main['errors']!=0;secondary_attempted=np.zeros(job['records'],bool)
for record in np.flatnonzero(~failed):
    q,mean=inputs[1];secondary_attempted[record]=True;raw,meta=branch_sample94(lib,q,mean,1,job['experiment'],limit=job['limit'],budget=job['budgets'][1],sector=1,offset=int(record),threads=1);np.savez_compressed(out/f'raw_1_{record}.npz',**raw);save(out,f'row_1_{record}.json',meta);metadata.append(meta);combined[record]|=raw['clicks'][0];failed[record]|=raw['errors'][0]!=0
N=job['records'];k=int(failed.sum());delta=plan['confidence']['failure_budget']/plan['confidence']['events'];lower=0. if k==0 else float(beta.ppf(delta,k,N-k+1));upper=1. if k==N else float(beta.ppf(1-delta,k+1,N-k));np.savez_compressed(out/'combined.npz',clicks=combined,failed=failed,secondary_attempted=secondary_attempted)
report=dict(completed=True,job=job,failures=k,completed_samples=N-k,secondary_attempted=int(secondary_attempted.sum()),failure_probability_lower=lower,failure_probability_upper=upper,full_click_tv_lower_from_failure=lower,seconds=sum(r['seconds'] for r in metadata),scope=__doc__+'No successful-sample throughput is claimed from partial or failed records.')
save(out,'summary.json',report);manifest(out);print(json.dumps(report),flush=True)
