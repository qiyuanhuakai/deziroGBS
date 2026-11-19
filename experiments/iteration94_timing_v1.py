"""Three strictly serial repetitions, same input/count, CPU1 and native CPU16.
Warm prepared-model timing includes output allocation, random generation and
sampling; excludes compilation, imports, model preparation and artifact I/O.
External preparation times are separately recorded. Frugal ABI internal layout and target construction ARE included; this is not pure cached sampling cost. Timing controls are not quality N.
The direct-click wrapper batches identical purpose5 words in C++, verified
against its original quality wrapper before any timings.
"""
import argparse
from iteration93_baseline_common_v1 import *
from iteration94_common_v2 import load93,sample93
from iteration94_branch_common_v1 import load_branch94,branch_sample94,gaussian94
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);quality=read(ROOT/'research/iteration94_quality_plan_v1.json');jobs=[]
for job in quality['jobs']:
    for threads in ([1,16] if job['family'] in ('mps','branch','direct') else [1]):jobs.append(dict(job,threads=threads,timing_name=job['name']+f'_t{threads}'))
N=2048;save(out,'plan.json',dict(scope=__doc__,jobs=jobs,records=N,repetitions=3,experiment_rule='0x94050000+rep*256+job.id; threads replay same control records',seed_rule='0x94060000+rep*256+job.id',hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),ROOT/'experiments/iteration93_baseline_common_v1.py',ROOT/'results/iteration94_build_v2/mps.dll',ROOT/'results/iteration94_branch_build_v1/branch.dll',ROOT/'native/gaussian_click_branch94_v1.cpp',ROOT/'native/gaussian_mps_core94_v2.cpp',ROOT/'experiments/iteration94_branch_common_v1.py',ROOT/'results/iteration93_rng_batch_build_v1/rng.dll')}));lib,runtime=load93();branchlib,branchruntime=load_branch94();save(out,'backend.json',lib._backend);rnglib=C.CDLL(str(ROOT/'results/iteration93_rng_batch_build_v1/rng.dll'));rng=rnglib.rng93_uniforms;rng.argtypes=[C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,DP,C.c_char_p,C.c_int];assert read(ROOT/'results/iteration93_rng_batch_build_v1/validation.json')['passed'];models={};inputs={};branch_inputs={};rows=[]
def run(job,N,experiment,seed):
    q=inputs[job['case']];name=job['timing_name'];start=time.perf_counter()
    if job['family']=='mps':
        raw,meta=sample93(lib,q['u'],q['t'],q['alpha'],N,experiment,threads=job['threads'],d=job['d'],bond=job['bond'],budget=job['budget'],cap=256,noise=q['noise']);raw['failed']=raw['errors']!=0
    elif job['family']=='branch':
        covariance,mean=branch_inputs[job['case']];raw,meta=branch_sample94(branchlib,covariance,mean,N,experiment,limit=job['limit'],budget=job['budget'],threads=job['threads']);raw['failed']=raw['errors']!=0
    elif job['family']=='direct':
        model=models[name];draws=np.zeros((N,len(q['u'])));error=C.create_string_buffer(4096);assert rng(experiment,0,0,N,len(q['u']),ptr(draws),error,len(error))==0,error.value;raw,meta=optical_sample(model.model,model.handle,draws,job['budget'],threads=job['threads']);raw['failed']=(raw['errors'][:,0]!=0)|(raw['statistics'][:,5]!=len(q['u']))
    else:raw,meta=models[name].run(N,0,experiment,seed)
    elapsed=time.perf_counter()-start;return raw,dict(seconds=elapsed,failures=int(raw['failed'].sum()))
try:
    for job in jobs:
        inputs.setdefault(job['case'],dict(np.load(ROOT/f"results/iteration93_reference_v2/input_{job['case']}.npz")))
        if job['family'] not in ('mps','branch'):models[job['timing_name']]=Baseline93(job,inputs[job['case']],threads=job['threads']);save(out,f"setup_{job['timing_name']}.json",models[job['timing_name']].meta)
        if job['family']=='branch' and job['case'] not in branch_inputs:
            setup_start=time.perf_counter();branch_inputs[job['case']]=gaussian94(inputs[job['case']]);save(out,f"setup_{job['timing_name']}.json",dict(covariance_preparation_seconds=time.perf_counter()-setup_start))
        raw,meta=run(job,16,0x94058000+job['id'],0x94068000+job['id']);save(out,f"warmup_{job['timing_name']}.json",meta)
    for rep in range(3):
        ordered=jobs[rep:]+jobs[:rep]
        for job in ordered:
            raw,meta=run(job,N,0x94050000+rep*256+job['id'],0x94060000+rep*256+job['id']);name=job['timing_name'];np.savez_compressed(out/f'raw_{rep}_{name}.npz',**raw);row=dict(repetition=rep,job=job,records=N,**meta);rows.append(row);save(out,f'row_{rep}_{name}.json',row);print(rep,name,meta,flush=True)
finally:
    for model in models.values():model.close()
summary=[]
for job in jobs:
    selected=[r for r in rows if r['job']['timing_name']==job['timing_name']];summary.append(dict(job=job,records=N,median_seconds=float(np.median([r['seconds'] for r in selected])),seconds=[r['seconds'] for r in selected],failures=[r['failures'] for r in selected]))
save(out,'summary.json',dict(completed=True,rows=summary,scope=__doc__));manifest(out)
