"""Freeze both sampling routes, direct controls and all confidence events before draws."""
from iteration94_common_v2 import *
path=ROOT/'research/iteration94_quality_plan_v1.json';assert not path.exists();jobs=[]
for case,name in enumerate(('pure4','displaced4','pure8','lossy_tmsv')):
    jobs.append(dict(name=name+'_mps',case=case,family='mps',d=8 if case==2 else 16,bond=32 if case==2 else (64 if case==3 else 128),budget=.02 if case==2 else .01))
    for limit,budget in ((4,1.),(64,.01)):jobs.append(dict(name=f'{name}_branch{limit}',case=case,family='branch',limit=limit,budget=budget))
    jobs.append(dict(name=name+'_pe',case=case,family='pe'))
    if case!=1:
        for cutoff in ((2,3,4) if case==0 else ((2,3,5) if case==2 else (4,))):jobs.append(dict(name=f'{name}_frugal{cutoff}',case=case,family='frugal',cutoff=cutoff))
        jobs.append(dict(name=name+'_direct16',case=case,family='direct',budget=16))
events=[]
for k,job in enumerate(jobs):
    job.update(id=k,records=65536,experiment=0x94010000+k,threads=1 if job['family'] in ('pe','frugal') else 16,batch_size=512 if job['family']=='mps' else 2048)
    metrics=['click_lower','click_upper']
    if job['family'] in ('mps','pe'):metrics+=['total_cdf'] if job['case']==2 else ['pnr_lower','pnr_upper']
    if job['family'] in ('mps','branch'):metrics+=['failure_upper']
    events.extend(f"{job['name']}:{metric}" for metric in metrics)
actual=dict(name='actual_1152_dual_mps',family='actual',records=64,experiment=0x94020000,threads=16,batch_size=16,d=6,bond=32,budgets=[.1,.01],cap=256)
actual_branch=dict(name='actual_1152_dual_branch',records=64,experiment=0x94020001,threads=16,batch_size=16,limit=8,budgets=[.1,.01],scope='Fail immediately when either spectrum fails; do not run secondary for records already failed in main. Preserve original global record IDs for every sampled spectrum.')
events+=['actual_1152_dual_mps:failure_upper','actual_1152_dual_mps:total_cdf','actual_1152_dual_branch:failure_lower','actual_1152_dual_branch:failure_upper'];assert len(jobs)==26 and len(events)==82
files=[Path(__file__),ROOT/'native/gaussian_mps_core94_v2.cpp',ROOT/'native/gaussian_click_branch94_v1.cpp',ROOT/'native/gaussian_click_gram94_v1.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'experiments/iteration94_native_quality_v1.py',ROOT/'experiments/iteration94_baseline_quality_v1.py',ROOT/'experiments/iteration94_actual_quality_v1.py',ROOT/'experiments/iteration94_branch_actual_quality_v1.py',ROOT/'experiments/iteration93_baseline_common_v1.py',ROOT/'results/iteration94_build_v2/mps.dll',ROOT/'results/iteration94_branch_build_v1/branch.dll',ROOT/'results/iteration93_reference_v2/validation.json']
assert read(ROOT/'results/iteration94_branch_audit_v1/validation.json')['passed'] and read(ROOT/'results/iteration94_gates_v2/validation.json')['passed'] and read(ROOT/'results/iteration94_wait_audit_v4/validation.json')['passed']
plan=dict(scope=__doc__,jobs=jobs,actual=actual,actual_branch=actual_branch,pilot_records=4096,photon_cap=256,third_party_batch_seed_rule='0x94040000 + job.id*1024 + batch_index',confidence=dict(failure_budget=.01,events=len(events),event_names=events),assumptions='Nominal joint99 conditional on iid and fixed numerical references; domain separation does not prove iid. Failures remain target-zero outcomes. Small complete-tree diagnostics are separate numerical evidence, not iid samples. No failed-record deletion or refill.',hashes={f.relative_to(ROOT).as_posix():H(f) for f in files})
with path.open('x',encoding='utf-8') as f:json.dump(plan,f,indent=2);f.write('\n')
print(dict(jobs=len(jobs),events=len(events),records=sum(j['records'] for j in jobs)+actual['records']+actual_branch['records']),flush=True)
