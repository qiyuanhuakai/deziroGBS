"""Freeze all quality datasets and 111 confidence events before any quality draw."""
from iteration93_common_v5 import *
path=ROOT/'research/iteration93_quality_plan_v1.json';assert not path.exists();jobs=[]
for j,name in enumerate(('pure4','displaced4','pure8','lossy_tmsv')):
    configs=[(8,32,.02),(16,128,.01)] if j<2 else ([(4,8,.05),(8,32,.02)] if j==2 else [(8,32,.02),(16,64,.01)])
    for k,(d,bond,budget) in enumerate(configs):jobs.append(dict(name=f'{name}_mps{k}',case=j,family='mps',d=d,bond=bond,budget=budget))
    for depth in (3,5):jobs.append(dict(name=f'{name}_r{depth}',case=j,family='old',depth=depth))
    jobs.append(dict(name=f'{name}_pe',case=j,family='pe'))
    if j!=1:
        for cutoff in ((2,3,4) if j==0 else ((2,3,5) if j==2 else (4,))):jobs.append(dict(name=f'{name}_frugal{cutoff}',case=j,family='frugal',cutoff=cutoff))
        for budget in (4,16):jobs.append(dict(name=f'{name}_direct{budget}',case=j,family='direct',budget=budget))
events=[]
for k,job in enumerate(jobs):
    job.update(id=k,records=65536,experiment=0x93010000+k,threads=1 if job['family'] in ('pe','frugal') else 16,batch_size=512 if job['family']=='mps' else 2048)
    metrics=['click_lower','click_upper']
    if job['family'] in ('mps','old','pe'):metrics+=['total_cdf'] if job['case']==2 else ['pnr_lower','pnr_upper']
    if job['family']=='mps':metrics+=['failure_upper']
    events.extend(f"{job['name']}:{metric}" for metric in metrics)
actual=dict(name='actual_1152_dual',family='actual',records=64,experiment=0x93020000,threads=16,batch_size=16,d=6,bond=32,budgets=[.1,.01],cap=256)
events+=['actual_1152_dual:failure_upper','actual_1152_dual:total_cdf'];assert len(events)==111
files=[ROOT/'native/gaussian_mps_core93_v5.cpp',ROOT/'experiments/iteration93_common_v5.py',ROOT/'experiments/iteration93_quality_common_v1.py',ROOT/'results/iteration93_build_v5/mps.dll',ROOT/'results/iteration93_reference_v2/validation.json',Path(__file__)];assert read(ROOT/'results/iteration93_gates_v5/validation.json')['passed'];assert read(ROOT/'results/iteration93_reference_v2/validation.json')['passed']
plan=dict(scope=__doc__,jobs=jobs,actual=actual,pilot_records=4096,photon_cap=256,confidence=dict(failure_budget=.01,events=111,event_names=events),assumptions='Nominal iid bounds and fixed numerical references; domain separation is not proof of iid. All failed rows count as target-zero outcomes. No deletion/refill. Pure8 has full click and total-count CDF metrics, not full PNR TV. Actual1152 has full two-spectrum count CDF and an ideal-arithmetic truncation-plus-failure bound; no empirical full-PNR upper claim.',hashes={p.relative_to(ROOT).as_posix():H(p) for p in files});path.write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8');print(dict(jobs=len(jobs),events=len(events),records=sum(j['records'] for j in jobs)+actual['records']))
