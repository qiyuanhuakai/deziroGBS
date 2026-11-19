"""Recompute saved control identities and gate provenance before a research checkpoint."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
out=ROOT/'results/iteration94_checkpoint_verification_v2';out.mkdir(exist_ok=False)
checks=0
for folder in (ROOT/'results').glob('iteration94*'):
    if not folder.is_dir() or folder==out:continue
    if (folder/'plan.json').exists():
        for name,digest in read(folder/'plan.json').get('hashes',{}).items():assert H(ROOT/name)==digest,name;checks+=1
    if (folder/'manifest.json').exists():
        for name,digest in read(folder/'manifest.json').items():assert H(folder/name)==digest,(folder,name);checks+=1
pilot=ROOT/'results/iteration94_wait_pilot_v2';summary=read(pilot/'summary.json');plan=read(pilot/'plan.json');domains=set()
for case,j in zip(plan['cases'],range(len(plan['cases']))):
    for record in range(case['records']):
        for purpose in (1,2,3):
            key=(plan['experiment_base']+j,0,record,purpose);assert key not in domains;domains.add(key)
    old=np.load(pilot/f'raw_0_{j}_old.npz');new=np.load(pilot/f'raw_0_{j}_new.npz')
    for key in ('counts','errors'):assert np.array_equal(old[key],new[key]);checks+=1
    assert np.array_equal(old['statistics'][:,14:17],new['statistics'][:,14:17]);checks+=1
    medians={}
    for impl in ('old','new'):
        raw=np.load(pilot/f'raw_0_{j}_{impl}.npz');seconds=[]
        for rep in range(3):
            repeated=np.load(pilot/f'raw_{rep}_{j}_{impl}.npz')
            for key in raw.files:assert np.array_equal(raw[key],repeated[key]);checks+=1
            row=read(pilot/f'row_{rep}_{j}_{impl}.json');assert row['root_iterations']==int(raw['statistics'][:,2].sum());assert row['failures']==int(np.count_nonzero(raw['errors']));seconds.append(row['seconds']);checks+=2
        medians[impl]=float(np.median(seconds))
    stored=summary['comparisons'][j];assert medians==stored['median_seconds'];assert medians['old']/medians['new']==stored['old_over_new'];checks+=2
for folder,countfield,count in [('iteration94_gates_v2','tests',8),('iteration94_wait_audit_v4','checks',101),('iteration94_click_audit_v2','branch_checks',120)]:
    report=read(ROOT/'results'/folder/'validation.json');assert report['passed'] and report[countfield]==count;checks+=1
click=read(ROOT/'results/iteration94_click_audit_v2/validation.json')
for row in click['trees']:
    data=np.load(ROOT/f"results/iteration94_click_audit_v2/tree_{row['modes']}_{row['d']}_{row['rank']}.npz")
    actual=float(np.sum(abs(data['actual']-data['reference']))/2)
    assert actual==row['full_click_tv'] and actual<=row['weighted_compression_bound']+1e-13;checks+=1
# Instrumentation must preserve the complete original statistical trace.
for j in range(6):
 raw=np.load(ROOT/f'results/iteration94_profile_pilot_v1/raw_{j}.npz');prior=np.load(ROOT/f'results/iteration94_wait_pilot_v1/raw_0_{j}_new.npz')
 assert np.array_equal(raw['counts'],prior['counts']) and np.array_equal(raw['errors'],prior['errors']);assert np.array_equal(raw['statistics'][:,:18],prior['statistics']);checks+=3
# Recompute Gaussian Grams from the archived selected covariance, using 80 digits.
import mpmath as mp
mp.mp.dps=80;gram_checks=0;gram_error=0.
for j in range(6):
 folder=ROOT/'results/iteration94_gram_audit_v1';data=np.load(folder/f'case_{j}.npz');row=read(folder/f'row_{j}.json');active=list(map(int,data['active']));index={mode:i for i,mode in enumerate(active)};q=data['q_active'];mu=data['mean_active'];size=len(active)
 sets=[frozenset(mode for mode in active if (int(mask[mode//64])>>(mode%64))&1) for mask in data['masks']];cache={}
 def vacuum(selected):
  if selected not in cache:
   ids=[index[mode] for mode in sorted(selected)];ids+= [k+size for k in ids]
   if not ids:value=mp.mpf(1)
   else:
    matrix=mp.matrix([[float(q[a,b]) for b in ids] for a in ids]);mean=mp.matrix([float(mu[a]) for a in ids]);value=mp.exp(-(mean.T*mp.lu_solve(matrix,mean))[0]/2)/mp.sqrt(mp.det(matrix))
   cache[selected]=float(value)
  return cache[selected]
 for a,left in enumerate(sets):
  for b,right in enumerate(sets):
   for key,selected in [('gram',left|right),('zero_gram',left|right|{row['next_mode']})]:
    difference=abs(float(data[key][a,b])-vacuum(selected));assert difference<3e-14;gram_error=max(gram_error,difference);gram_checks+=1;checks+=1
 assert len(cache)==row['vacuum_evaluations'];checks+=1
result=dict(passed=True,checks=checks,control_addresses=sum(c['records'] for c in plan['cases']),control_purpose_domains=len(domains),deliberate_replay='Same control addresses across both implementations and all three timing repeats; not additional independent quality samples.',independent_quality_records=0,gram_entries_recomputed=gram_checks,maximum_80digit_gram_error=gram_error,scope='Research checkpoint; no completed iteration94 benchmark or general-input advantage.')
(out/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
(out/'source_hash.json').write_text(json.dumps({str(Path(__file__).relative_to(ROOT)):H(Path(__file__))},indent=2)+'\n')
print(json.dumps(result),flush=True)
