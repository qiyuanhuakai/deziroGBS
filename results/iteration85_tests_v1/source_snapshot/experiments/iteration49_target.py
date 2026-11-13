"""Certify fourteen-mode marginal of the unchanged exact144-mode B,F model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration35_run import certificate
from iteration36_run import envelope_records
from gbslab.certified_gaussian import ExactPairingModel
from gbslab.gaussian_envelope import GaussianEnvelope
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    start=time.perf_counter();out=ROOT/'results/iteration49_target';out.mkdir(exist_ok=False)
    names=['research/iteration49_prefix_plan.md',*[f'experiments/{n}.py' for n in ('iteration49_target','iteration22_external','iteration33_run','verify_iteration34','iteration35_run','iteration36_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    for n in ('inputs.npz','case_07.audit.json','case_12.audit.json'):
        name='results/iteration35_panel/'+n;sha=digest(ROOT/name);assert sha==json.loads((ROOT/'results/iteration35_manifest.json').read_text())['artifacts'][name];deps[name]=sha
    manifest_name='results/iteration47_manifest.json';deps[manifest_name]=digest(ROOT/manifest_name);old_manifest=json.loads((ROOT/manifest_name).read_text());old_files=[f'results/iteration47_target/certificates_{i:02d}.json' for i in range(32)]
    for name in old_files:deps[name]=digest(ROOT/name);assert deps[name]==old_manifest['artifacts'][name]
    save(out/'plan.json',dict(depth=14,source_hashes=hashes,dependencies=deps,reused_shards=old_files,reused_nonempty_subsets=8191,new_nonempty_subsets=8192,control_masks=[(1<<i)-1 for i in range(1,14)]))
    arrays=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=decode(json.loads((ROOT/'results/iteration35_panel/case_07.audit.json').read_text()));model=ExactPairingModel(arrays['B'],arrays['F'])
    assert encode(certificate(model.core))==encode(base['core']) and encode(certificate(model.noise_certificate))==encode(base['noise']) and model.operator_radius==base['operator_radius'] and model.center_denominator==base['center_denominator'] and np.array_equal(model.center_numerators,np.array(base['center_numerators'],dtype=object))
    print('physical model matched',flush=True);env=GaussianEnvelope(model);timings={};tick=time.perf_counter();old_records={}
    for name in old_files:
        for record in decode(json.loads((ROOT/name).read_text())):
            key=tuple(record['modes']);assert key not in env.vacua and 13 not in key;env.vacua[key]=tuple(record['vacuum_interval'])
            if key==tuple(range(len(key))):old_records[key]=record
    assert len(env.vacua)==8192;timings['old_load_seconds']=time.perf_counter()-tick;tick=time.perf_counter();control=GaussianEnvelope(model)
    for size in range(1,14):control.vacuum(tuple(range(size)))
    controls=envelope_records(control)
    assert all(encode(r)==encode(old_records[tuple(r['modes'])]) for r in controls);save(out/'recomputed_controls.json',encode(controls));timings['control_recompute_seconds']=time.perf_counter()-tick;tick=time.perf_counter();n=14;vac=[env.vacuum(tuple(j for j in range(n) if mask>>j&1)) for mask in range(1<<n)];timings['new_subset_generation_seconds']=time.perf_counter()-tick;tick=time.perf_counter();raw=[];intervals=[]
    for clicks in range(1<<n):
        zeros=((1<<n)-1)^clicks;subset=clicks;lo=hi=F(0)
        while True:
            a,b=vac[zeros|subset]
            if subset.bit_count()%2:lo-=b;hi-=a
            else:lo+=a;hi+=b
            if not subset:break
            subset=(subset-1)&clicks
        raw.append((lo,hi));intervals.append((max(F(0),lo),min(F(1),hi)));assert intervals[-1][0]<=intervals[-1][1]
    timings['atom_inclusion_seconds']=time.perf_counter()-tick;tick=time.perf_counter();law=dict(modes=tuple(range(n)),raw_intervals=raw,intervals=intervals);save(out/'target.json',encode(dict(law=law,transfer_upper=base['transfers']['-40']['all_measurement_TV_transfer_upper'])))
    records=envelope_records(env)
    for offset in range(0,len(records),256):save(out/f'certificates_{offset//256:02d}.json',encode(records[offset:offset+256]))
    timings['write_seconds']=time.perf_counter()-tick
    assert len(records)==8192 and all(13 in r['modes'] for r in records) and all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(completed=True,enclosures=16383,new_enclosures=len(records),reused_enclosures=8191,shards=32,atoms=16384,unchanged_sources=True,unchanged_dependencies=True,timings=timings,seconds=time.perf_counter()-start));print('target complete',flush=True)
if __name__=='__main__':main()
