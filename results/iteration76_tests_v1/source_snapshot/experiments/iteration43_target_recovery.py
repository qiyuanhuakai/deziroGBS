"""Certify twelve-mode marginal of the unchanged exact144-mode B,F model."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,time
from pathlib import Path
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
    start=time.perf_counter();out=ROOT/'results/iteration43_target_recovery';out.mkdir(exist_ok=False)
    names=['research/iteration43_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration43_target_recovery','iteration22_external','iteration33_run','verify_iteration34','iteration35_run','iteration36_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    for n in ('inputs.npz','case_07.audit.json','case_12.audit.json'):
        name='results/iteration35_panel/'+n;sha=digest(ROOT/name);assert sha==json.loads((ROOT/'results/iteration35_manifest.json').read_text())['artifacts'][name];deps[name]=sha
    save(out/'plan.json',dict(depth=12,source_hashes=hashes,dependencies=deps))
    arrays=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=decode(json.loads((ROOT/'results/iteration35_panel/case_07.audit.json').read_text()));model=ExactPairingModel(arrays['B'],arrays['F'])
    assert encode(certificate(model.core))==encode(base['core']) and encode(certificate(model.noise_certificate))==encode(base['noise']) and model.operator_radius==base['operator_radius'] and model.center_denominator==base['center_denominator'] and np.array_equal(model.center_numerators,np.array(base['center_numerators'],dtype=object))
    print('physical model matched',flush=True);env=GaussianEnvelope(model);law=env.law(tuple(range(12)));save(out/'target.json',encode(dict(law=law,transfer_upper=base['transfers']['-40']['all_measurement_TV_transfer_upper'])))
    records=envelope_records(env)
    for offset in range(0,len(records),256):save(out/f'certificates_{offset//256:02d}.json',encode(records[offset:offset+256]))
    assert len(records)==4095 and all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(completed=True,enclosures=len(records),shards=16,atoms=4096,unchanged_sources=True,unchanged_dependencies=True,seconds=time.perf_counter()-start));print('target complete',flush=True)
if __name__=='__main__':main()
