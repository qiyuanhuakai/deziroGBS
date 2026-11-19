"""Certify full144 vacuum and four complete single-click outputs."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,gzip,hashlib,time
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
from gbslab.compact_gaussian_evidence import pack_enclosure,unpack_enclosure,semantic_digest
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    tick=time.perf_counter();out=ROOT/'results/iteration57_target';out.mkdir(exist_ok=False)
    names=['research/iteration57_overlap_plan.md','research/iteration57_interval_note.md',*[f'experiments/{n}.py' for n in ('iteration57_target','iteration22_external','iteration33_run','verify_iteration34','iteration35_run','iteration36_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]]
    hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    m=read('results/iteration35_manifest.json')
    for name in ('inputs.npz','case_07.audit.json'):
        n='results/iteration35_panel/'+name;deps[n]=digest(ROOT/n);assert deps[n]==m['artifacts'][n]
    arrays=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=read('results/iteration35_panel/case_07.audit.json');model=ExactPairingModel(arrays['B'],arrays['F']);assert encode(certificate(model.core))==encode(base['core']) and encode(certificate(model.noise_certificate))==encode(base['noise']) and tuple(base['vacuum_interval'])==model.vacuum_interval
    env=GaussianEnvelope(model);allm=tuple(range(144));vac=env.vacuum(allm);global_vac=model.vacuum_interval;assert max(vac[0],global_vac[0])<=min(vac[1],global_vac[1]);zero=(max(vac[0],global_vac[0]),min(vac[1],global_vac[1]));targets=[dict(output='0'*144,probability_interval=zero)]
    for j in (0,47,95,143):
        a,b=env.vacuum(tuple(i for i in range(144) if i!=j));targets.append(dict(output=''.join('1' if i==j else '0' for i in range(144)),probability_interval=(max(F(0),a-zero[1]),min(F(1),b-zero[0]))));print('target single click',j,flush=True)
    records=envelope_records(env);storage=[]
    for i,r in enumerate(records):
        packed=pack_enclosure(r,model.center_numerators,model.center_denominator);assert semantic_digest(unpack_enclosure(packed,model.center_numerators,model.center_denominator))==semantic_digest(r)
        raw=(json.dumps(encode(packed),indent=2)+'\n').encode();data=gzip.compress(raw,compresslevel=6,mtime=0);name=f'certificate_{i:02d}.json.gz';(out/name).write_bytes(data);storage.append(dict(path=name,modes=r['modes'],raw_bytes=len(raw),compressed_bytes=len(data),raw_sha256=hashlib.sha256(raw).hexdigest()))
    save(out/'target.json',encode(dict(targets=targets,model_vacuum_interval=global_vac,envelope_vacuum_interval=vac,transfer_upper=base['transfers']['-40']['all_measurement_TV_transfer_upper'])));save(out/'storage.json',storage)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items())
    save(out/'summary.json',dict(completed=True,full_outputs=5,enclosures=5,seconds=time.perf_counter()-tick,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True));print('physical targets complete',flush=True)


if __name__=='__main__':main()
