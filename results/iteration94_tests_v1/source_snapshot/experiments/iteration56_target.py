"""Incremental16-bit exact144-mode target with compact lossless witnesses."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,gzip,hashlib
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
from gbslab.compact_gaussian_evidence import pack_enclosure,unpack_enclosure,semantic_digest,integer_mobius_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    started=time.perf_counter();out=ROOT/'results/iteration56_target';out.mkdir(exist_ok=False);names=['research/iteration56_prefix_plan.md',*[f'experiments/{n}.py' for n in ('iteration56_target','iteration22_external','iteration33_run','verify_iteration34','iteration35_run','iteration36_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};timings={};storage=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    manifest=read('results/iteration35_manifest.json')
    for n in ('inputs.npz','case_07.audit.json','case_12.audit.json'):
        name='results/iteration35_panel/'+n;deps[name]=digest(ROOT/name);assert deps[name]==manifest['artifacts'][name]
    old_files=[]
    for it in (47,49,53):
        manifest=read(f'results/iteration{it}_manifest.json')
        for i in range(64 if it==53 else 32):
            name=f'results/iteration{it}_target/certificates_{i:02d}.json'+('.gz' if it==53 else '');deps[name]=digest(ROOT/name);assert deps[name]==manifest['artifacts'][name];old_files.append(name)
    save(out/'plan.json',dict(depth=16,source_hashes=hashes,dependencies=deps,reused_shards=old_files,reused_nonempty_subsets=32767,new_nonempty_subsets=32768,control_masks=[(1<<i)-1 for i in range(1,16)],compact_format='real-local-enclosure-v1'))
    arrays=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=read('results/iteration35_panel/case_07.audit.json');model=ExactPairingModel(arrays['B'],arrays['F']);assert encode(certificate(model.core))==encode(base['core']) and encode(certificate(model.noise_certificate))==encode(base['noise']) and model.operator_radius==base['operator_radius'] and model.center_denominator==base['center_denominator'] and np.array_equal(model.center_numerators,np.array(base['center_numerators'],dtype=object));env=GaussianEnvelope(model);selected={};tick=time.perf_counter()
    for name in old_files:
        for stored in read(name):
            r=unpack_enclosure(stored,model.center_numerators,model.center_denominator) if name.endswith('.gz') else stored
            key=tuple(r['modes']);assert key not in env.vacua and 15 not in key;env.vacua[key]=tuple(r['vacuum_interval'])
            if key==tuple(range(len(key))):selected[key]=r
    assert len(env.vacua)==32768;timings['old_load_seconds']=time.perf_counter()-tick;tick=time.perf_counter();control=GaussianEnvelope(model)
    for size in range(1,16):control.vacuum(tuple(range(size)))
    controls=envelope_records(control);assert all(encode(r)==encode(selected[tuple(r['modes'])]) for r in controls);save(out/'recomputed_controls.json',encode(controls));timings['control_recompute_seconds']=time.perf_counter()-tick;tick=time.perf_counter();vac=[env.vacuum(tuple(j for j in range(16) if mask>>j&1)) for mask in range(65536)];timings['new_subset_generation_seconds']=time.perf_counter()-tick;print('subsets complete',flush=True);tick=time.perf_counter();law=integer_mobius_law(vac);law['modes']=tuple(range(16));timings['atom_transform_seconds']=time.perf_counter()-tick;save(out/'target.json',encode(dict(law=law,transfer_upper=base['transfers']['-40']['all_measurement_TV_transfer_upper'])));tick=time.perf_counter();records=envelope_records(env);assert len(records)==32768 and all(15 in r['modes'] for r in records)
    for offset in range(0,len(records),256):
        full=records[offset:offset+256];packed=[pack_enclosure(r,model.center_numerators,model.center_denominator) for r in full];assert all(semantic_digest(unpack_enclosure(p,model.center_numerators,model.center_denominator))==semantic_digest(r) for p,r in zip(packed,full));fullbytes=(json.dumps(encode(full),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8');raw=(json.dumps(encode(packed),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8');data=gzip.compress(raw,compresslevel=6,mtime=0);name=f'certificates_{offset//256:02d}.json.gz';(out/name).write_bytes(data);storage.append(dict(path=name,records=len(full),expanded_json_bytes=len(fullbytes),compact_json_bytes=len(raw),compressed_bytes=len(data),compact_json_sha256=hashlib.sha256(raw).hexdigest()));print('target shard',offset//256,flush=True)
    timings['compact_and_write_seconds']=time.perf_counter()-tick;save(out/'storage.json',storage);assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,enclosures=65535,new_enclosures=32768,reused_enclosures=32767,shards=128,atoms=65536,unchanged_sources=True,unchanged_dependencies=True,timings=timings,seconds=time.perf_counter()-started,source_hashes=hashes,dependencies=deps));print('target complete',flush=True)
if __name__=='__main__':main()
