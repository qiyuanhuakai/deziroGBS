"""All16383 old certificates roundtrip and full14-bit transform regression."""
import sys,json,shutil,time,gzip,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration53_independent import expand_record,direct_integer_law
from gbslab.compact_gaussian_evidence import pack_enclosure,integer_mobius_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration53_controls';out.mkdir(exist_ok=False);names=['research/iteration53_prefix_plan.md',*[f'experiments/{n}.py' for n in ('verify_iteration53_controls','iteration53_independent','iteration22_external','iteration33_run','verify_iteration34')],'src/gbslab/compact_gaussian_evidence.py'];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[];vac={(): (F(1),F(1))};count=0
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base=read('results/iteration35_panel/case_07.audit.json');C=np.array(base['center_numerators'],dtype=object);dc=base['center_denominator'];old_validation=read('results/iteration49_prefix_validation/validation.json');check('old complete physical audit',old_validation['count']==old_validation['passed'])
    for it in (47,49):
        manifest=read(f'results/iteration{it}_manifest.json')
        for shard in range(32):
            n=f'results/iteration{it}_target/certificates_{shard:02d}.json';records=read(n);check('bound old shard '+n,deps[n]==manifest['artifacts'][n]==old_validation['dependencies'][n]);started=time.perf_counter();packed=[]
            for record in records:
                key=tuple(record['modes']);check('old subset unique '+str(key),key not in vac);vac[key]=tuple(record['vacuum_interval']);p=pack_enclosure(record,C,dc);expanded=expand_record(p,C,dc);check('independent full semantic roundtrip '+str(key),expanded==record);packed.append(p);count+=1
            full=(json.dumps(encode(records),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8');raw=(json.dumps(encode(packed),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8');z=gzip.compress(raw,compresslevel=6,mtime=0);check('lossless gzip '+n,gzip.decompress(z)==raw and z[4:8]==b'\x00'*4);rows.append(dict(source_round=it,shard=shard,records=len(records),expanded_bytes=len(full),compact_bytes=len(raw),compressed_bytes=len(z),seconds=time.perf_counter()-started,compact_sha256=hashlib.sha256(raw).hexdigest()));print('old compact shard',it,shard,flush=True)
    check('all old14 subsets',count==16383 and len(vac)==16384);table=[vac[tuple(j for j in range(14) if mask>>j&1)] for mask in range(16384)];started=time.perf_counter();fast=integer_mobius_law(table);fast_seconds=time.perf_counter()-started;started=time.perf_counter();raw,law,terms=direct_integer_law(table);direct_seconds=time.perf_counter()-started;target=read('results/iteration49_target/target.json')['law'];check('bound old target',deps['results/iteration49_target/target.json']==old_validation['dependencies']['results/iteration49_target/target.json']);check('all14bit raw and physical intervals equal',fast['raw_intervals']==raw==[tuple(x) for x in target['raw_intervals']] and fast['intervals']==law==[tuple(x) for x in target['intervals']] and terms==3**14);check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'storage.json',rows);save(out/'transform.json',dict(fast_seconds=fast_seconds,direct_seconds=direct_seconds,interval_subtractions=fast['interval_subtractions'],direct_terms=terms,scope='Same exact14-bit target intervals; concurrent single-run observations'));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),old_certificates=count,atoms=16384,source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
