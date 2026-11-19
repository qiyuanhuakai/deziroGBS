"""Bind round-53 lower-bound increments and lossless storage accounting."""
import json,sys,shutil,gzip,hashlib
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration53_comparisons';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration53_comparisons','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    validations={k:read(f'results/iteration53_{k}/validation.json') for k in ('prefix_validation','controls')}
    for k,v in validations.items():check('complete '+k,v['count']==v['passed']>0)
    v=validations['prefix_validation'];check('coverage',v['physical_certificates']==32769 and v['replayed_steps']==9792 and v['prefix_trees']==4)
    check('old compact regression coverage',validations['controls']['old_certificates']==16383 and validations['controls']['atoms']==16384)
    new=read('results/iteration53_prefix_validation/bounds.json');old=read('results/iteration49_prefix_validation/bounds.json');manifest=read('results/iteration49_manifest.json');check('bound old lower bytes',deps['results/iteration49_prefix_validation/bounds.json']==manifest['artifacts']['results/iteration49_prefix_validation/bounds.json']==v['dependencies']['results/iteration49_prefix_validation/bounds.json']);rows=[]
    depth=read('results/iteration53_prefix_validation/depth_bounds.json');check('all depth entries',len(depth)==60 and len({(r['cutoff'],r['depth']) for r in depth})==60)
    for a,b in zip(old,new):
        k=a['cutoff'];lo=b['full144_TV_lower'];before=a['full144_TV_lower'];check(f'K{k} identity',b['cutoff']==k and b['old_fourteen_mode_lower']==before);check(f'K{k} strengthened',lo>before and b['strictly_strengthened']);check(f'K{k} shifted transfer',b['full144_shifted_TV_lower']==max(F(0),lo-b['transfer_upper'])>0)
        check(f'K{k} event bound',0<b['event']['TV_lower']<=1 and b['event']['sampler_probability']-b['event']['target_probability_interval'][1]==b['event']['TV_lower']);d=next(r for r in depth if r['cutoff']==k and r['depth']==15);check(f'K{k} deepest bound',d['TV_lower']==lo and d['shifted_lower']==b['full144_shifted_TV_lower']);rows.append(dict(cutoff=k,old_lower=before,new_lower=lo,shifted_lower=b['full144_shifted_TV_lower'],increment=lo-before,factor=lo/before,percent_gain=100*(lo/before-1)))
    storage=read('results/iteration53_target/storage.json');check('64 new shards',len(storage)==64 and sum(r['records'] for r in storage)==16384)
    for r in storage:
        n='results/iteration53_target/'+r['path'];deps[n]=digest(ROOT/n);raw=(ROOT/n).read_bytes();expanded=gzip.decompress(raw);check('new shard bytes '+n,len(raw)==r['compressed_bytes'] and len(expanded)==r['compact_json_bytes'] and hashlib.sha256(expanded).hexdigest()==r['compact_json_sha256'] and deps[n]==v['dependencies'][n])
    totals={k:sum(r[k] for r in storage) for k in ('expanded_json_bytes','compact_json_bytes','compressed_bytes')};totals['expanded_over_compact']=F(totals['expanded_json_bytes'],totals['compact_json_bytes']);totals['expanded_over_compressed']=F(totals['expanded_json_bytes'],totals['compressed_bytes']);totals['compact_over_compressed']=F(totals['compact_json_bytes'],totals['compressed_bytes'])
    oldstorage=read('results/iteration53_controls/storage.json');check('all old storage controls',len(oldstorage)==64 and sum(r['records'] for r in oldstorage)==16383);oldtotals={k:sum(r[k] for r in oldstorage) for k in ('expanded_bytes','compact_bytes','compressed_bytes')}
    native=[]
    for k in range(1,5):
        r=read(f'results/iteration53_prefix/global_{k}.storage.json')
        for kind,m in r.items():
            n='results/iteration53_prefix/'+m['path'];deps[n]=digest(ROOT/n);raw=(ROOT/n).read_bytes();expanded=gzip.decompress(raw);check('native bytes '+n,len(raw)==m['compressed_bytes'] and len(expanded)==m['raw_bytes'] and hashlib.sha256(expanded).hexdigest()==m['raw_sha256'] and deps[n]==v['dependencies'][n]);native.append(dict(cutoff=k,kind=kind,**m))
    nativetotals={key:sum(r[key] for r in native) for key in ('raw_bytes','compressed_bytes')}
    transform=read('results/iteration53_prefix_validation/transform.json');check('operation counts',transform['direct_terms']==3**15 and transform['producer_interval_subtractions']==15*2**14);controltransform=read('results/iteration53_controls/transform.json');check('old operation counts',controltransform['direct_terms']==3**14 and controltransform['interval_subtractions']==14*2**13)
    target=read('results/iteration53_target/summary.json');check('timing stages covered',0<sum(target['timings'].values())<=target['seconds']);check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()))
    save(out/'rows.json',encode(rows));save(out/'storage.json',encode(dict(new_certificates=totals,old_controls=oldtotals,native=nativetotals)));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));save(out/'summary.json',dict(completed=True,cutoffs=4,new_certificates=16384,old_controls=16383,scope='Full144 TV lower bounds only. Storage ratios are measured bytes; observations are not isolated timing benchmarks.'));print('comparisons',len(checks),flush=True)
if __name__=='__main__':main()
