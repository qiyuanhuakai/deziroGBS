"""Bind depth16 gains, sparse witnesses, full coverage and storage costs."""
import json,sys,shutil,gzip,hashlib
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration56_comparisons';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration56_comparisons','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    validations={k:read(f'results/iteration56_{k}/validation.json') for k in ('prefix_validation','events_validation')}
    for k,v in validations.items():check('complete '+k,v['count']==v['passed']>0)
    v=validations['prefix_validation'];ev=validations['events_validation'];check('physical and native coverage',v['physical_certificates']==65537 and v['old_transform_atoms']==32768 and v['replayed_steps']==9792 and v['prefix_trees']==4)
    check('event coverage',ev['cutoffs']==4 and ev['parent_splits']==131072 and ev['sparse_events']==64)
    new=read('results/iteration56_prefix_validation/bounds.json');old=read('results/iteration53_prefix_validation/bounds.json');manifest=read('results/iteration53_manifest.json');check('old bounds binding',deps['results/iteration53_prefix_validation/bounds.json']==manifest['artifacts']['results/iteration53_prefix_validation/bounds.json']==v['dependencies']['results/iteration53_prefix_validation/bounds.json'])
    events=read('results/iteration56_events_validation/rows.json');produced=read('results/iteration56_events/rows.json');depth=read('results/iteration56_prefix_validation/depth_bounds.json');check('all depth entries',len(depth)==64 and len({(r['cutoff'],r['depth']) for r in depth})==64);rows=[]
    for a,b,e,pr in zip(old,new,events,produced):
        k=a['cutoff'];before=a['full144_TV_lower'];lower=e['full144_lower'];check(f'K{k} identity',b['cutoff']==e['cutoff']==pr['cutoff']==k and b['old_fifteen_mode_lower']==before)
        check(f'K{k} combined bound',lower==pr['combined_prefix_TV'][0]>=b['full144_TV_lower'] and e['shifted_lower']==max(F(0),lower-b['transfer_upper']))
        deepest=next(r for r in depth if r['cutoff']==k and r['depth']==16);check(f'K{k} deepest raw bound',deepest['TV_lower']==b['full144_TV_lower'])
        check(f'K{k} gain consistency',e['gain_interval'][0]>=0 and pr['combined_prefix_TV'][0]-a['prefix_TV_intervals']['uint64'][1]<=e['gain_interval'][1] and e['gain_interval'][0]<=pr['combined_prefix_TV'][1]-a['prefix_TV_intervals']['uint64'][0])
        for side in ('program_excess','target_excess'):
            selected=[r for r in e['sparse'] if r['side']==side];check(f'K{k} {side} budgets',[r['budget'] for r in selected]==[1,4,16,64,256,1024,4096,'all'])
            check(f'K{k} {side} sparse monotonicity',all(a['lower']<=b['lower'] for a,b in zip(selected,selected[1:])))
            check(f'K{k} {side} event bounded by fine TV',all(0<=r['lower']<=pr['combined_prefix_TV'][1] for r in selected))
        rows.append(dict(cutoff=k,old_lower=before,direct_lower=b['full144_TV_lower'],new_lower=lower,shifted_lower=e['shifted_lower'],increment=lower-before,percent_gain=100*(lower/before-1),strictly_strengthened=lower>before,gain_interval=e['gain_interval'],opposite_parents=e['opposite_parents'],sparse=e['sparse']))
    storage=read('results/iteration56_target/storage.json');check('128 new shards',len(storage)==128 and sum(r['records'] for r in storage)==32768)
    for r in storage:
        n='results/iteration56_target/'+r['path'];deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();raw=gzip.decompress(b);check('new shard bytes '+n,len(b)==r['compressed_bytes'] and len(raw)==r['compact_json_bytes'] and hashlib.sha256(raw).hexdigest()==r['compact_json_sha256'] and deps[n]==v['dependencies'][n])
    totals={k:sum(r[k] for r in storage) for k in ('expanded_json_bytes','compact_json_bytes','compressed_bytes')};totals['expanded_over_compressed']=F(totals['expanded_json_bytes'],totals['compressed_bytes'])
    native=[]
    for k in range(1,5):
        r=read(f'results/iteration56_prefix/global_{k}.storage.json')
        for kind,m in r.items():
            n='results/iteration56_prefix/'+m['path'];deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();raw=gzip.decompress(b);check('native bytes '+n,len(b)==m['compressed_bytes'] and len(raw)==m['raw_bytes'] and hashlib.sha256(raw).hexdigest()==m['raw_sha256'] and deps[n]==v['dependencies'][n]);native.append(m)
    native_totals={key:sum(r[key] for r in native) for key in ('raw_bytes','compressed_bytes')};split_totals={key:sum(r['storage'][key] for r in produced) for key in ('raw_bytes','compressed_bytes')}
    transform=read('results/iteration56_prefix_validation/transform.json');check('operation counts',transform['direct_terms']==3**16 and transform['producer_interval_subtractions']==16*2**15)
    target=read('results/iteration56_target/summary.json');check('timing stages covered',0<sum(target['timings'].values())<=target['seconds']);check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()))
    save(out/'rows.json',encode(rows));save(out/'storage.json',encode(dict(new_certificates=totals,native=native_totals,split_witnesses=split_totals)));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=len(checks),source_hashes=hashes,dependencies=deps));save(out/'summary.json',dict(completed=True,cutoffs=4,new_certificates=32768,reused_certificates=32767,old_transform_atoms=32768,parent_splits=131072,sparse_events=64,strengthened=sum(r['strictly_strengthened'] for r in rows),scope='Full144 lower bounds only. Sparse witnesses extracted after full laws are computed. Storage costs are measured bytes; timings are not isolated benchmarks.'));print('comparisons',len(checks),flush=True)


if __name__=='__main__':main()
