"""Independent endpoint optimization and exhaustive per-parent split audit."""
import sys,json,gzip,hashlib,shutil,heapq
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration56_events_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration56_events','iteration22_external','iteration33_run','verify_iteration34')]
    hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];shards=[];count=0;rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):
        deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def flush():
        if checks:
            name=f'checks_{len(shards):03d}.json';save(out/name,checks);shards.append(name);checks.clear()
    def check(n,ok):
        nonlocal count
        count+=1;checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
        if len(checks)>=20000:flush()
    summary=read('results/iteration56_events/summary.json');primary=read('results/iteration56_prefix_validation/validation.json')
    check('production complete',summary['completed'] and summary['cutoffs']==4 and summary['parents_per_cutoff']==32768 and summary['events_per_cutoff']==16 and summary['unchanged_sources'] and summary['unchanged_dependencies'])
    check('physical audit complete',primary['count']==primary['passed'] and primary['physical_certificates']==65537 and primary['replayed_steps']==9792)
    for n,s in summary['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration56_events/source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    physical=read('results/iteration56_target/target.json');p=physical['law']['intervals'];transfer=physical['transfer_upper'];check('bound physical target',deps['results/iteration56_target/target.json']==primary['dependencies']['results/iteration56_target/target.json'])
    old=read('results/iteration53_prefix_validation/bounds.json');oldmanifest=read('results/iteration53_manifest.json');check('old bound bytes',deps['results/iteration53_prefix_validation/bounds.json']==oldmanifest['artifacts']['results/iteration53_prefix_validation/bounds.json'])
    primarybounds=read('results/iteration56_prefix_validation/bounds.json');produced=read('results/iteration56_events/rows.json');check('four unique rows',[r['cutoff'] for r in produced]==[1,2,3,4])
    for row in produced:
        k=row['cutoff'];tag=f'K{k} ';name=f'results/iteration56_prefix/global_{k}.audit.json.gz';q=read(name)['laws']['uint64']['distribution'];check(tag+'bound program law',deps[name]==primary['dependencies'][name] and len(q)==len(p)==65536 and sum(q)==1)
        events=read(f'results/iteration56_events/global_{k}.events.json');check(tag+'event count',len(events)==16)
        for index,event in enumerate(events):
            side='program_excess' if index<8 else 'target_excess';budget=(1,4,16,64,256,1024,4096,'all')[index%8]
            scores={i:(q[i]-p[i][1] if side=='program_excess' else p[i][0]-q[i]) for i in range(65536)}
            eligible=[i for i,s in scores.items() if s>0];limit=len(eligible) if budget=='all' else min(budget,len(eligible));expected=heapq.nlargest(limit,eligible,key=lambda i:(scores[i],-i))
            check(tag+f'event {index} optimal certified selection',event['side']==side and event['budget']==budget and event['masks']==expected)
            Q=sum((q[i] for i in expected),F(0));P=(sum((p[i][0] for i in expected),F(0)),sum((p[i][1] for i in expected),F(0)));gap=max(F(0),Q-P[1] if side=='program_excess' else P[0]-Q)
            check(tag+f'event {index} probability and lower',event['sampler_probability']==Q and tuple(event['target_probability_interval'])==P and event['TV_lower']==gap==sum((scores[i] for i in expected),F(0)))
        meta=row['storage'];name='results/iteration56_events/'+meta['path'];b=(ROOT/name).read_bytes();raw=gzip.decompress(b);check(tag+'split gzip bytes',len(raw)==meta['raw_bytes'] and len(b)==meta['compressed_bytes'] and hashlib.sha256(raw).hexdigest()==meta['raw_sha256'] and b[4:8]==bytes(4));split=read(name);check(tag+'all parent records',len(split['rows'])==32768)
        lower=upper=midgain=fine_mid=coarse_mid=F(0);opposite=0
        for i,r in enumerate(split['rows']):
            x=(q[i]-p[i][1],q[i]-p[i][0]);y=(q[i+32768]-p[i+32768][1],q[i+32768]-p[i+32768][0]);corners=[(abs(a)+abs(b)-abs(a+b))/2 for a in x for b in y];a,b=sum(x)/2,sum(y)/2;g=(abs(a)+abs(b)-abs(a+b))/2
            check(tag+f'parent {i} input',r['parent']==i and r['child_difference_intervals']==[list(x),list(y)])
            check(tag+f'parent {i} exact rectangle range',tuple(r['gain_interval'])==(min(corners),max(corners)) and min(corners)<=g<=max(corners) and r['midpoint_gain']==g)
            direction=1 if x[1]<0<y[0] else -1 if y[1]<0<x[0] else 0
            check(tag+f'parent {i} opposite signs',r['certified_opposite_orientation']==direction)
            lower+=min(corners);upper+=max(corners);midgain+=g;fine_mid+=(abs(a)+abs(b))/2;coarse_mid+=abs(a+b)/2;opposite+=direction!=0
        check(tag+'split sum and identity',tuple(split['gain_interval'])==tuple(row['gain_interval'])==(lower,upper) and split['midpoint_gain']==midgain==fine_mid-coarse_mid and split['fine_midpoint_half_l1']==fine_mid and split['coarse_midpoint_half_l1']==coarse_mid)
        check(tag+'opposite count',row['certified_opposite_parents']==split['certified_opposite_parents']==opposite)
        before=next(r for r in old if r['cutoff']==k)['prefix_TV_intervals']['uint64'];fine=next(r for r in primarybounds if r['cutoff']==k)['prefix_TV_intervals']['uint64'];propagated=(before[0]+lower,before[1]+upper);combined=(max(fine[0],propagated[0]),min(fine[1],propagated[1]))
        check(tag+'independent bound propagation',row['old_prefix_TV']==before and row['direct_prefix_TV']==fine and tuple(row['propagated_prefix_TV'])==propagated and tuple(row['combined_prefix_TV'])==combined and combined[0]<=combined[1] and row['full144_TV_lower']==combined[0] and row['shifted_full144_TV_lower']==max(F(0),combined[0]-transfer))
        top=heapq.nlargest(16,split['rows'],key=lambda r:(r['gain_interval'][0],-r['parent']));check(tag+'largest split contributions',row['top_gain_parents']==top)
        sparse=[dict(side=e['side'],budget=e['budget'],atoms=len(e['masks']),lower=e['TV_lower'],fraction_of_full_prefix_lower=e['TV_lower']/combined[0]) for e in events]
        rows.append(dict(cutoff=k,full144_lower=combined[0],shifted_lower=max(F(0),combined[0]-transfer),gain_interval=(lower,upper),opposite_parents=opposite,sparse=sparse));print('verified events',k,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));flush()
    save(out/'rows.json',encode(rows));save(out/'validation.json',dict(count=count,passed=count,check_shards=shards,cutoffs=4,parent_splits=131072,sparse_events=64,source_hashes=hashes,dependencies=deps))


if __name__=='__main__':main()
