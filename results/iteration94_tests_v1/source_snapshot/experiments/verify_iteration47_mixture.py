"""Independent native laws, relabeling, cancellation and holdout classifications."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration41_independent import layer_law
from iteration45_independent import tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration47_mixture_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration47_mixture','iteration22_external','iteration33_run','verify_iteration34','iteration41_independent','iteration45_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[];cache={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration47_mixture';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('48 complete mixtures',summary['planned']==summary['completed']==48)
    for n,s in summary['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    validation={44:read('results/iteration44_recovery_v2/validation.json'),46:read('results/iteration46_native_validation/validation.json'),47:read('results/iteration47_native_validation/validation.json')}
    for it,v in validation.items():check(f'{it} native proof passed',v['count']==v['passed'])
    def native(it,i,k):
        key=(it,i,k)
        if key not in cache:
            path=f'results/iteration{it}_native/case_{i:02d}_k{k}.audit.json';audit=read(path);check('bound native '+str(key),deps[path]==validation[it]['dependencies'][path]);raw=read(f'results/iteration{it}_native/case_{i:02d}_k{k}.stdout.json');law=layer_law({r['prefix']:r['threshold_hex'] for r in raw['prefix_tree']},10,'uint64');check('independent native law '+str(key),law==audit['laws']['uint64']['distribution']);cache[key]=law
        return cache[key]
    for index,job in enumerate(plan['jobs']):
        rec=read(base+f'/case_{index:02d}.json');it=job['round'];i=job['first'];k=job['cutoff'];folder='panel' if it==44 else 'inputs';source=read(f'results/iteration{it}_{folder}/case_{i:02d}.json');other=read(f'results/iteration{it}_{folder}/case_{i+1:02d}.json');n=source['job']['modes'];target=source['target']['target_intervals']+[(F(0),F(0))]*(1024-(1<<n));p=native(it,i,k);q=native(it,i+1,k)
        def indexmap(mask):return (mask//(1<<n))*(1<<n)+int(format(mask%(1<<n),f'0{n}b')[::-1],2)
        aligned=[q[indexmap(mask)] for mask in range(1024)];check(f'{index} physical relabeling',all(source['target']['target_intervals'][m]==other['target']['target_intervals'][indexmap(m)] for m in range(1<<n)));mixed=[(x+y)/2 for x,y in zip(p,aligned)];reverse_oriented=[(q[m]+p[indexmap(m)])/2 for m in range(1024)];record=rec['mixture'];gap=sum(abs(x-y) for x,y in zip(p,aligned))/2;check(f'{index} exact fair mixture',rec['job']==job and record['law']==mixed and record['aligned_reverse']==aligned and record['reverse_oriented_law']==reverse_oriented and record['weights']==[F(1,2),F(1,2)] and record['order_TV']==gap);check(f'{index} exact equivariance',all(reverse_oriented[indexmap(m)]==mixed[m] for m in range(1024)) and sum(mixed)==1 and min(mixed)>=0)
        ptv=tv(p,target);qtv=tv(aligned,target);mtv=tv(mixed,target);average=tuple((a+b)/2 for a,b in zip(ptv,qtv));check(f'{index} all TV intervals',list(ptv)==rec['forward_TV'] and list(qtv)==rec['reverse_TV'] and list(mtv)==rec['mixture_TV'] and list(average)==rec['average_TV']);check(f'{index} convexity',mtv[1]<=average[1]);lo=hi=F(0)
        for a,b,(l,h) in zip(p,aligned,target):
            candidates=[l,h]+[x for x in (a,b,(a+b)/2) if l<=x<=h];values=[(abs(a-x)+abs(b-x)-abs(a+b-2*x))/4 for x in candidates];lo+=min(values);hi+=max(values)
        check(f'{index} independent cancellation',0<=lo<=hi<=gap/2 and rec['cancellation_gain_interval'][0]<=lo<=hi<=rec['cancellation_gain_interval'][1]);threshold=F(1,10**10);better=mtv[1]+threshold<min(ptv[0],qtv[0]);worse=mtv[0]>min(ptv[1],qtv[1])+threshold;check(f'{index} material classifications',better==rec['material_better_both'] and worse==rec['material_worse_best'] and rec['strict_better_both']==(mtv[1]<min(ptv[0],qtv[0])));rows.append(dict(sequence=index,job=job,forward_TV=ptv,reverse_TV=qtv,mixture_TV=mtv,independent_cancellation_interval=(lo,hi),order_TV=gap,material_better_both=better,material_worse_best=worse));print('verified mixture',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),mixtures=48,independent_native_laws=len(cache),heldout_better_both=sum(r['job']['held_out'] and r['material_better_both'] for r in rows),heldout_worse_best=sum(r['job']['held_out'] and r['material_worse_best'] for r in rows),source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
