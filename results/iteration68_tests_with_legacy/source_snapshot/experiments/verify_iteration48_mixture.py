"""Independent piecewise-linear error geometry on already certified program laws."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration45_independent import tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration48_mixture_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration48_mixture','iteration22_external','iteration33_run','verify_iteration34','iteration45_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration48_mixture';summary=read(base+'/summary.json');check('48 complete',summary['planned']==summary['completed']==48)
    for n,s in summary['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    oldvalidation=read('results/iteration47_mixture_validation/validation.json');check('old independent native proofs passed',oldvalidation['count']==oldvalidation['passed']==678);manifest=read('results/iteration47_manifest.json');check('old validation bound to manifest',deps['results/iteration47_mixture_validation/validation.json']==manifest['artifacts']['results/iteration47_mixture_validation/validation.json'])
    for index in range(48):
        rec=read(base+f'/case_{index:02d}.json');name=f'results/iteration47_mixture/case_{index:02d}.json';old=read(name);check(f'{index} frozen certified mixture',deps[name]==manifest['artifacts'][name]==oldvalidation['dependencies'][name] and rec['job']==old['job']);job=old['job'];folder='panel' if job['round']==44 else 'inputs';name=f"results/iteration{job['round']}_{folder}/case_{job['first']:02d}.json";source=read(name);check(f'{index} certified physical target',deps[name]==oldvalidation['dependencies'][name]);target=source['target']['target_intervals']+[(F(0),F(0))]*(1024-(1<<old['modes']));p=[2*x-y for x,y in zip(old['mixture']['law'],old['mixture']['aligned_reverse'])];q=old['mixture']['aligned_reverse'];mix=old['mixture']['law'];g=rec['geometry'];pt=tv(p,target);qt=tv(q,target);mt=tv(mix,target);check(f'{index} independent TV',g['forward_TV']==list(pt) and g['reverse_TV']==list(qt) and g['mixture_TV']==list(mt));sums={};counts=dict(shared_over=0,shared_under=0,opposite=0,exact_zero=0,unresolved=0)
        for mask,(a,b,(l,h)) in enumerate(zip(p,q,target)):
            # Opposite-sign cancellation is min(distance to either branch)/2;
            # it is zero outside the segment between a and b.
            low,high=sorted((a,b));middle=(low+high)/2
            def gain(t):return max(F(0),min(t-low,high-t))/2
            minimum=min(gain(l),gain(h));maximum=max(gain(l),gain(h),gain(middle) if l<=middle<=h else F(0));over=(max(F(0),low-h),max(F(0),low-l));under=(max(F(0),l-high),max(F(0),h-high));k=str(bin(mask).count('1'));bucket=sums.setdefault(k,dict(atoms=0,cancellation=[F(0),F(0)],shared_over=[F(0),F(0)],shared_under=[F(0),F(0)]));bucket['atoms']+=1
            for name,v in [('cancellation',(minimum,maximum)),('shared_over',over),('shared_under',under)]:
                for j in (0,1):bucket[name][j]+=v[j]
            if a==b==l==h:counts['exact_zero']+=1
            elif a>h and b>h:counts['shared_over']+=1
            elif a<l and b<l:counts['shared_under']+=1
            elif (a<l and b>h) or (b<l and a>h):counts['opposite']+=1
            else:counts['unresolved']+=1
        check(f'{index} all click-stratified cancellation/common bias',sums==g['click_bins'] and counts==g['sign_counts']);total=[sum(v['cancellation'][j] for v in sums.values()) for j in (0,1)];check(f'{index} summed cancellation',total==g['cancellation_interval']);check(f'{index} common bias totals',all(g[name+'_interval']==[sum(v[name][j] for v in sums.values()) for j in (0,1)] for name in ('shared_over','shared_under')))
        lo=pt[0]-qt[1];hi=pt[1]-qt[0];imb=(max(F(0),lo,-hi)/2,max(abs(lo),abs(hi))/2);check(f'{index} branch imbalance',list(imb)==g['branch_imbalance_interval']);margin=(total[0]-imb[1],total[1]-imb[0]);check(f'{index} geometry identity interval',list(margin)==g['improvement_over_best_interval']);direct=(min(pt[0],qt[0])-mt[1],min(pt[1],qt[1])-mt[0]);check(f'{index} independent direct improvement intersects geometry',max(direct[0],margin[0])<=min(direct[1],margin[1]));check(f'{index} cancellation convexity budget',0<=total[0]<=total[1]<=g['order_TV']/2 and g['order_TV']==sum(abs(a-b) for a,b in zip(p,q))/2);threshold=F(1,10**10);check(f'{index} unchanged material classification',(margin[0]>threshold)==old['material_better_both'] and (margin[1]<-threshold)==old['material_worse_best']);rows.append(dict(sequence=index,job=job,cancellation=tuple(total),branch_imbalance=imb,improvement_over_best=margin,shared_over=g['shared_over_interval'],shared_under=g['shared_under_interval'],click_bins=sums));print('verified geometry',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),cases=48,source_hashes=hashes,dependencies=deps,scope='Reanalysis of frozen47 panel, not new holdout evidence or fitted weights'))
if __name__=='__main__':main()
