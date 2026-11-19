"""Recompute shared-target comparisons and explicit mode-order event witness."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration46_comparison_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration46_comparisons','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];witnesses=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration46_analysis';rows=read(base+'/comparisons.json');orders=read(base+'/order_comparisons.json');summary=read(base+'/summary.json');hier=read('results/iteration46_hierarchy_validation/rows.json');lookup={(r['input'],r['order']):r for r in hier}
    for n,s in summary['dependencies'].items():check('analysis dependency '+n,digest(ROOT/n)==s);deps[n]=s
    for r in rows:
        i=r['case'];source=read(f'results/iteration46_inputs/case_{i:02d}.json');check(f'{i} physical identity',r['job']==source['job']);values={}
        for k in (2,3,4):values[f'Frugal_K{k}']=read(f'results/iteration46_native/case_{i:02d}_k{k}.audit.json')['physical_TV_intervals']['uint64']
        for k in (2,3):values[f'maxent_degree{k}']=lookup[i,k]['true_maxent_TV']
        if source['job']['modes']==4:values['maxent_full_information']=lookup[i,4]['true_maxent_TV']
        values['MPS_max_budget']=read(f'results/iteration46_mps/case_{i*4+3:02d}.json')['sampler_physical_TV_interval'];check(f'{i} all method errors',values==r['values']);pairs={(a,b):values[b][0]-values[a][1] for a in values for b in values if a!=b and values[a][1]<values[b][0]};check(f'{i} all strict comparisons',pairs=={(p['better'],p['worse']):p['certified_gap'] for p in r['strict_comparisons']})
    for seq,r in enumerate(orders):
        i,j=r['case_pair'];source=read(f'results/iteration46_inputs/case_{i:02d}.json');n=source['job']['modes'];method=r['method']
        if method=='MPS':
            config=r['configuration'];d=config['dimension'];rank=config['rank'];slot=2*((d==8) if n==4 else (d==3))+(rank==8);a=read(f'results/iteration46_mps/case_{i*4+slot:02d}.json')['sampling']['word_law'];b=read(f'results/iteration46_mps/case_{j*4+slot:02d}.json')['sampling']['word_law']
        elif method.startswith('Frugal'):
            k=int(method[-1]);a=read(f'results/iteration46_native/case_{i:02d}_k{k}.audit.json')['laws']['uint64']['distribution'];b=read(f'results/iteration46_native/case_{j:02d}_k{k}.audit.json')['laws']['uint64']['distribution']
        else:
            k=int(method[-1]);a=read(f"results/iteration46_hierarchy/case_{lookup[i,k]['case']:02d}.json")['projection']['law'];b=read(f"results/iteration46_hierarchy/case_{lookup[j,k]['case']:02d}.json")['projection']['law']
        aligned=[]
        for x in range(len(a)):
            index=x if x==1<<n and len(a)==(1<<n)+1 else (x//(1<<n))*(1<<n)+int(format(x%(1<<n),f'0{n}b')[::-1],2);aligned.append(b[index])
        distance=sum(abs(x-y) for x,y in zip(a,aligned))/2;check(f'order{seq} independent permutation',distance==r['aligned_TV']);event=[x for x in range(len(a)) if a[x]>aligned[x]];pa=sum(a[x] for x in event);pb=sum(aligned[x] for x in event);check(f'order{seq} event witness',pa-pb==distance)
        if method.startswith('Frugal'):
            target=source['target']['target_intervals']+[(F(0),F(0))]*(len(a)-(1<<n));lo=sum(target[x][0] for x in event);hi=sum(target[x][1] for x in event);ea=max(F(0),lo-pa,pa-hi);eb=max(F(0),lo-pb,pb-hi);ta=rows[i]['values'][method];tb=rows[j]['values'][method];check(f'order{seq} triangle target consistency',distance<=ta[1]+tb[1] and ea<=ta[1] and eb<=tb[1]);witnesses.append(dict(case_pair=[i,j],method=method,event=event,forward_probability=pa,aligned_reverse_probability=pb,physical_event_interval=(lo,hi),forward_event_TV_lower=ea,reverse_event_TV_lower=eb,order_TV=distance,at_least_one_orientation_TV_lower=distance/2))
    maximum=max(witnesses,key=lambda r:r['order_TV']);check('maximum order witness',float(maximum['order_TV'])==summary['max_order_TV'][maximum['method']]);check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'maximum_order_witness.json',encode(maximum));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),shared_inputs=8,order_comparisons=len(orders),native_event_witnesses=len(witnesses),source_hashes=hashes,dependencies=deps));print('maximum event',maximum['case_pair'],maximum['method'],float(maximum['order_TV']),float(maximum['forward_event_TV_lower']),float(maximum['reverse_event_TV_lower']),flush=True)
if __name__=='__main__':main()
