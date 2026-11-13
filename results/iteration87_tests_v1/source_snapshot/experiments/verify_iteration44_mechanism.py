"""Independent Walsh matching and forward conditionals on all four-mode controls."""
import json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration41_independent import parity_law,forward
ROOT=Path(__file__).resolve().parents[1]
def main():
    out=ROOT/'results/iteration44_mechanism_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration44_mechanism','iteration22_external','iteration33_run','verify_iteration34','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};checks=[];deps={};rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    plan=read('results/iteration44_mechanism/plan.json');summary=read('results/iteration44_mechanism/summary.json');check('all12complete',plan['cases']==list(range(12)) and summary['planned']==summary['completed']==12 and summary['unchanged_sources'])
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration44_mechanism/source_snapshot'/n))
    for n,s in plan['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    validated=read('results/iteration44_recovery_v2/validation.json');check('main audit passed',validated['count']==validated['passed']==113573)
    for i in range(12):
        r=read(f'results/iteration44_panel/case_{i:02d}.json');m=read(f'results/iteration44_mechanism/case_{i:02d}.json');mu=r['moment_midpoints'];pairs=[(a,b) for b in range(4) for a in range(b)];p=[(1+x)/2 for x in mu[:4]];c={(a,b):(mu[4+k]-mu[a]*mu[b])/4 for k,(a,b) in enumerate(pairs)};law=parity_law(p,c);tag=f'case{i} ';check(tag+'independent matching',law==m['controls']['matching'] and p==m['pvac'] and encode(c)==encode(m['covariance']) and min(law)>0 and sum(law)==1);features=[1<<j for j in range(4)]+[(1<<a)|(1<<b) for a,b in pairs]
        for mask,wanted in zip(features,mu):check(tag+'same low moment '+str(mask),sum(x*(-1)**((atom&mask).bit_count()) for atom,x in enumerate(law))==wanted)
        rational_sampler=[];no_clamps=True
        for atom in range(16):
            prefix=''.join(str((atom>>j)&1) for j in range(4));raw=forward(prefix,p,c,False);clamped=forward(prefix,p,c,True);check(tag+'forward mass '+str(atom),raw['mass']==m['controls']['unclamped'][atom] and clamped['mass']==m['controls']['clamped'][atom]);mass=F(1)
            for j,(previous,z) in enumerate(clamped['records']):
                check(tag+f'valid ratio {atom}:{j}',previous>0 and 0<=z<=previous);mass*=z/previous if prefix[j]=='0' else 1-z/previous
            rational_sampler.append(mass);no_clamps &= clamped['clipped']==0
        check(tag+'same nonnegative law',rational_sampler==m['controls']['sampler']==m['controls']['unclamped']==law and no_clamps and not m['controls']['saturated']);target=r['target']['target_intervals'];tv=(sum(max(F(0),a-x,x-b) for x,(a,b) in zip(law,target))/2,sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(law,target))/2);rho=r['bridge']['atom_radius'];bridge=r['bridge']['TV_to_true_feasible_upper'];upper=tv[1]+bridge;gap=r['true_maxent_physical_TV_interval'][0]-upper
        check(tag+'strict true-moment counterexample',m['strictly_positive'] and m['true_moment_feasible'] and min(law)>rho and tuple(m['matching_TV'])==tv and m['true_feasible_physical_TV_upper']==upper and m['true_maxent_excess_lower']==gap>0);rows.append(dict(case=i,matching_TV=tv,true_feasible_upper=upper,true_maxent_excess_lower=gap));print('verified same-moment control',i,flush=True)
    check('unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=12,same_moment_counterexamples=12,source_hashes=hashes,dependencies=deps));print('mechanism verified',len(checks),flush=True)
if __name__=='__main__':main()
