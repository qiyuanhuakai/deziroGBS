"""All paired old/new cases and predeclared holdout refinement comparisons."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration45_independent import tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration51_comparisons';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration51_comparisons','iteration22_external','iteration33_run','verify_iteration34','iteration45_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];paired=[];rows=[];records=[];groups={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    def valid(n):
        v=read(n+'/validation.json');check(n+' validated',v['count']==v['passed']);return v
    def bound(n,v):
        r=read(n);check('bound '+n,deps[n]==v['dependencies'][n]);return r
    v=valid('results/iteration51_noise_validation');iv=valid('results/iteration51_integrated_validation');ov=valid('results/iteration50_noise_validation');cv=valid('results/iteration51_controls');oldinputs=set()
    for i in range(50):
        r=bound(f'results/iteration51_noise/case_{i:02d}.json',v);integ=bound(f'results/iteration51_integrated/case_{i:02d}.json',iv);job=r['job'];key=(job['pairing'],job['sigma'],job['dimension']);check('input binding '+str(i),job==integ['job']);actual=tv(r['mixture_law'],r['continuous_target']+[(F(0),F(0))]);finite=tv(r['mixture_law'],integ['integral']['law']+[(F(0),F(0))]);check('recomputed full errors '+str(i),actual==tuple(r['full_Gaussian_TV']) and finite==tuple(integ['program_vs_finite_continuous_TV']))
        radius=r['grid']['standard_radius'];N=job['cells'];k={16:2,64:3,256:4,1024:5}[N];analytic=job['sigma']*(F(k+1,4**k)+F(1,2**((k+1)**2//2)));check('universal ideal envelope '+str(i),r['discretization_TV_upper']<analytic)
        row=dict(case=i,job=job,radius=radius,actual_TV=actual,finite_continuous_TV=finite,discretization_upper=r['discretization_TV_upper'],total_upper=r['total_TV_upper'],universal_ideal_upper=analytic,selector_upper=r['selector_TV'][1],arithmetic_upper=r['weighted_arithmetic_TV_upper'],representation_bias=integ['finite_continuous_vs_Gaussian_signed_vacuum_bias']);rows.append(row);groups.setdefault(key,[]).append(row);records.append(r)
        if i<36:
            old=bound(f'results/iteration50_noise/case_{i:02d}.json',ov);oldinputs.add(key)
            names_np=[f'results/iteration{it}_noise/case_{i:02d}.npz' for it in (51,50)];arrays=[]
            for n,vv in zip(names_np,(v,ov)):deps[n]=digest(ROOT/n);check('bound '+n,deps[n]==vv['dependencies'][n]);arrays.append(np.load(ROOT/n)['tensor_0'])
            check('paired same tensor/target '+str(i),job==old['job'] and np.array_equal(*arrays) and r['continuous_target']==old['continuous_target']);oldtv=tv(old['mixture_law'],old['continuous_target']+[(F(0),F(0))]);diff=(actual[0]-oldtv[1],actual[1]-oldtv[0]);equal=r['mixture_law']==old['mixture_law'];status='identical_program' if equal else 'improved' if diff[1]<0 else 'regressed' if diff[0]>0 else 'unresolved';check('N64 exact old program '+str(i),N!=64 or equal);paired.append(dict(case=i,job=job,old_TV=oldtv,new_TV=actual,status=status,difference=diff,old_bound=old['discretization_TV_upper'],new_bound=r['discretization_TV_upper']))
    check('predeclared holdout disjoint old panel',all((r['job']['pairing'],r['job']['sigma'],r['job']['dimension']) not in oldinputs for r in rows[36:48]))
    changes=[]
    for key,group in groups.items():
        for a,b in zip(group,group[1:]):
            check('same input refinement '+str((a['case'],b['case'])),a['job']['cells']<b['job']['cells'] and a['radius']<b['radius']);check('strict coupling bound decrease '+str(b['case']),b['discretization_upper']<a['discretization_upper']);changes.append(dict(from_case=a['case'],to_case=b['case'],key=key,finite_error_strictly_reduced=b['finite_continuous_TV'][1]<a['finite_continuous_TV'][0],actual_error_strictly_reduced=b['actual_TV'][1]<a['actual_TV'][0],actual_error_strictly_increased=b['actual_TV'][0]>a['actual_TV'][1]))
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));summary=dict(cases=50,paired_old_cases=36,paired_counts={s:sum(r['status']==s for r in paired) for s in ('improved','regressed','identical_program','unresolved')},holdout_cases=12,holdout_distinct_inputs=4,refinement_edges=len(changes),finite_error_reduced=sum(r['finite_error_strictly_reduced'] for r in changes),actual_error_reduced=sum(r['actual_error_strictly_reduced'] for r in changes),actual_error_increased=sum(r['actual_error_strictly_increased'] for r in changes),scope='Ideal growing-radius convergence does not imply monotone total error or fixed-precision arbitrary accuracy; old paired cases retained in full')
    save(out/'paired.json',encode(paired));save(out/'rows.json',encode(rows));save(out/'refinements.json',encode(changes));save(out/'summary.json',summary);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
