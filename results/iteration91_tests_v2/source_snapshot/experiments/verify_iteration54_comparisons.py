"""Compare all predeclared refinements, finite representations and correlations."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def order(a,b):
    if b[1]<a[0]:return 'improved'
    if b[0]>a[1]:return 'worsened'
    return 'unresolved'

def main():
    out=ROOT/'results/iteration54_comparisons';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration54_comparisons','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    v=read('results/iteration54_noise_validation/validation.json');c=read('results/iteration54_controls/validation.json');check('complete audits',v['count']==v['passed'] and v['cases']==26 and v['nodes']==1524 and v['trace_nodes']==10666 and c['count']==c['passed'] and c['old_cases']==50);rows=read('results/iteration54_noise_validation/rows.json');check('all rows',len(rows)==26);cases=[]
    for i,r in enumerate(rows):
        name=f'results/iteration54_noise/case_{i:02d}.json';rec=read(name);check('audited source '+str(i),deps[name]==v['dependencies'][name] and rec['job']==r['job']);cases.append(rec);a=r['actual_TV'];b=r['reference_TV'];d=r['finite_continuous_TV'];check('triangle relation '+str(i),a[1]>=max(F(0),b[0]-d[1]) and a[0]<=b[1]+d[1]);check('zero failure '+str(i),rec['mixture_law'][-1]==0);check('separate budgets '+str(i),a[1]<=r['total_upper'] and r['total_upper']==min(F(1),r['state_upper']+r['selector_upper']+r['arithmetic_upper']+r['discretization_upper']))
    refinements=[]
    for family in ('vacuum','common','opposite','quadrature','strong','complex_holdout'):
        for d in (4,6):
            group=sorted([r for r in rows if r['job']['family']==family and r['job']['dimension']==d],key=lambda r:r['job']['cells'])
            for a,b in zip(group,group[1:]):
                ia,ib=a['case'],b['case'];check(f'fixed finite reference {ia}:{ib}',cases[ia]['finite_continuous']==cases[ib]['finite_continuous'] and cases[ia]['target']==cases[ib]['target']);check(f'bound tightening {ia}:{ib}',b['discretization_upper']<a['discretization_upper']);refinements.append(dict(family=family,dimension=d,before=ia,after=ib,before_cells=a['job']['cells'],after_cells=b['job']['cells'],physical=order(a['actual_TV'],b['actual_TV']),finite_reference=order(a['finite_continuous_TV'],b['finite_continuous_TV']),ideal_grid=order(a['ideal_grid_TV'],b['ideal_grid_TV'])))
    dimensions=[]
    for a in rows:
        if a['job']['dimension']!=4:continue
        b=next(r for r in rows if r['job']['dimension']==6 and r['job']['family']==a['job']['family'] and r['job']['cells']==a['job']['cells']);check(f'same infinite target dimension {a["case"]}',cases[a['case']]['target']==cases[b['case']]['target']);dimensions.append(dict(family=a['job']['family'],cells=a['job']['cells'],before=a['case'],after=b['case'],physical=order(a['actual_TV'],b['actual_TV']),reference=order(a['reference_TV'],b['reference_TV']),reduction_factor_interval=(a['actual_TV'][0]/b['actual_TV'][1],a['actual_TV'][1]/b['actual_TV'][0])))
    a=cases[4]['target'];b=cases[8]['target'];check('same single-mode Gaussian marginals',all(a['records'][m]['determinant']==b['records'][m]['determinant'] and a['vacuum_intervals'][m]==b['vacuum_intervals'][m] for m in (1,2)));check('different joint determinant',a['records'][3]['determinant']!=b['records'][3]['determinant']);check('directional generator variance',a['generator_variance']==F(2,3) and b['generator_variance']==6);joint=(a['vacuum_intervals'][3][0]-b['vacuum_intervals'][3][1],a['vacuum_intervals'][3][1]-b['vacuum_intervals'][3][0]);check('strict joint vacuum gap',joint[0]>0);full=(sum(max(F(0),l-y,x-h) for (l,h),(x,y) in zip(a['intervals'],b['intervals']))/2,sum(max(abs(l-y),abs(h-x)) for (l,h),(x,y) in zip(a['intervals'],b['intervals']))/2);check('exact structure two times gap',full[0]<=2*joint[1] and 2*joint[0]<=full[1]);correlation=dict(common_case=4,opposite_case=8,single_mode_vacuum_intervals=[a['vacuum_intervals'][m] for m in (1,2)],common_joint_vacuum=a['vacuum_intervals'][3],opposite_joint_vacuum=b['vacuum_intervals'][3],joint_gap=joint,full_law_TV=full,common_generator_variance=a['generator_variance'],opposite_generator_variance=b['generator_variance'])
    check('all paired edges',len(refinements)==14 and len(dimensions)==13);storage={key:sum(r['storage'][key] for r in rows) for key in ('raw_bytes','compressed_bytes','nodes','trace_nodes')};summary=dict(completed=True,cases=26,holdout_cases=sum(r['job']['holdout'] for r in rows),refinement_edges=len(refinements),physical_refinements={s:sum(r['physical']==s for r in refinements) for s in ('improved','worsened','unresolved')},finite_refinements={s:sum(r['finite_reference']==s for r in refinements) for s in ('improved','worsened','unresolved')},dimension_pairs=len(dimensions),nonvacuum_dimension_improved=sum(r['family']!='vacuum' and r['physical']=='improved' for r in dimensions),storage=storage,scope='Rank-one correlated complex Gaussian drive on two-mode squeezed states. No arbitrary-rank noise or large-scale throughput claim.');check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'refinements.json',encode(refinements));save(out/'dimensions.json',encode(dimensions));save(out/'correlation.json',encode(correlation));save(out/'summary.json',summary);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
