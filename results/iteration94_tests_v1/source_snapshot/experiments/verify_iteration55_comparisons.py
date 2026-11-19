"""Audit every coordinate refinement, equal-cost allocation and input-size pair."""
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
    out=ROOT/'results/iteration55_comparisons';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration55_comparisons','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    v=read('results/iteration55_noise_validation/validation.json');c=read('results/iteration55_controls/validation.json');check('complete audits',v['count']==v['passed'] and v['cases']==24 and v['nodes']==3456 and v['trace_nodes']==24184 and v['unique_polynomials']==6 and c['count']==c['passed'] and c['old_cases']==26 and c['fixed_density_controls']==24);rows=read('results/iteration55_noise_validation/rows.json');check('24 rows',len(rows)==24);cases=[]
    for i,r in enumerate(rows):
        name=f'results/iteration55_noise/case_{i:02d}.json';rec=read(name);check('bound input '+str(i),deps[name]==v['dependencies'][name] and rec['job']==r['job']);cases.append(rec);a,b,d=r['actual_TV'],r['reference_TV'],r['finite_continuous_TV'];check('triangle '+str(i),a[1]>=max(F(0),b[0]-d[1]) and a[0]<=b[1]+d[1]);check('zero failure '+str(i),rec['mixture_law'][-1]==0);check('complete budgets and nodes '+str(i),a[1]<=r['total_upper']==min(F(1),r['state_upper']+r['selector_upper']+r['arithmetic_upper']+r['discretization_upper']) and r['storage']['nodes']==(r['job']['cells'][0]+2)*(r['job']['cells'][1]+2))
    edges=[];allocations=[];changes=[]
    for start in (0,4,8,12,16,20):
        for aa,bb,axis in ((0,1,0),(0,2,1),(1,3,1),(2,3,0)):
            a,b=rows[start+aa],rows[start+bb];ia,ib=a['case'],b['case'];check(f'fixed physical and finite input {ia}:{ib}',cases[ia]['finite_continuous']==cases[ib]['finite_continuous'] and cases[ia]['target']==cases[ib]['target']);check(f'coupling shrinks {ia}:{ib}',b['discretization_upper']<a['discretization_upper']);edges.append(dict(family=a['job']['family'],dimension=a['job']['dimension'],before=ia,after=ib,axis=axis,physical=order(a['actual_TV'],b['actual_TV']),finite_reference=order(a['finite_continuous_TV'],b['finite_continuous_TV']),ideal_grid=order(a['ideal_grid_TV'],b['ideal_grid_TV']),before_nodes=a['storage']['nodes'],after_nodes=b['storage']['nodes']))
        a,b=rows[start+1],rows[start+2];check('equal cost '+str(start),a['storage']['nodes']==b['storage']['nodes']==108);allocations.append(dict(family=a['job']['family'],dimension=a['job']['dimension'],first_axis_case=a['case'],second_axis_case=b['case'],second_axis_vs_first_physical=order(a['actual_TV'],b['actual_TV']),second_axis_vs_first_finite=order(a['finite_continuous_TV'],b['finite_continuous_TV'])))
        laws=[cases[start+j]['mixture_law'] for j in range(4)];first=[b-a for a,b in zip(laws[0],laws[1])];second=[b-a for a,b in zip(laws[0],laws[2])];interaction=[d-b-c+a for a,b,c,d in zip(*laws)];half=lambda v:sum(abs(x) for x in v)/2;cancellation=half(first)+half(second)-half([a+b for a,b in zip(first,second)]);opposed=[i for i,(a,b) in enumerate(zip(first,second)) if a*b<0];check('exact coordinate change identity '+str(start),all(laws[3][i]==laws[0][i]+first[i]+second[i]+interaction[i] for i in range(5)) and sum(first)==sum(second)==sum(interaction)==0);check('opposing coordinate changes '+str(start),cancellation>0 and len(opposed)>0);changes.append(dict(start_case=start,first_axis_change=first,second_axis_change=second,interaction=interaction,interaction_half_l1=half(interaction),opposing_outcomes=opposed,opposing_change_cancellation=cancellation,scope='Exact program-law changes from the shared coarse grid; not isolated continuous-axis errors.'))
    produced=read('results/iteration55_noise/rows.json');check('six fresh and eighteen cached inputs',len(produced)==24 and sum(not r['input_reference_cache_hit'] for r in produced)==6)
    for i,r in enumerate(produced):check('timing and cache '+str(i),r['case']==i and r['status']=='completed' and r['input_reference_cache_hit']==(i%4!=0) and 0<sum(r['timing'].values())<=r['seconds'])
    dimensions=[]
    for a in rows:
        if a['job']['dimension']!=4:continue
        b=next(r for r in rows if r['job']['dimension']==6 and r['job']['family']==a['job']['family'] and r['job']['cells']==a['job']['cells']);check('same physical target dimension '+str(a['case']),cases[a['case']]['target']==cases[b['case']]['target']);dimensions.append(dict(family=a['job']['family'],cells=a['job']['cells'],before=a['case'],after=b['case'],physical=order(a['actual_TV'],b['actual_TV']),reference=order(a['reference_TV'],b['reference_TV']),factor=(a['actual_TV'][0]/b['actual_TV'][1],a['actual_TV'][1]/b['actual_TV'][0])))
    U=[[F(3,5),F(4,5)],[-F(4,5),F(3,5)]];check('exact passive orthogonality',all(sum(U[i][k]*U[j][k] for k in range(2))==int(i==j) for i in range(2) for j in range(2)));B=cases[0]['pairing_real'];rot=[[sum(U[i][a]*B[a][b]*U[j][b] for a in range(2) for b in range(2)) for j in range(2)] for i in range(2)];check('exact passive pairing transform',rot==cases[8]['pairing_real']);check('all comparisons',len(edges)==24 and len(allocations)==6 and len(dimensions)==12);storage={k:sum(r['storage'][k] for r in rows) for k in ('raw_bytes','compressed_bytes','nodes','trace_nodes')};summary=dict(completed=True,cases=24,holdout_cases=sum(r['job']['holdout'] for r in rows),refinement_edges=24,physical_refinements={s:sum(e['physical']==s for e in edges) for s in ('improved','worsened','unresolved')},finite_refinements={s:sum(e['finite_reference']==s for e in edges) for s in ('improved','worsened','unresolved')},dimension_pairs=12,dimension_improved=sum(d['physical']=='improved' for d in dimensions),allocation_pairs=6,second_axis_better_physical=sum(a['second_axis_vs_first_physical']=='improved' for a in allocations),second_axis_better_finite=sum(a['second_axis_vs_first_finite']=='improved' for a in allocations),storage=storage,scope='Two real Gaussian drives on two-mode inputs. Equal grid-node counts do not establish equal runtime. No full phase-space rank4 noise or large-scale speed claim.');check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'refinements.json',encode(edges));save(out/'allocations.json',encode(allocations));save(out/'coordinate_changes.json',encode(changes));save(out/'dimensions.json',encode(dimensions));save(out/'checks.json',checks);save(out/'summary.json',summary);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
