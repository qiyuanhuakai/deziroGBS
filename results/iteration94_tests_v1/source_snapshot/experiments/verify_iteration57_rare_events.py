"""All predeclared complete rare outputs: relative bias and conditional shape."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration57_rare_events';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration57_rare_events','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    validation=read('results/iteration57_comparisons/validation.json');check('complete prior audit',validation['count']==validation['passed']>0)
    target=read('results/iteration57_target/target.json');check('target bound',deps['results/iteration57_target/target.json']==validation['dependencies']['results/iteration57_target/target.json'])
    for k in range(1,5):
        path=f'results/iteration57_overlap/global_{k}.atoms.json';atoms=read(path);check(f'K{k} bound selected atoms',deps[path]==validation['dependencies'][path] and [a['output'] for a in atoms]==[r['output'] for r in target['targets']]);q=[a['probabilities']['uint64'] for a in atoms];p=[a['target_interval'] for a in atoms];Q=sum(q);A=sum(a for a,b in p);B=sum(b for a,b in p);check(f'K{k} positive rare mass',0<Q<=1 and 0<A<=B<=1)
        ratios=[];conditional=[]
        for i,(x,(a,b)) in enumerate(zip(q,p)):
            check(f'K{k} atom{i} physical binding',(a,b)==tuple(target['targets'][i]['probability_interval']) and 0<a<=b)
            ratio=(x/b,x/a);check(f'K{k} atom{i} ratio division',ratio[0]*b==x==ratio[1]*a and ratio[0]<=ratio[1])
            # Sharper normalized atom intervals use its own endpoint in numerator and denominator.
            lo=a/(a+B-b);hi=b/(b+A-a);check(f'K{k} atom{i} conditional interval',0<=lo<=hi<=1 and lo*(a+B-b)==a and hi*(b+A-a)==b)
            ratios.append(dict(output=atoms[i]['output'],program_probability=x,target_interval=(a,b),program_over_target_interval=ratio));conditional.append((lo,hi))
        qcond=[x/Q for x in q];check(f'K{k} normalized conditional program',sum(qcond)==1 and sum(a for a,b in conditional)<=1<=sum(b for a,b in conditional));tv=(sum(max(F(0),a-x,x-b) for x,(a,b) in zip(qcond,conditional))/2,sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(qcond,conditional))/2)
        # A different interval construction via a common denominator must enclose these bounds.
        loose=[(a/B,b/A) for a,b in p];check(f'K{k} normalized endpoint refinement',all(c<=a<=b<=d for (a,b),(c,d) in zip(conditional,loose)))
        event_ratio=(Q/B,Q/A);absolute_gap=(max(F(0),A-Q),max(F(0),B-Q));check(f'K{k} event probability and ratio',event_ratio[0]*B==Q==event_ratio[1]*A and absolute_gap[0]<=absolute_gap[1])
        rows.append(dict(cutoff=k,atom_ratios=ratios,selected_program_mass=Q,selected_target_interval=(A,B),program_over_target_event_interval=event_ratio,target_over_program_event_interval=(A/Q,B/Q),absolute_event_TV_lower=absolute_gap[0],conditional_program=qcond,conditional_target_intervals=conditional,conditional_TV_interval=tv,all_selected_outputs_underrepresented=all(r['program_over_target_interval'][1]<1 for r in ratios)))
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=len(checks),cutoffs=4,complete_output_ratios=20,conditional_events=4,source_hashes=hashes,dependencies=deps));save(out/'summary.json',dict(completed=True,scope='All five predeclared rare complete outputs are analyzed. Conditional distances concern this tiny event only, not the unconditional full law. No new empirical sampling or event selection.'));print('rare checks',len(checks),flush=True)


if __name__=='__main__':main()
