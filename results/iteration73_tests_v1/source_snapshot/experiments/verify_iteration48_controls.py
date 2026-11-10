"""Closed Gaussian vacuum and analytic geometric Schmidt controls."""
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration48_controls';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration48_controls','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    validation=read('results/iteration48_gaussian_validation/validation.json');check('Gaussian independent audit complete',validation['cases']==60 and validation['count']==validation['passed'])
    for index in range(60):
        name=f'results/iteration48_gaussian/case_{index:02d}.json';r=read(name);check(f'{index} bound independent input',deps[name]==validation['dependencies'][name]);j=r['job'];n=j['modes'];real=r['input']['real'];imag=r['input']['imag'];alpha=([F(0),F(0)] if j['displacement']==0 else [F(1,2),F(-1,4)])[:n];beta=([F(0),F(0)] if j['displacement']==0 else [F(1,4),F(1,2)])[:n]
        # Re(conj(alpha)^T B conj(alpha)), independently expanded in reals.
        quadratic=sum(real[a][b]*(alpha[a]*alpha[b]-beta[a]*beta[b])+imag[a][b]*(alpha[a]*beta[b]+beta[a]*alpha[b]) for a in range(n) for b in range(n));exponent=sum(x*x+y*y for x,y in zip(alpha,beta))-quadratic;full=r['gaussian_target']['records'][-1];check(f'{index} Bargmann full-vacuum exponent',full['exponent']==exponent);check(f'{index} Bargmann full-vacuum normalization',full['determinant']*r['bridge']['physical']['determinant']==1)
        if j['family']=='paired':
            lam=r['input']['parameter'];rank=min(j['bond'],j['dimension']);ideal_fidelity=1-lam**(2*rank);a,b=r['bridge']['full_fidelity_interval'];check(f'{index} analytic Schmidt truncation agreement',max(abs(a-ideal_fidelity),abs(b-ideal_fidelity))<F(1,10**28));rows.append(dict(case=index,job=j,analytic_rank=rank,optimal_rank_fidelity=ideal_fidelity,optimal_rank_trace_lower=lam**rank,certified_state_trace=r['bridge']['full_trace_interval'],measured_Gaussian_TV=r['full_Gaussian_TV'],total_TV_upper=r['total_TV_upper']))
            if j['displacement']==0:
                p00=1-lam**2;target=r['gaussian_target']['intervals'];check(f'{index} paired zero-displacement analytic full law',all(l<=p<=h for (l,h),p in zip(target,(p00,F(0),F(0),1-p00))))
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'paired_controls.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=60,paired_controls=len(rows),source_hashes=hashes,dependencies=deps,scope='Independent closed full-vacuum formula and exact geometric Schmidt control; agreement tolerance applies to constructed float state, not certificate width'))
if __name__=='__main__':main()
