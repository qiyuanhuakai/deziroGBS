"""Extended-domain independence, all old-node regression, ideal limit bounds."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration51_normal_independent import cdf,exp_negative,pi_bounds
from gbslab.growing_normal_grid import growing_cdf,scaled_exp_minus,growing_grid
from gbslab.extended_displaced_fock import displaced_finite_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration51_controls';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration51_controls','iteration51_normal_independent','iteration48_independent','iteration22_external','iteration33_run','verify_iteration34')]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];nodes=0
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    for x in [F(k,4) for k in range(-32,33)]:
        a,b=growing_cdf(x);l,h=cdf(x);check('independent extended CDF '+str(x),a<=l<=h<=b)
    for x in [F(k,4) for k in range(257)]:
        a,b=scaled_exp_minus(x);l,h=exp_negative(x);check('independent extended exp '+str(x),a<=l<=h<=b)
    old=read('results/iteration50_noise_validation/validation.json');manifest=read('results/iteration50_manifest.json');check('old validated',old['count']==old['passed'])
    for i in range(36):
        n=f'results/iteration50_noise/case_{i:02d}.json';r=read(n);check('bound old record '+str(i),deps[n]==old['dependencies'][n]);n=f'results/iteration50_noise/case_{i:02d}.npz';deps[n]=digest(ROOT/n);check('bound old tensor '+str(i),deps[n]==old['dependencies'][n]);t=np.load(ROOT/n)['tensor_0'];g=growing_grid(r['job']['sigma'],r['job']['cells'],radius=4);check('all old selector/node regression '+str(i),g['nodes']==r['grid']['nodes'] and g['probabilities_hex']==r['grid']['probabilities_hex'] and g['cdf_hex']==r['grid']['cdf_hex'])
        n=f'results/iteration50_noise_validation/case_{i:02d}.independent_nodes.json';ind=read(n);check('old independent coefficient snapshot '+str(i),deps[n]==manifest['artifacts'][n])
        for j,(component,reference) in enumerate(zip(r['components'],ind)):
            nodes+=1;alpha=component['alpha'];result=displaced_finite_law([t],[complex(float(alpha))]);l,h=reference['coefficient_interval'];el,eh=exp_negative(alpha*alpha);a,b=result['intervals'][0]
            check(f'{i}/{j} extended finite measurement regression',a<=max(F(0),l*el)<=min(F(1),h*eh)<=b and result['vacuum_records'][1]['coefficient_interval']==tuple(component['finite_target']['vacuum_records'][1]['coefficient_interval']) and result['integer_amplitudes']==[tuple(x) for x in component['finite_target']['integer_amplitudes']] and result['integer_norm']==component['finite_target']['integer_norm'])
        print('regression',i,flush=True)
    pl,ph=pi_bounds();check('universal prefactor bound',pl>2)
    # For k>=2: (k+2)/(4*(k+1))<=1/3, and floor((k+2)^2/2)-floor((k+1)^2/2)>=3.
    # Together with e>2 (1+1+1/2>2), these prove geometric decay of the
    # ideal coupling envelope. The table is evidence, not a finite test of infinity.
    check('exp base strict bound',F(1)+F(1)+F(1,2)>2)
    sequence=[]
    for k in range(2,65):
        N=4**k;R=k+1;upper=F(1,2)*(F(R,N)+F(1,2**(R*R//2)));geometric=F(1,8*3**(k-2));check('geometric envelope '+str(k),upper<=geometric)
        if sequence:check('one third decay '+str(k),upper<=sequence[-1]['ideal_TV_upper']/3)
        sequence.append(dict(k=k,cells=N,radius=R,ideal_TV_upper=upper,geometric_upper=geometric,word_grid_count_bound=min(F(1),F(N+1,1<<53)),executed_domain=k<=5))
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'ideal_limit.json',encode(dict(sequence=sequence,scope='Ideal Gaussian displacement coupling only. Count bound is a separate finite-word discrepancy bound relative to a floating CDF, not total selector error; no actual execution above1024 cells.')));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),old_cases=36,old_nodes=nodes,extended_cdf_points=65,extended_exp_points=257,source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
