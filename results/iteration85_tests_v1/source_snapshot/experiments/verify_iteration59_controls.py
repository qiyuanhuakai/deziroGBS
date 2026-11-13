"""Frozen one/two-drive regressions and rank4 direct-density controls."""
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration47_radicals import density_radicals
from iteration59_independent import density_polynomials,evaluate_density,integrate_density,serialize_polynomial
from gbslab.multidrive_gaussian_noise import integrated_linear_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration59_controls';out.mkdir(exist_ok=False);names=['research/iteration59_noise_plan.md',*[f'experiments/{n}.py' for n in ('verify_iteration59_controls','iteration59_independent','iteration48_independent','iteration47_radicals','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];cache={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    oldcases=0
    for iteration,count in ((54,26),(55,24)):
        manifest=read(f'results/iteration{iteration}_manifest.json');valid=read(f'results/iteration{iteration}_noise_validation/validation.json');check(f'old{iteration} validation',valid['count']==valid['passed'] and deps[f'results/iteration{iteration}_noise_validation/validation.json']==manifest['artifacts'][f'results/iteration{iteration}_noise_validation/validation.json'])
        for i in range(count):
            name=f'results/iteration{iteration}_noise/case_{i:02d}.json';rec=read(name);check(name+' certified bytes',deps[name]==manifest['artifacts'][name]==valid['dependencies'][name]);name=f'results/iteration{iteration}_noise/case_{i:02d}.npz';deps[name]=digest(ROOT/name);check(name+' certified bytes',deps[name]==manifest['artifacts'][name]==valid['dependencies'][name]);z=np.load(ROOT/name);t=[z['tensor_0'],z['tensor_1']];v=[rec['job']['direction']] if iteration==54 else rec['job']['directions'];s=[rec['job']['sigma']] if iteration==54 else rec['job']['sigmas'];key=(tuple(a.tobytes() for a in t),tuple(tuple(tuple(x) for x in d) for d in v),tuple(s))
            if key not in cache:cache[key]=(integrated_linear_law(t,v,s),integrate_density(density_polynomials(t,v),s))
            new,ind=cache[key]
            if iteration==55:
                clean=lambda d:{k:v for k,v in d.items() if k!='scope'};check(f'old2 exact all fields {i}',encode(clean(new))==encode(clean(rec['finite_continuous'])))
            for mask,(a,b) in enumerate(ind['intervals']):
                l,h=new['intervals'][mask];check(f'new regression {iteration}:{i}:{mask}',l<=a<=b<=h);l,h=rec['finite_continuous']['intervals'][mask];check(f'old regression {iteration}:{i}:{mask}',l<=a<=b<=h)
            save(out/f'regression_{iteration}_{i:02d}.json',encode(dict(new=new,independent=ind)));oldcases+=1
        print('regressions complete',iteration,count,flush=True)
    summary=read('results/iteration59_noise/summary.json');check('six completed programs',summary['planned']==summary['finished']==summary['completed']==6);controls=0
    for index in range(6):
        rec=read(f'results/iteration59_noise/case_{index:02d}.json');name=f'results/iteration59_noise/case_{index:02d}.npz';deps[name]=digest(ROOT/name);z=np.load(ROOT/name);t=[z['tensor_0'],z['tensor_1']];directions=rec['job']['directions'];poly=density_polynomials(t,directions);save(out/f'polynomial_{index:02d}.json',encode(serialize_polynomial(poly)));summaries=[]
        for node in (0,1295):
            ij=tuple(int(x) for x in np.unravel_index(node,(6,6,6,6)));xs=[g['nodes'][k] for g,k in zip(rec['grids'],ij)];alpha=np.array([complex(float(sum(v[k][0]*x for v,x in zip(directions,xs))),float(sum(v[k][1]*x for v,x in zip(directions,xs)))) for k in range(2)]);direct=density_radicals(t,alpha);evaluated=evaluate_density(poly,xs)
            for mask,(a,b) in enumerate(zip(direct,evaluated)):
                clean=lambda d:{k:v for k,v in d.items() if v};check(f'exact radical {index}:{node}:{mask}',clean(a['radical_coefficients'])==clean(b['radical_coefficients']));check(f'root {index}:{node}:{mask}',a['coefficient_interval']==b['coefficient_interval'])
            summaries.append(dict(node=node,pair=ij,direct=direct,evaluated=evaluated));controls+=1
        save(out/f'fixed_nodes_{index:02d}.json',encode(summaries));print('rank4 density controls',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),old_cases=oldcases,old_rank1_cases=26,old_rank2_cases=24,unique_old_inputs=len(cache),fixed_density_controls=controls,source_hashes=hashes,dependencies=deps));print('controls complete',len(checks),flush=True)

if __name__=='__main__':main()
