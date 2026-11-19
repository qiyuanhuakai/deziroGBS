"""Old rank-one integrals and exact cached-polynomial/direct-density controls."""
import sys,json,shutil,gzip
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration47_radicals import density_radicals
from iteration55_independent import density_polynomials,evaluate_density,integrate_density,serialize_polynomial
from gbslab.linear_gaussian_noise import integrated_linear_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration55_controls';out.mkdir(exist_ok=False);names=['research/iteration55_noise_plan.md',*[f'experiments/{n}.py' for n in ('verify_iteration55_controls','iteration55_independent','iteration48_independent','iteration47_radicals','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];cache={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    old=read('results/iteration54_manifest.json');v=read('results/iteration54_noise_validation/validation.json');check('old complete',v['count']==v['passed'])
    for i in range(26):
        name=f'results/iteration54_noise/case_{i:02d}.json';rec=read(name);check('old record bytes '+str(i),deps[name]==old['artifacts'][name]);name=f'results/iteration54_noise/case_{i:02d}.npz';deps[name]=digest(ROOT/name);check('old tensor bytes '+str(i),deps[name]==old['artifacts'][name]);z=np.load(ROOT/name);t=[z['tensor_0'],z['tensor_1']];direction=rec['job']['direction'];sigma=rec['job']['sigma'];key=(tuple(a.tobytes() for a in t),tuple(tuple(x) for x in direction),sigma)
        if key not in cache:cache[key]=(integrated_linear_law(t,[direction],[sigma]),integrate_density(density_polynomials(t,[direction]),[sigma]))
        new,ind=cache[key]
        for mask,(a,b) in enumerate(ind['intervals']):
            l,h=new['intervals'][mask];check(f'new rank-one {i}:{mask}',l<=a<=b<=h);l,h=rec['finite_continuous']['intervals'][mask];check(f'old rank-one {i}:{mask}',l<=a<=b<=h)
        save(out/f'regression_{i:02d}.json',encode(dict(new=new,independent=ind)))
    controls=0
    for index in (0,4,8,12,16,20):
        rec=read(f'results/iteration55_noise/case_{index:02d}.json');name=f'results/iteration55_noise/case_{index:02d}.npz';deps[name]=digest(ROOT/name);z=np.load(ROOT/name);t=[z['tensor_0'],z['tensor_1']];directions=rec['job']['directions'];poly=density_polynomials(t,directions);save(out/f'polynomial_{index:02d}.json',encode(serialize_polynomial(poly)));summaries=[]
        for i,j in ((0,0),(1,2),(4,5),(5,4)):
            xs=[rec['grids'][0]['nodes'][i],rec['grids'][1]['nodes'][j]];alpha=np.array([complex(float(sum(v[k][0]*x for v,x in zip(directions,xs))),float(sum(v[k][1]*x for v,x in zip(directions,xs)))) for k in range(2)]);direct=density_radicals(t,alpha);evaluated=evaluate_density(poly,xs)
            for mask,(a,b) in enumerate(zip(direct,evaluated)):
                # A polynomial can retain zero radical groups absent in the direct sum.
                clean=lambda d:{k:v for k,v in d.items() if v};check(f'exact radical control {index}:{i}:{j}:{mask}',clean(a['radical_coefficients'])==clean(b['radical_coefficients']));check(f'root control {index}:{i}:{j}:{mask}',a['coefficient_interval']==b['coefficient_interval'])
            summaries.append(dict(pair=(i,j),direct=direct,evaluated=evaluated));controls+=1
        save(out/f'fixed_nodes_{index:02d}.json',encode(summaries));print('density controls',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),old_cases=26,unique_old_inputs=len(cache),fixed_density_controls=controls,source_hashes=hashes,dependencies=deps));print('controls complete',len(checks),flush=True)
if __name__=='__main__':main()
