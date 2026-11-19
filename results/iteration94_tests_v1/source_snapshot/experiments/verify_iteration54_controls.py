"""Independent analytic integral controls against every old one-mode record."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration54_independent import continuous_density
from gbslab.rank_one_noise import integrated_finite_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration54_controls';out.mkdir(exist_ok=False);names=['research/iteration54_noise_plan.md',*[f'experiments/{n}.py' for n in ('verify_iteration54_controls','iteration54_independent','iteration48_independent','iteration47_radicals','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};checks=[];deps={};cache={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    manifest=read('results/iteration51_manifest.json');oldv=read('results/iteration51_integrated_validation/validation.json');check('old audit complete',oldv['count']==oldv['passed'])
    for i in range(50):
        n=f'results/iteration51_integrated/case_{i:02d}.json';r=read(n);check('old integral bytes '+str(i),deps[n]==manifest['artifacts'][n]);n=f'results/iteration51_noise/case_{i:02d}.npz';deps[n]=digest(ROOT/n);check('old state bytes '+str(i),deps[n]==manifest['artifacts'][n]);t=np.load(ROOT/n)['tensor_0'];sigma=r['job']['sigma'];key=(t.tobytes(),sigma)
        if key not in cache:cache[key]=(integrated_finite_law([t],[(1,0)],sigma),continuous_density([t],[(F(1),F(0))],sigma))
        new,ind=cache[key]
        for j,(a,b) in enumerate(ind['intervals']):
            l,h=new['intervals'][j];check(f'new independent {i}:{j}',l<=a<=b<=h);l,h=r['integral']['law'][j];check(f'old independent {i}:{j}',l<=a<=b<=h)
        save(out/f'case_{i:02d}.json',encode(dict(new=new,independent=ind)))
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),old_cases=50,unique_inputs=len(cache),source_hashes=hashes,dependencies=deps));print('old controls',len(checks),flush=True)
if __name__=='__main__':main()
