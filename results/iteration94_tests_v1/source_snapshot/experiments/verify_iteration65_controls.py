"""Frozen d6 input regressions and selected-grid direct-density controls."""
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
from iteration65_support import snapshot
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration65_controls';hashes=snapshot(out);deps={};checks=[];cache={}
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    manifest=read('results/iteration60_manifest.json');oldvalid=read('results/iteration60_noise_validation/validation.json');check('old complete',oldvalid['count']==oldvalid['passed'] and deps['results/iteration60_noise_validation/validation.json']==manifest['artifacts']['results/iteration60_noise_validation/validation.json']);summary=read('results/iteration65_noise/summary.json');check('eleven completed programs',summary['planned']==summary['finished']==summary['completed']==11);controls=0;regressions=0
    for index in range(11):
        rec=read(f'results/iteration65_noise/case_{index:02d}.json');name=f'results/iteration65_noise/case_{index:02d}.npz';deps[name]=digest(ROOT/name);z=np.load(ROOT/name);t=[z['tensor_0'],z['tensor_1']];directions=rec['job']['directions'];key=(tuple(a.tobytes() for a in t),tuple(tuple(tuple(x) for x in v) for v in directions));poly=cache.get(key);poly=density_polynomials(t,directions) if poly is None else poly;cache[key]=poly;save(out/f'polynomial_{index:02d}.json',encode(serialize_polynomial(poly)));summaries=[]
        if index<3:
            baseline=2*index+1;name=f'results/iteration60_noise/case_{baseline:02d}.json';old=read(name);check('baseline JSON bytes '+str(index),deps[name]==manifest['artifacts'][name]==oldvalid['dependencies'][name]);name=f'results/iteration60_noise/case_{baseline:02d}.npz';deps[name]=digest(ROOT/name);check('baseline tensor bytes '+str(index),deps[name]==manifest['artifacts'][name]==oldvalid['dependencies'][name]);zz=np.load(ROOT/name);check('unchanged d6 tensors '+str(index),all(np.array_equal(a,zz[f'tensor_{j}']) for j,a in enumerate(t)));check('unchanged continuous input '+str(index),rec['finite_continuous']==old['finite_continuous'] and rec['bridge']==old['bridge'] and rec['target']==old['target']);regressions+=1
        dims=tuple(N+2 for N in rec['job']['cells']);last=math.prod(dims)-1
        for node in (0,last):
            ij=tuple(int(x) for x in np.unravel_index(node,dims));xs=[g['nodes'][k] for g,k in zip(rec['grids'],ij)];alpha=np.array([complex(float(sum(v[k][0]*x for v,x in zip(directions,xs))),float(sum(v[k][1]*x for v,x in zip(directions,xs)))) for k in range(2)]);direct=density_radicals(t,alpha);evaluated=evaluate_density(poly,xs)
            for mask,(a,b) in enumerate(zip(direct,evaluated)):
                clean=lambda d:{k:v for k,v in d.items() if v};check(f'exact radical {index}:{node}:{mask}',clean(a['radical_coefficients'])==clean(b['radical_coefficients']));check(f'root {index}:{node}:{mask}',a['coefficient_interval']==b['coefficient_interval'])
            summaries.append(dict(node=node,pair=ij,direct=direct,evaluated=evaluated));controls+=1
        save(out/f'fixed_nodes_{index:02d}.json',encode(summaries));print('budget density controls',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),baseline_regressions=regressions,fixed_density_controls=controls,source_hashes=hashes,dependencies=deps));print('controls complete',len(checks),flush=True)

if __name__=='__main__':main()
