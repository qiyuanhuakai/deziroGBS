"""Exact output equality with old repeated-path transfers and Fock enumeration."""
import sys,json,shutil,time
from pathlib import Path
import numpy as np
from iteration22_external import digest,save
from iteration52_prefix_density import integer_density_law
from iteration45_independent import integer_density_law as old_law
from gbslab.exact_mps_law import represented_threshold_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration52_prefix_density_validation';out.mkdir(exist_ok=False);names=['research/iteration52_audit_plan.md',*[f'experiments/{n}.py' for n in ('verify_iteration52_prefix_density','iteration52_prefix_density','iteration45_independent','iteration22_external')],'src/gbslab/exact_mps_law.py'];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    cases=[]
    for seed,n,d,bond in [(52001,2,3,2),(52002,3,3,3),(52003,4,3,4),(52004,4,4,5)]:
        rng=np.random.default_rng(seed);bonds=[1]+[bond]*(n-1)+[1];cases.append((f'random{seed}',[(rng.integers(-4,5,size=(bonds[j],d,bonds[j+1]))+1j*rng.integers(-4,5,size=(bonds[j],d,bonds[j+1])))/8 for j in range(n)]))
    n='results/iteration52_mps/case_00.npz';deps[n]=digest(ROOT/n);z=np.load(ROOT/n);cases.append(('eight_mode_rank32_d3',[z[f'tensor_{j}'] for j in range(8)]))
    cases.append(('zero_branches',[np.array([1,0,0],complex).reshape(1,3,1)]*4))
    for label,tensors in cases:
        started=time.perf_counter();old=old_law(tensors);old_seconds=time.perf_counter()-started;new=integer_density_law(tensors,return_stats=True);exact=represented_threshold_law(tensors,max_states=65536);check(label+' identical integer masses norm scale law',new[:4]==old);check(label+' independent Fock enumeration',new[0]==exact['integer_threshold_masses'] and new[1]==exact['integer_norm'] and new[2]==exact['amplitude_denominator'] and new[3]==exact['distribution']);check(label+' complete shared traversal',new[4]['prefix_transfers']==2*((1<<len(tensors))-1) and new[4]['leaf_outputs']==1<<len(tensors));rows.append(dict(label=label,old_seconds=old_seconds,**new[4]));print('prefix control',label,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'timings.json',rows);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=6,source_hashes=hashes,dependencies=deps,scope='Exact reuse of identical threshold-prefix density; timing observations concurrent, not isolated benchmark'))
if __name__=='__main__':main()
