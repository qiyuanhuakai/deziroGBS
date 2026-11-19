"""Post-inspection control: same tensor entries, different memory layouts."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration45_mps_trace import enumerate_trace,public_calibration
from iteration45_independent import grid_below
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration45_serialization';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration45_serialization','iteration45_mps_trace','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    for i in range(18):
        base='results/iteration45_mps';old=read(base+f'/case_{i:02d}.json');layout=read(f'results/iteration45_mps_recovery/case_{i:02d}.layout.json');name=base+f'/case_{i:02d}.npz';deps[name]=digest(ROOT/name);stored=np.load(ROOT/name);tensors=[stored[f'tensor_{j}'] for j in range(4)];sampling=enumerate_trace(tensors);calibration=public_calibration(tensors,sampling['trace'],452000+i);trace={tuple(r['path']):r for r in sampling['trace']};layers={():[0,F(1),F(1)]};real=[F(0)]*17;word=[F(0)]*17
        for depth in range(4):
            nxt={}
            for path,(mask,rm,wm) in layers.items():
                node=trace[path];check(f'{i} input {path}',node['status']=='completed' and node['real_mass']==rm and node['word_mass']==wm and node['mask']==mask);p=np.array([float.fromhex(x) for x in node['probabilities_hex']]);cdf=p.cumsum();cdf/=cdf[-1];check(f'{i} CDF {path}',[float(x).hex() for x in cdf]==node['cdf_hex']);previous=F(0);last=0
                for c,v in enumerate(cdf):
                    edge=F(float(v));upper=grid_below(edge,53);r=rm*(edge-previous);w=wm*F(upper-last,1<<53);previous=edge;last=upper
                    if not r and not w:continue
                    if float.fromhex(node['weights_hex'][c])<=0:real[-1]+=r;word[-1]+=w
                    elif depth==3:real[mask|((c!=0)<<depth)]+=r;word[mask|((c!=0)<<depth)]+=w
                    else:nxt[path+(c,)]=[mask|((c!=0)<<depth),r,w]
            layers=nxt
        check(f'{i} independent complete laws',real==sampling['real_law'] and word==sampling['word_law'] and sum(real)==sum(word)==1);law=old['represented']['distribution']+[F(0)];change=sum(abs(a-b) for a,b in zip(word,old['sampling']['word_law']))/2;distance=sum(abs(a-b) for a,b in zip(word,law))/2;save(out/f'case_{i:02d}.json',encode(dict(job=old['job'],layout=layout,sampling=sampling,calibration=calibration,TV_layout_change=change,TV_to_stored_state=distance)));rows.append(dict(case=i,layout_changed=layout['rebuilt_strides']!=layout['serialized_strides'],law_changed=change>0,TV_layout_change=change,TV_to_stored_state=distance));print('serialization control',i,float(change),flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=18,source_hashes=hashes,dependencies=deps,post_inspection=True));save(out/'summary.json',dict(planned=18,completed=18,unchanged_sources=True,unchanged_dependencies=True,laws_changed=sum(r['law_changed'] for r in rows),max_TV_layout_change=max(float(r['TV_layout_change']) for r in rows),max_TV_to_stored_state=max(float(r['TV_to_stored_state']) for r in rows)))
if __name__=='__main__':main()
