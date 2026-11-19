"""Frozen input panel and provenance for actual Gaussian-SVD program closure."""
import sys,json,shutil,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.set_int_max_str_digits(0)

def encode(x):
    if isinstance(x,F):return {'rational':str(x)}
    if isinstance(x,np.ndarray):return encode(x.tolist())
    if isinstance(x,np.generic):return encode(x.item())
    if isinstance(x,complex):return {'complex_hex':[x.real.hex(),x.imag.hex()]}
    if isinstance(x,dict):return {str(k):encode(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [encode(v) for v in x]
    return x

def save(p,x):p.write_bytes((json.dumps(encode(x),indent=2)+'\n').encode())
def digest(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def snapshot(out):
    out.mkdir(parents=True,exist_ok=False)
    helpers=('iteration24_reference','iteration46_displaced_trace','iteration47_radicals','iteration48_independent','iteration54_independent','iteration45_independent','iteration46_displacement_independent','iteration51_normal_independent','verify_iteration34','iteration22_external','iteration33_run')
    files=list((ROOT/'experiments').glob('iteration66*.py'))+list((ROOT/'experiments').glob('verify_iteration66*.py'))+[ROOT/'experiments'/f'{n}.py' for n in helpers]+list((ROOT/'src').rglob('*.py'))+[ROOT/'research/iteration66_gsvd_plan.md']
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    for n in hashes:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    return hashes

def families():
    raw=[('pair2',[[F(0),F(1,3)],[F(1,3),F(0)]],(F(1),F(0))),('phase2',[[F(48,125),F(-14,125)],[F(-14,125),F(-48,125)]],(F(3,5),F(4,5))),('chain3',[[F(1,5),F(1,5),F(0)],[F(1,5),F(-1,5),F(1,5)],[F(0),F(1,5),F(0)]],(F(1),F(0))),('dense_phase3',[[F(1,2),F(1,3),F(1,4)],[F(1,3),F(-1,4),F(1,5)],[F(1,4),F(1,5),F(1,3)]],(F(3,5),F(4,5)))]
    out=[dict(family=n,real=[[x*p[0] for x in row] for row in a],imag=[[x*p[1] for x in row] for row in a],modes=len(a),new_holdout=False) for n,a,p in raw]
    out.append(dict(family='mixed_holdout3',real=[[F(1,5),F(1,8),F(1,10)],[F(1,8),F(-1,6),F(1,9)],[F(1,10),F(1,9),F(1,7)]],imag=[[F(1,10),F(-1,12),F(1,14)],[F(-1,12),F(1,9),F(1,16)],[F(1,14),F(1,16),F(-1,10)]],modes=3,new_holdout=True))
    return out

def constructions():return [dict(family_index=i,**f,dimension=d,rank=r,action=a) for i,f in enumerate(families()) for a,d,r in [('baseline',4,2),('rank',4,6),('dimension',6,2),('joint',6,6)]]
def displacement(index,modes):return np.array([[0,0,0],[.25+.125j,-.125+.25j,.1875-.125j],[-.5+.25j,.375-.25j,.125+.375j]][index][:modes],complex)
