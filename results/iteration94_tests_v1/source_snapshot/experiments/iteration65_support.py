"""Frozen new inputs, immutable source capture and declared finite input preparation."""
import sys,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from gbslab.gaussian_fock_bridge import gaussian_fidelity,hermite_box
from gbslab.multidrive_gaussian_noise import gaussian_linear_target,integrated_linear_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def new_families():
    basis=[[(1,0),(0,0)],[(0,1),(0,0)],[(0,0),(1,0)],[(0,0),(0,1)]]
    return [dict(family='real_holdout65',pairing=F(5,16),phase=(1,0),rotated=False,directions=basis,sigmas=[F(3,16),F(1,8),F(1,8),F(1,4)],holdout=True),
            dict(family='phase_mixed_holdout65',pairing=F(7,16),phase=(F(5,13),F(12,13)),rotated=True,directions=[[(1,0),(F(1,4),0)],[(0,1),(0,0)],[(0,0),(1,0)],[(0,0),(F(-1,2),1)]],sigmas=[F(1,4),F(3,16),F(1,8),F(3,16)],holdout=True)]

def snapshot(out):
    out.mkdir(exist_ok=False)
    helpers=('iteration22_external','iteration33_run','verify_iteration34','iteration46_displaced_trace','iteration45_independent','iteration47_radicals','iteration48_independent','iteration51_normal_independent','iteration54_independent','iteration59_independent','verify_iteration58_coupling')
    files=sorted({*ROOT.glob('experiments/iteration65*.py'),*ROOT.glob('experiments/verify_iteration65*.py'),*(ROOT/f'experiments/{n}.py' for n in helpers),*ROOT.glob('src/**/*.py'),ROOT/'research/iteration65_noise_plan.md'})
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in files}
    for p in files:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    return hashes

def prepare(job,d,continuous=True):
    shape=[[F(24,25),-F(7,25)],[-F(7,25),-F(24,25)]] if job['rotated'] else [[F(0),F(1)],[F(1),F(0)]]
    br=[[x*job['pairing']*job['phase'][0] for x in row] for row in shape];bi=[[x*job['pairing']*job['phase'][1] for x in row] for row in shape]
    occ,H=hermite_box(br,bi,[d,d]);A=np.array([complex(float(a),float(b))/math.sqrt(math.prod(math.factorial(k) for k in o)) for o,(a,b) in zip(occ,H)]).reshape(d,d);A/=np.linalg.norm(A)
    u,s,vh=np.linalg.svd(A,full_matrices=False);t=[np.ascontiguousarray((u*s).reshape(1,d,d)),np.ascontiguousarray(vh.reshape(d,d,1))]
    bridge=gaussian_fidelity(t,br,bi);target=gaussian_linear_target(br,bi,job['directions'],job['sigmas']);reference=integrated_linear_law(t,job['directions'],job['sigmas']) if continuous else None
    return br,bi,t,bridge,target,reference,s
