"""Exact determinant certificates for physical first/pair parity cumulants."""
import sys,json,shutil,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')];sys.set_int_max_str_digits(0)
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import exact_determinant
from gbslab.rational_gaussian import reciprocal_sqrt


def main():
    out=ROOT/'results/iteration40_physical_coefficients';out.mkdir(exist_ok=False);names=['experiments/iteration40_physical_coefficients.py','experiments/verify_iteration34.py','experiments/iteration22_external.py','experiments/iteration33_run.py','src/gbslab/rational_gaussian.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    manifest=json.loads((ROOT/'results/iteration35_manifest.json').read_text())
    for n in ('results/iteration35_panel/inputs.npz','results/iteration35_panel/case_12.audit.json','results/iteration35_validation/validation.json'):
        deps[n]=digest(ROOT/n);assert deps[n]==manifest['artifacts'][n]
    assert json.loads((ROOT/'results/iteration35_panel/case_12.audit.json').read_text())['status']=='strictly_physical';V=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];selection=json.loads((ROOT/'results/iteration40_saturation/selection.json').read_text());n=selection['mode']+1;deps['results/iteration40_saturation/selection.json']=digest(ROOT/'results/iteration40_saturation/selection.json');deps['results/iteration40_saturation/global_target.json']=digest(ROOT/'results/iteration40_saturation/global_target.json');native=json.loads((ROOT/'results/iteration40_saturation/global_target.json').read_text());records=[];vac={};begin=time.perf_counter();save(out/'plan.json',dict(modes=n,output_bits=256,source_hashes=hashes,dependencies=deps,exact_covariance='round35 shifted binary64 entries interpreted as exact rationals; quantum physicality certificate reused'))
    subsets=[(i,) for i in range(n)]+[(j,i) for i in range(n) for j in range(i)]
    for modes in subsets:
        ix=tuple(modes)+tuple(i+144 for i in modes);Q=[[(F(float(V[i,j]))+int(i==j))/2 for j in ix] for i in ix];det=exact_determinant(Q);assert det>0;interval=reciprocal_sqrt((det,det),256);assert 0<=interval[0]<=interval[1]<=1;vac[modes]=interval;records.append(dict(modes=modes,determinant=det,vacuum_interval=interval))
    first=[(2*vac[i,][0]-1,2*vac[i,][1]-1) for i in range(n)];pairs=[]
    for i in range(n):
        for j in range(i):
            a,b=vac[j,i];x,y=vac[j,];z,w=vac[i,];pairs.append(dict(modes=(j,i),interval=(4*(a-y*w),4*(b-x*z))))
    native_pairs={(j,i):F(float.fromhex(x)) for j,i,x in native['pairs']};differences=[max(abs(F(float.fromhex(native['first_hex'][i]))-a),abs(F(float.fromhex(native['first_hex'][i]))-b)) for i,(a,b) in enumerate(first)]+[max(abs(native_pairs[tuple(r['modes'])]-a) for a in r['interval']) for r in pairs];assert max(differences)<F(1,10**10)
    for offset in range(0,len(records),256):save(out/f'certificates_{offset//256:02d}.json',encode(records[offset:offset+256]))
    save(out/'coefficients.json',encode(dict(modes=n,cutoff=2,first_intervals=first,pairs=pairs,scope='Intervals contain exact parity cumulants of the certified physical shifted Gaussian target; coefficient correlations are conservatively forgotten')));assert hashes=={n:digest(ROOT/n) for n in names};save(out/'summary.json',dict(source_hashes=hashes,dependencies=deps,unchanged_sources=True,certificates=len(records),shards=(len(records)+255)//256,first_cumulants=n,pair_cumulants=len(pairs),max_native_coefficient_difference=float(max(differences)),seconds=time.perf_counter()-begin));print('physical coefficients',len(records),'max native difference',float(max(differences)),flush=True)
if __name__=='__main__':main()
