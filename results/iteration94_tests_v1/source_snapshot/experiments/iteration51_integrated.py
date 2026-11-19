"""Analytic continuous integral controls for all50 frozen displacement grids."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.real_noise_integral import integrated_vacuum
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration51_integrated';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration51_integrated','iteration22_external','iteration33_run','verify_iteration34')]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')];hashes={n:digest(ROOT/n) for n in names};deps={};rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    for index in range(50):
        old=read(f'results/iteration51_noise/case_{index:02d}.json');name=f'results/iteration51_noise/case_{index:02d}.npz';deps[name]=digest(ROOT/name);tensor=np.load(ROOT/name)['tensor_0'];integral=integrated_vacuum(tensor,old['job']['sigma']);program_error=physical_tv_interval(old['mixture_law'],integral['law']+[(F(0),F(0))]);a,b=integral['vacuum_interval'];c,d=old['continuous_target'][0];representation=(a-d,b-c);numerics=(old['mixture_law'][0]-b,old['mixture_law'][0]-a);save(out/f'case_{index:02d}.json',encode(dict(job=old['job'],integral=integral,program_vs_finite_continuous_TV=program_error,finite_continuous_vs_Gaussian_signed_vacuum_bias=representation,program_vs_finite_continuous_signed_vacuum_bias=numerics)));rows.append(dict(case=index,job=old['job'],representation_bias=representation,grid_and_numeric_bias=numerics,program_error=program_error));print('integrated control',index,flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'rows.json',encode(rows));save(out/'summary.json',dict(planned=50,completed=50,unchanged_sources=True,unchanged_dependencies=True,source_hashes=hashes,dependencies=deps,scope='Predeclared analytic continuous integral control for growing grids; not a throughput algorithm'))
if __name__=='__main__':main()
