"""Replay both public C++ routes against frozen prototype output records."""
from pathlib import Path
import argparse,os,shutil,sys,json
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from gbslab.native_gaussian_reset_parallel import NativeParallelGaussianResetSampler
from gbslab.native_direct_click import NativeDirectClickSampler
from iteration70_build import digest,save

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--build',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);build=args.build.resolve();reference=args.reference.resolve();runtime=[Path(shutil.which('g++')).parent];blas=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'))
    sources=[Path(__file__),ROOT/'native/gaussian_reset_parallel.cpp',ROOT/'native/gaussian_direct_click.cpp',ROOT/'native/gaussian_reset_cpu.cpp',ROOT/'src/gbslab/native_gaussian_reset_parallel.py',ROOT/'src/gbslab/native_direct_click.py',ROOT/'src/gbslab/native_gaussian_reset.py'];hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    for p in sources:
        dest=out/'source_snapshot'/p.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
    save(out/'plan.json',dict(source_hashes=hashes,libraries={p.name:digest(p) for p in build.glob('*.dll')},scope='Public-source build and interfaces replay frozen prototype PNR and direct-click records. Finite count equality is regression evidence, not a universal distribution theorem.'))
    rows=[];dependencies={}
    for case,selector in [(0,'openblas'),(1,'anchor'),(5,'anchor')]:
        p=ROOT/f'results/iteration69_benchmark/input_{case}.npz';dependencies[p.relative_to(ROOT).as_posix()]=digest(p);d=np.load(p)
        with NativeParallelGaussianResetSampler(d['v'],d['x0'],build/'gaussian_reset_parallel.dll',mean=d['mean'],noise_variance=d['variance'],runtime_dirs=runtime,blas_library=blas,workers=16) as sampler:
            for rep in range(3):
                p=reference/f'iteration70_parallel_pilot_v1_20261110/case_{case}_{rep}_1.npz';dependencies[p.relative_to(ROOT).as_posix()]=digest(p);expected=np.load(p)['counts'];result=sampler.sample(64,704100+case*100+rep,selector=selector);actual=result['counts'];row=dict(route='pnr',case=case,rep=rep,samples=64,mismatches=int(np.count_nonzero(actual!=expected)),statistics=result['statistics']);rows.append(row);save(out/f'pnr_{case}_{rep}.json',row);np.savez_compressed(out/f'pnr_{case}_{rep}.npz',counts=actual);assert row['mismatches']==0,row
    for case in range(4):
        base=ROOT/f'results/iteration69_archive/iteration69_quality_pilot_20261110/input_{case}.npz';core=ROOT/f'results/iteration69_archive/iteration69_displaced_v6_20261110/input_{case}.npz';dependencies[base.relative_to(ROOT).as_posix()]=digest(base);dependencies[core.relative_to(ROOT).as_posix()]=digest(core);a=np.load(base);b=np.load(core)
        for displaced in (False,True):
            label='displaced' if displaced else 'centered';x=b['x0'] if displaced else a['x0'];variance=b['noise_variance'] if displaced else np.zeros(x.shape[0]);p=reference/f'iteration70_direct_click_pilot_v1_20261110/counts_{case}_{label}.npz';dependencies[p.relative_to(ROOT).as_posix()]=digest(p);expected=np.load(p)['clicks'][:100000]
            with NativeDirectClickSampler(a['v'],x,build/'gaussian_direct_click.dll',noise_variance=variance,workers=16,runtime_dirs=runtime) as sampler:result=sampler.sample(100000,708001+case)
            actual=result['clicks'];row=dict(route='direct_click',case=case,label=label,samples=100000,mismatches=int(np.count_nonzero(actual!=expected)),statistics=result['statistics']);rows.append(row);save(out/f'direct_{case}_{label}.json',row);np.savez_compressed(out/f'direct_{case}_{label}.npz',clicks=actual);assert row['mismatches']==0,row
    save(out/'summary.json',dict(passed=True,rows=rows,dependencies=dependencies,unchanged_sources=hashes=={p.relative_to(ROOT).as_posix():digest(p) for p in sources}));save(out/'manifest.json',dict(artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}));print(json.dumps(dict(passed=True,samples=sum(r['samples'] for r in rows))),flush=True)
if __name__=='__main__':main()
