"""Replay the archived quality inputs through the public native interface."""
from pathlib import Path
import argparse,json,os,shutil,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from gbslab.native_gaussian_reset import NativeGaussianResetSampler
from iteration69_build import digest,save
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--library',type=Path,required=True);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--diagonal-reference',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);reference=args.reference.resolve();diagonal=args.diagonal_reference.resolve();runtime=[Path(shutil.which('g++')).parent]
    sources=[Path(__file__),ROOT/'src/gbslab/native_gaussian_reset.py',ROOT/'native/gaussian_reset_cpu.cpp'];hashes={p.relative_to(ROOT).as_posix():digest(p) for p in sources}
    for p in sources:
        target=out/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    save(out/'plan.json',dict(source_hashes=hashes,library_sha256=digest(args.library),samples_per_case_selector=50000,scope='Replay four frozen inputs and seeds through public assembly and Python interface. Compare counts with archived prototype outputs. Count equality is a finite regression result, not a full-law certificate.'))
    rows=[];dependencies={}
    for case in range(4):
        path=reference/f'input_{case}.npz';data=np.load(path);dependencies[str(path)]=digest(path)
        with NativeGaussianResetSampler(data['v'],data['x0'],args.library,mean=data['mean'],noise_variance=data['noise_variance'],runtime_dirs=runtime) as model:
            for selector,folder,method in [('anchor',reference,5),('diagonal',diagonal,3)]:
                target=folder/f'case_{case}_method_{method}.npz';dependencies[str(target)]=digest(target);expected=np.load(target)['counts'][:50000]
                result=model.sample(50000,690801+case,selector=selector);counts=result['counts'];passed=bool(np.array_equal(counts,expected));row=dict(case=case,selector=selector,passed=passed,mismatched_entries=int(np.count_nonzero(counts!=expected)),statistics=result['statistics']);rows.append(row);save(f'case_{case}_{selector}.json',row);np.savez_compressed(out/f'case_{case}_{selector}.npz',counts=counts);print(row,flush=True)
    save(out/'summary.json',dict(passed=all(row['passed'] for row in rows),rows=rows,dependencies=dependencies,unchanged_sources=hashes=={p.relative_to(ROOT).as_posix():digest(p) for p in sources}));save(out/'manifest.json',dict(artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}));assert all(row['passed'] for row in rows)
if __name__=='__main__':main()
