"""Explicit GNU C++/Eigen build; no compilation or downloads on import."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time
ROOT=Path(__file__).resolve().parents[1]
def digest(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(p,value):
    with Path(p).open('xb') as f:f.write((json.dumps(value,indent=2)+'\n').encode())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--eigen',type=Path,required=True);args=parser.parse_args()
    if os.name!='nt':raise RuntimeError('This driver currently targets Windows x86-64 GNU C++')
    eigen=args.eigen.resolve()
    if not (eigen/'Eigen/Dense').is_file():raise ValueError('Eigen root must contain Eigen/Dense')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    compiler=Path(shutil.which('g++') or 'missing-g++');names=['experiments/iteration69_build.py','native/gaussian_reset_cpu.cpp','src/gbslab/native_gaussian_reset.py']
    hashes={name:digest(ROOT/name) for name in names}
    for name in names:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    save(out/'eigen_headers.json',{p.relative_to(eigen).as_posix():digest(p) for p in (eigen/'Eigen').rglob('*') if p.is_file()})
    command=[str(compiler),'-O3','-march=native','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native/gaussian_reset_cpu.cpp'),'-o',str(out/'gaussian_reset_cpu.dll')]
    save(out/'plan.json',dict(source_hashes=hashes,command=command,compiler_sha256=digest(compiler),scope='Approximate Gaussian-reset PNR law. Single native thread; explicit optional OpenBLAS. x87 64-bit precision guarded and restored. Build uses local Eigen headers, never downloads dependencies.'))
    start=time.perf_counter();run=subprocess.run(command,capture_output=True,text=True,errors='replace');save(out/'build.json',dict(returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,seconds=time.perf_counter()-start));assert run.returncode==0
    assert hashes=={name:digest(ROOT/name) for name in names}
    save(out/'summary.json',dict(passed=True,source_hashes=hashes,dll_sha256=digest(out/'gaussian_reset_cpu.dll')))
    save(out/'manifest.json',dict(artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}));print('Gaussian reset native build passed',flush=True)
if __name__=='__main__':main()
