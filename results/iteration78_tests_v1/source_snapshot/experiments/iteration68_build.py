"""Build the explicit Windows CPU and CUDA FP64 trajectory libraries."""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,time,uuid
ROOT=Path(__file__).resolve().parents[1]
def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(path,value):
    with Path(path).open('xb') as stream:stream.write((json.dumps(value,indent=2)+'\n').encode())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--cpu-only',action='store_true');args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if os.name!='nt':raise RuntimeError('This reproducible build driver currently targets Windows')
    names=['experiments/iteration68_build.py','native/displaced_threshold_cpu.cpp','native/displaced_threshold_cuda.cu']
    hashes={name:digest(ROOT/name) for name in names}
    for name in names:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    save(out/'plan.json',dict(source_hashes=hashes,cpu_only=args.cpu_only,arithmetic='FP64; contraction disabled; no fast math',scope='The existing version 2 algorithms are copied byte-for-byte into public native sources.'))
    compiler=Path(shutil.which('g++') or 'missing-g++');command=[str(compiler),'-O3','-march=native','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-Wall','-Wextra',str(ROOT/names[1]),'-o',str(out/'native_cpu.dll')]
    start=time.perf_counter();run=subprocess.run(command,capture_output=True,text=True,errors='replace');save(out/'cpu_build.json',dict(command=command,compiler_sha256=digest(compiler),returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,seconds=time.perf_counter()-start));assert run.returncode==0
    if not args.cpu_only:
        nvcc=Path(shutil.which('nvcc') or 'missing-nvcc');vcvars=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat';assert vcvars.is_file()
        build=Path(os.environ['LOCALAPPDATA'])/'Temp'/('deziro_gbs_cuda68_'+uuid.uuid4().hex);build.mkdir(exist_ok=False)
        shutil.copy2(ROOT/names[2],build/'kernel.cu');(build/'environment.bat').write_text('@call "'+str(vcvars)+'" >nul\n@set\n',encoding='ascii')
        values=subprocess.check_output(['cmd','/d','/c','environment.bat'],cwd=build,text=True,errors='replace');environment=os.environ.copy()
        for line in values.splitlines():
            if '=' in line:key,value=line.split('=',1);environment[key]=value
        command=[str(nvcc),'-O3','-arch=sm_120','--fmad=false','-std=c++17','-shared','-Xcompiler','/MD,/utf-8','kernel.cu','-lcublas','-o','native_cuda.dll']
        start=time.perf_counter();run=subprocess.run(command,cwd=build,env=environment,capture_output=True,text=True,errors='replace');save(out/'cuda_build.json',dict(command=command,compiler_sha256=digest(nvcc),compiler_version=subprocess.check_output([str(nvcc),'--version'],text=True),build_directory=str(build),returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,seconds=time.perf_counter()-start));assert run.returncode==0;shutil.copy2(build/'native_cuda.dll',out/'native_cuda.dll')
    assert hashes=={name:digest(ROOT/name) for name in names}
    save(out/'summary.json',dict(completed=True,source_hashes=hashes,libraries={p.name:digest(p) for p in out.glob('*.dll')}))
    save(out/'manifest.json',dict(artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}));print('CPU/CUDA build completed',flush=True)
if __name__=='__main__':main()
