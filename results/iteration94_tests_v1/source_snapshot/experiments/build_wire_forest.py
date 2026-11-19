"""Explicit, non-overwriting build of the scalar-wire forest adapter."""
import argparse,json,os,shutil,subprocess
from pathlib import Path
from time import perf_counter
from iteration23_support import digest,save
ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);compiler=a.compiler.resolve()
    names=['experiments/build_wire_forest.py','experiments/batched_forest_wire.cpp','experiments/batched_forest_kernel.cpp','experiments/batched_forest_kernel.h','experiments/portable_scalar_codec.cpp'];hashes={n:digest(ROOT/n) for n in names}
    for name in names:
        dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    dependencies=[]
    for name in ('libquadmath-0.dll','libstdc++-6.dll','libgcc_s_seh-1.dll'):
        path=Path(subprocess.check_output([str(compiler),'-print-file-name='+name],text=True).strip()).resolve()
        if not path.is_file():raise FileNotFoundError(path)
        dependencies.append(dict(path=str(path),sha256=digest(path)))
    library=out/'libwire_forest.dll';command=[str(compiler),'-O2','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-Wall','-Wextra',str(ROOT/'experiments/batched_forest_wire.cpp'),'-o',str(library),'-lquadmath'];start=perf_counter()
    process=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace');(out/'compiler.stdout.log').write_text(process.stdout,encoding='utf-8');(out/'compiler.stderr.log').write_text(process.stderr,encoding='utf-8')
    metadata=dict(command=command,compiler=str(compiler),compiler_sha256=digest(compiler),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True),returncode=process.returncode,build_seconds=perf_counter()-start,source_hashes=hashes,runtime_dependencies=dependencies,abi_version=2,wire_bytes=24,scientific_kernel='Unmodified batched_forest_kernel.cpp; quad-to-native-long-double conversion indicators unchanged',arithmetic_environment='x87 precision 64 and nearest; SSE nearest, FTZ/DAZ disabled; controls restored at exit')
    metadata['unchanged_sources']=hashes=={n:digest(ROOT/n) for n in names}
    if library.is_file():metadata['library_sha256']=digest(library)
    save(out/'build.json',metadata);print(json.dumps(dict(returncode=process.returncode,build_seconds=metadata['build_seconds'],unchanged_sources=metadata['unchanged_sources'])))
    if process.returncode:raise RuntimeError('Build failed; frozen diagnostics retained')
    assert metadata['unchanged_sources']
if __name__=='__main__':main()
