import argparse,shutil,subprocess
from pathlib import Path
from iteration23_support import digest,save
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--compiler',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['experiments/iteration31_reference_build.py','experiments/iteration31_native_reference.cpp','experiments/batched_forest_wire.cpp','experiments/batched_forest_kernel.cpp','experiments/batched_forest_kernel.h','experiments/portable_scalar_codec.cpp'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    exe=out/'native_reference.exe';cmd=[str(a.compiler),'-O2','-std=c++20','-ffp-contract=off','-fno-fast-math',str(ROOT/'experiments/iteration31_native_reference.cpp'),'-o',str(exe),'-lquadmath'];r=subprocess.run(cmd,capture_output=True,text=True);(out/'compiler.stdout.log').write_text(r.stdout);(out/'compiler.stderr.log').write_text(r.stderr)
    info=dict(command=cmd,returncode=r.returncode,source_hashes=hashes,compiler_sha256=digest(a.compiler),unchanged_sources=hashes=={n:digest(ROOT/n) for n in names})
    if exe.is_file():info['executable_sha256']=digest(exe)
    save(out/'build.json',info);print(info['returncode']);assert r.returncode==0 and info['unchanged_sources']
if __name__=='__main__':main()
