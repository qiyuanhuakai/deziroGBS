"""Build coherent core with long double norms and strict floating-point flags."""
import argparse,subprocess
from iteration92_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
source=ROOT/'native/gaussian_coherent_core92_v1.cpp';files=[Path(__file__),source,ROOT/'native/gaussian_rng_domain84_v1.hpp',ROOT/'experiments/iteration92_common_v1.py'];save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));cmd=[shutil.which('g++'),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math',str(source),'-o',str(out/'core.dll')];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));manifest(out);assert r.returncode==0,r.stderr;print('built coherent92',flush=True)
