"""Build and cross-check the prefix-tree recorder for complete small and large-prefix laws."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,subprocess,time,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
base=ROOT/'results/iteration75_frugal_cpu_build_v1';command=json.loads((base/'build_64.json').read_text())['command'];command=[str(ROOT/'native/frugal_prefix81_adapter_v1.cpp') if s==str(ROOT/'native/frugal_cpu75_adapter_v1.cpp') else s for s in command];command[-1]=str(OUT/'prefix.dll');command[1:1]=['-I',str(ROOT/'results/iteration39_build/generated')];paths=[Path(__file__),ROOT/'native/frugal_prefix81_adapter_v1.cpp',ROOT/'native/frugal_cpu75_adapter_v1.cpp',ROOT/'results/iteration39_build/generated/frugal_extended_hooks.hpp'];save('plan.json',dict(command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in paths},parent_manifest=H(base/'manifest.json'),scope=__doc__))
for p in paths:shutil.copy2(p,OUT/p.name)
start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr
save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Prefix audit library built',flush=True)
