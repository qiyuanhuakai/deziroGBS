from pathlib import Path
import os,sys,ctypes as C,shutil,subprocess,json,hashlib
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler,DP
OUT=ROOT/'results/iteration74_drift_v1';OUT.mkdir(exist_ok=False);gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';source=ROOT/'native/gaussian_pnr_drift74_v1.cpp';library=OUT/'drift.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-o',str(library)];r=subprocess.run(command,capture_output=True,text=True,errors='replace');(OUT/'build.json').write_bytes(json.dumps(dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)).encode());assert r.returncode==0,r.stderr
data=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');t=data['t'];x=np.diag(np.r_[-t/(1+t),t/(1-t)])
with NativePolynomialPNRSampler(data['v'],x,library,runtime_dirs=[gcc.parent]) as model:
    model.lib.drift74.argtypes=[C.c_void_p,C.c_int,C.c_uint64,DP,C.c_char_p,C.c_int];out=np.zeros(5);err=C.create_string_buffer(4096);rc=model.lib.drift74(model.handle,1000000,744101,out.ctypes.data_as(DP),err,len(err));result=dict(returncode=rc,error=err.value.decode(),target=np.trace(x)/2,first=out[0],second=out[1],before=out[2],after=out[3],zero=out[4]);print(result,flush=True);(OUT/'summary.json').write_bytes(json.dumps(result,indent=2).encode())
for p in (Path(__file__),source):shutil.copy2(p,OUT/p.name)
(OUT/'manifest.json').write_bytes(json.dumps(dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}),indent=2).encode())
