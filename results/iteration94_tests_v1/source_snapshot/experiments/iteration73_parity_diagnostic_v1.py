"""Instrument the preserved polynomial survival failure without repairing it."""
from pathlib import Path
import os,sys,json,shutil,subprocess,hashlib,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_polynomial_pnr import NativePolynomialPNRSampler
OUT=ROOT/'results/iteration73_parity_diagnostic_v1';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
source=ROOT/'native/gaussian_pnr_polynomial73_diagnostic_v1.cpp';gcc=Path(shutil.which('g++'));dll=OUT/'diagnostic.dll';command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(dll)];save('plan.json',dict(command=command,source_sha256=H(source),driver_sha256=H(Path(__file__)),seed=736001,count=100000,workers=1,x=[[-.25,0],[0,.5]],mean=[0,0],scope='Same arithmetic as v1; adds diagnostic exception text only. Every failed batch is retained as failure, no partial records accepted.'))
r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0;runtimes=[gcc.parent];rows=[]
for retained in (1,2,3):
    try:
        with NativePolynomialPNRSampler(np.ones((1,1),complex),np.diag([-.25,.5]),dll,retained=retained,workers=1,runtime_dirs=runtimes) as model:r=model.sample(100000,736001)
        np.savez_compressed(OUT/f'counts_{retained}.npz',counts=r['counts']);row=dict(retained=retained,success=True)
    except ArithmeticError as e:row=dict(retained=retained,success=False,error=str(e))
    rows.append(row);print(row,flush=True)
save('summary.json',dict(rows=rows));shutil.copy2(source,OUT/source.name);shutil.copy2(__file__,OUT/Path(__file__).name)
