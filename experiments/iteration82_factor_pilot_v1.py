"""Build factor-rejection PNR sampler; complete large first-ever sampling,
small actual-law export, worker reproducibility and unchanged exact low branch.
Pilot timings are exploratory; final comparisons require isolated repetitions.
"""
from pathlib import Path
import os,sys,json,hashlib,shutil,subprocess,argparse,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_factor_pnr import NativeFactorPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';runtime=[gcc.parent];sources=[Path(__file__),ROOT/'src/gbslab/native_factor_pnr.py',*sorted((ROOT/'native').glob('*.cpp')),*sorted((ROOT/'native').glob('*.hpp'))]
save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},workers=16,fixed_records=100000,fixed_total=34,unconditional_large_records=100000,squeezing=np.tanh(1.1)))
command=[str(gcc),'-O3','-std=c++20','-shared','-fopenmp','-ffp-contract=off','-fno-fast-math','-I',str(eigen),str(ROOT/'native/gaussian_pnr_factor82_v1.cpp'),'-o',str(OUT/'factor.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));assert r.returncode==0,r.stderr;print('Factor sampler built',flush=True)
base=ROOT/'results/iteration81_factor_pilot_v2';rows=[]
for old in json.loads((base/'summary.json').read_text())['rows']:
 if not old['success']:continue
 label=old['label'];d=np.load(base/f'input_{label}.npz');w=d['weights'];k=d['K'];v=d['v'];chunks=[np.load(base/f'factors_{label}_{first}.npz') for first in range(0,len(w),256)];aa=np.concatenate([f['left'] for f in chunks]);bb=np.concatenate([f['right'] for f in chunks]);start=time.perf_counter()
 try:
  with NativeFactorPNRSampler(v,np.tanh(1.1),OUT/'factor.dll',kernel=k,left=aa,right=bb,latent_weights=w,exact_total_budget=8,workers=16,runtime_dirs=runtime) as model:
   constructor=time.perf_counter()-start;row=dict(label=label,inputs=v.shape[1],outputs=v.shape[0],latent=len(w),constructor_seconds=constructor,statistics=model.plan_statistics,passed=True)
   if len(w)<=512 and len(v)<=16:
    law=model.law();P=k[None]*aa[:,:,None]*bb[:,None,:];P/=P.sum((1,2))[:,None,None];actual=law['oriented_pairs'];tv=abs(P-actual).sum((1,2))/2;row['maximum_oriented_pair_numeric_tv']=float(tv.max());assert tv.max()<1e-10
    np.savez_compressed(OUT/f'law_{label}.npz',**law,ideal_oriented_pairs=P,ideal_weights=w)
    for l in range(len(w)):np.savez_compressed(OUT/f'integer_{label}_{l}.npz',**model.integer_tables(l))
   count=100000 if label.startswith('wide') or label=='large128' else 4096;r=model.sample(count,0x82000000000000,fixed_total=34);assert np.all(r['counts'].sum(1)==34);row.update(fixed_records=count,fixed_seconds=r['wall_seconds'],fixed_statistics=r['statistics']);np.savez_compressed(OUT/f'counts_{label}.npz',counts=r['counts'])
   x=model.sample(257,820081,workers=1,fixed_total=14);y=model.sample(257,820081,workers=16,fixed_total=14);assert np.array_equal(x['counts'],y['counts']);row['worker_replay']=True
   if label=='large128':
    r=model.sample(100000,820128);assert np.array_equal(r['counts'].sum(1),r['total_photons']) and np.all(r['total_photons']%2==0);row.update(unconditional_seconds=r['wall_seconds'],unconditional_statistics=r['statistics'],mean_photons=float(r['total_photons'].mean()));np.savez_compressed(OUT/'unconditional_large128.npz',counts=r['counts'],totals=r['total_photons'])
 except (ArithmeticError,ValueError) as e:row=dict(label=label,passed=False,error=str(e))
 rows.append(row);save(f'row_{label}.json',row);print(row,flush=True)
# Strict low-total replay uses the same seed, input and exact budget as iteration80.
d=np.load(base/'input_strong4.npz');f=np.load(base/'factors_strong4_0.npz');t=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz')['t']
with NativeFactorPNRSampler(d['v'],t,OUT/'factor.dll',kernel=d['K'],left=f['left'],right=f['right'],latent_weights=d['weights'],exact_total_budget=64,workers=16,runtime_dirs=runtime) as model:r=model.sample(10000,0x88000000000000+2**40)
assert np.array_equal(r['counts'],np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')['counts']);np.savez_compressed(OUT/'strict_replay.npz',counts=r['counts'])
save('summary.json',dict(completed=True,all_passed=all(r['passed'] for r in rows),rows=rows,strict_low_branch_replay=True));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
