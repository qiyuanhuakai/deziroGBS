"""Cache calibrated alias plans; require bitwise replay of uncached samples.
Payload bytes exclude vector allocators and Python copies; timings are a pilot.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_cached_transport_pnr import NativeCachedTransportPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';sources=[Path(__file__),ROOT/'src/gbslab/native_cached_transport_pnr.py',*sorted((ROOT/'native').glob('*.cpp')),*sorted((ROOT/'native').glob('*.hpp'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},workers=16,records=100000,wide_cases=[0,1,2,3]));command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native/gaussian_pnr_transport80_v4.cpp'),'-o',str(OUT/'cached.dll')];start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,seconds=time.perf_counter()-start,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('cached built',flush=True)
rows=[]
for case in (0,1,2,3):
    base=ROOT/'results/iteration80_wide_pilot_v1';d=np.load(base/f'input_{case}.npz');weights=d['weights'];table=np.concatenate([np.load(base/f'table_{case}_{first}.npz')['couplings'] for first in range(0,len(weights),64)]);start=time.perf_counter()
    with NativeCachedTransportPNRSampler(d['v'],.5,OUT/'cached.dll',couplings=table,latent_weights=weights,exact_total_budget=0,workers=16,runtime_dirs=runtime) as model:
        built=time.perf_counter()-start;r=model.sample(100000,0xae000000000000+case*2**40,fixed_total=34);old=np.load(base/f'counts_{case}.npz')['counts'];assert np.array_equal(r['counts'],old);a=model.sample(257,0xaf000000000000+case*2**40,workers=1,fixed_total=14);b=model.sample(257,0xaf000000000000+case*2**40,workers=16,fixed_total=14);assert np.array_equal(a['counts'],b['counts']);row=dict(case=case,inputs=d['v'].shape[1],outputs=d['v'].shape[0],constructor_seconds=built,sampling_seconds=r['wall_seconds'],records=100000,alias_payload_bytes=model.alias_payload_bytes,bitwise_uncached_replay=True,worker_replay=True);np.savez_compressed(OUT/f'counts_{case}.npz',counts=r['counts'],actual_weights=model.actual_latent_weights)
    rows.append(row);save(f'case_{case}.json',row);print(row,flush=True)
d=np.load(ROOT/'results/iteration80_ipf_pilot_v1/strong4.npz');t=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz')['t']
with NativeCachedTransportPNRSampler(d['v'],t,OUT/'cached.dll',couplings=d['couplings'],latent_weights=d['weights'],exact_total_budget=64,workers=16,runtime_dirs=runtime) as model:r=model.sample(10000,0x88000000000000+2**40)
assert np.array_equal(r['counts'],np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')['counts']);np.savez_compressed(OUT/'strict_replay.npz',counts=r['counts'])
save('summary.json',dict(passed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
