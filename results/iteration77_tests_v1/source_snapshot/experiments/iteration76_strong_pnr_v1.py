"""Higher-input fixed-energy-density stress: pure parity gives a full-TV lower bound.
Three fixed events share failure .01 under ideal iid samples. This is not a full
target upper bound or a fixed-PRNG certificate; every returned photon is retained.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_projected_pnr import NativeProjectedPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
binary=ROOT/'results/iteration76_projected_pilot_v3/projected.dll';save('plan.json',dict(scope=__doc__,driver_sha256=H(Path(__file__)),binary_sha256=H(binary),sizes=[8,32,128],records=[256,256,128],squeezing=.5,retained=7,workers=16,joint_failure=.01,input_seed=760800,seed_base=0x85000000000000,stride=2**40));shutil.copy2(__file__,OUT/Path(__file__).name);runtime=[Path(shutil.which('g++')).parent];rows=[]
for case,(n,count) in enumerate([(8,256),(32,256),(128,128)]):
    m=3*n;rng=np.random.default_rng(760800+case);v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);t=np.full(n,np.tanh(.5));z=mp.mpf(float(t[0]))**2;shape=mp.mpf(n)/2;term=(1-z)**shape;mass=term
    for pairs in range(1,4):term*=z*(shape+pairs-1)/pairs;mass+=term
    np.savez_compressed(OUT/f'input_{case}.npz',v=v,t=t);target=dict(mean=str(n*z/(1-z)),vacuum=str((1-z)**shape),ideal_prefix7_tv_upper=str(1-mass));save(f'target_{case}.json',target);seed=0x85000000000000+case*2**40;save(f'started_{case}.json',dict(inputs=n,outputs=m,records=count,seed=seed));start=time.perf_counter()
    try:
        with NativeProjectedPNRSampler(v,t,binary,retained=7,workers=16,runtime_dirs=runtime) as model:r=model.sample(count,seed)
        counts=r['counts'];np.savez_compressed(OUT/f'counts_{case}.npz',counts=counts);total=counts.sum(axis=1);odd=int((total%2!=0).sum());radius=math.sqrt(math.log(300)/(2*count));row=dict(success=True,case=case,inputs=n,outputs=m,records=count,wall_seconds=time.perf_counter()-start,statistics=r['statistics'],odd_records=odd,odd_fraction=odd/count,nominal_joint99_full_tv_lower=max(0,odd/count-radius),radius=radius,mean_photons=float(total.mean()),maximum_photons=int(total.max()),target=target)
    except Exception as e:row=dict(success=False,case=case,inputs=n,outputs=m,records=count,wall_seconds=time.perf_counter()-start,error=str(e),target=target)
    rows.append(row);save(f'row_{case}.json',row);print(row,flush=True)
save('summary.json',dict(completed=True,all_samples_succeeded=all(r['success'] for r in rows),rows=rows));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
