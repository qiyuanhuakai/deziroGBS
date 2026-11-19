"""Audit actual one-worker Frugal seed derivation for the frozen94 quality plan."""
import argparse
from iteration94_common_v2 import *

def worker_seed_python(seed):
    mask=(1<<64)-1;mt=[seed]
    for i in range(1,312):mt.append((6364136223846793005*(mt[-1]^(mt[-1]>>62))+i)&mask)
    for i in range(312):
        x=(mt[i]&0xffffffff80000000)|(mt[(i+1)%312]&0x7fffffff);mt[i]=mt[(i+156)%312]^(x>>1)^(0xb5026f5aa96619e9 if x&1 else 0)
    y=mt[0];y^=(y>>29)&0x5555555555555555;y^=(y<<17)&0x71d67fffeda60000;y^=(y<<37)&0xfff7eee000000000;y^=y>>43;return y&0xffffffff

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);planpath=ROOT/'research/iteration94_quality_plan_v1.json';quality=read(planpath);runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));dll=ROOT/'results/iteration93_rng_batch_build_v1/rng.dll';lib=C.CDLL(str(dll));labels=[];seeds=[]
for job in quality['jobs']:
    if job['family']=='frugal':
        for batch in range(job['records']//job['batch_size']):seeds.append(0x94040000+job['id']*1024+batch);labels.append(dict(job=job['name'],batch=batch))
outer=np.array(seeds,np.uint32);inner=np.zeros_like(outer);fn=lib.frugal93_worker_seeds;fn.argtypes=[C.c_int,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)];assert fn(len(outer),outer.ctypes.data_as(C.POINTER(C.c_uint32)),inner.ctypes.data_as(C.POINTER(C.c_uint32)))==0
reference=np.array([worker_seed_python(int(seed)) for seed in outer],np.uint32);assert np.array_equal(reference,inner);assert len(set(map(int,inner)))==len(inner)==224
previous=read(ROOT/'results/iteration93_rng_batch_build_v1/frugal_worker_seeds.json');overlap=set(map(int,inner))&{r['worker_seed'] for r in previous};assert not overlap
files=[Path(__file__),planpath,dll,ROOT/'native/gaussian_rng_batch93_v1.cpp',ROOT/'native/gaussian_rng_domain84_v1.hpp']
save(out,'plan.json',dict(scope=__doc__,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files}));save(out,'frugal_worker_seeds.json',[dict(label,external_seed=int(a),worker_seed=int(b)) for label,a,b in zip(labels,outer,inner)]);np.savez_compressed(out/'workers.npz',external=outer,native=inner,independent_python=reference)
save(out,'validation.json',dict(passed=True,actual_worker_seeds=224,unique_actual_worker_seeds=224,overlap_with_iteration93=0,independent_mt19937_64_replay=True,scope='No deterministic worker-seed reuse; no claim that finite PRNG streams are mathematically iid.'));manifest(out);print('224 actual worker seeds independently reproduced; no reuse within94 or against93.',flush=True)
