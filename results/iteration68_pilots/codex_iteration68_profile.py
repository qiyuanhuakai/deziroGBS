from pathlib import Path
import os,sys,json,hashlib,shutil,cProfile,pstats,time
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import numpy as np
from gbslab.displaced_sampling import DisplacedThresholdSampler
from iteration23_support import load_mps
OUT=ROOT/'.git/iteration68_profile_20261110';OUT.mkdir(exist_ok=False)
sources=[Path(__file__),ROOT/'src/gbslab/displaced_sampling.py',ROOT/'src/gbslab/mps_overlap.py'];hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
for p in sources:target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
inputs=[ROOT/'results/iteration25_panel_retry/run_19.state.npz',ROOT/'results/iteration30_panel/large_random.npz'];deps={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs};start=time.perf_counter();sampler=DisplacedThresholdSampler(load_mps(inputs[0]));prep=time.perf_counter()-start
z=np.load(inputs[1]);beta=z['beta'][:128];u=z['uniforms'][:128];prof=cProfile.Profile();start=time.perf_counter();result=prof.runcall(sampler.sample,beta,u);elapsed=time.perf_counter()-start;prof.dump_stats(str(OUT/'profile.pstats'))
with (OUT/'profile.txt').open('w',encoding='utf-8') as stream:pstats.Stats(prof,stream=stream).strip_dirs().sort_stats('cumtime').print_stats(35)
np.savez_compressed(OUT/'normalized_input.npz',**{f'A{i}':a for i,a in enumerate(sampler.tensors)},beta=beta,uniforms=u)
np.savez_compressed(OUT/'reference.npz',**result)
(OUT/'summary.json').write_bytes((json.dumps(dict(modes=144,samples=128,preparation_seconds=prep,profiled_sampling_seconds=elapsed,tensor_elements=sum(a.size for a in sampler.tensors),max_bond=max(a.shape[2] for a in sampler.tensors),source_hashes=hashes,dependencies=deps,scope='Exploratory baseline profiling; no new native speed claim.'),indent=2)+'\n').encode())
assert all(hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h for n,h in hashes.items());print('profile',elapsed,'preparation',prep)
