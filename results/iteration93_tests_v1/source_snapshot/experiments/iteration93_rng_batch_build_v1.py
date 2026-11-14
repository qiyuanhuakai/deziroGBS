"""Build and verify batched purpose5 RNG against frozen84 per-row words."""
import argparse,subprocess
from iteration93_common_v5 import *
from iteration87_guarded_common_v1 import uniforms
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);src=ROOT/'native/gaussian_rng_batch93_v1.cpp';save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),src,ROOT/'native/gaussian_rng_domain84_v1.hpp')}));cmd=[shutil.which('g++'),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math',str(src),'-o',str(out/'rng.dll')];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(out/'rng.dll'));fn=lib.rng93_uniforms;fn.argtypes=[C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,DP,C.c_char_p,C.c_int];checks=[]
for experiment,sector,offset,count,m in ((0,0,0,17,4),(2**64-1,65535,2**32-17,17,1152),(0x9305ffff,1,12345,32,8)):
    observed=np.zeros((count,m));err=C.create_string_buffer(4096);assert fn(experiment,sector,offset,count,m,ptr(observed),err,len(err))==0,err.value;expected=uniforms(experiment,sector,offset,count,m);assert np.array_equal(observed,expected);checks.append(dict(experiment=experiment,sector=sector,offset=offset,records=count,modes=m,words=count*m));np.savez_compressed(out/f'case_{len(checks)}.npz',observed=observed,expected=expected)
for sector,offset,count,m in ((65536,0,1,1),(0,2**32,1,1),(0,0,-1,1),(0,0,1,0),(0,0,1,16385)):
    tmp=np.zeros(1);err=C.create_string_buffer(4096);assert fn(0,sector,offset,count,m,ptr(tmp),err,len(err))==1
def worker_seed_python(seed):
    mask=(1<<64)-1;mt=[seed]
    for i in range(1,312):mt.append((6364136223846793005*(mt[-1]^(mt[-1]>>62))+i)&mask)
    for i in range(312):
        x=(mt[i]&0xffffffff80000000)|(mt[(i+1)%312]&0x7fffffff);mt[i]=mt[(i+156)%312]^(x>>1)^(0xb5026f5aa96619e9 if x&1 else 0)
    y=mt[0];y^=(y>>29)&0x5555555555555555;y^=(y<<17)&0x71d67fffeda60000;y^=(y<<37)&0xfff7eee000000000;y^=y>>43;return y&0xffffffff
quality=read(ROOT/'research/iteration93_quality_plan_v1.json');seeds=[];labels=[]
for job in quality['jobs']:
    if job['family']=='frugal':
        for batch in range(job['records']//job['batch_size']):seeds.append(0x93040000+job['id']*1024+batch);labels.append(dict(job=job['name'],batch=batch))
outer=np.array(seeds,np.uint32);inner=np.zeros_like(outer);seedfn=lib.frugal93_worker_seeds;seedfn.argtypes=[C.c_int,C.POINTER(C.c_uint32),C.POINTER(C.c_uint32)];assert seedfn(len(seeds),outer.ctypes.data_as(C.POINTER(C.c_uint32)),inner.ctypes.data_as(C.POINTER(C.c_uint32)))==0;reference=np.array([worker_seed_python(int(s)) for s in outer],np.uint32);assert np.array_equal(inner,reference);assert len(np.unique(inner))==len(inner);save(out,'frugal_worker_seeds.json',[dict(label,external_seed=int(a),worker_seed=int(b)) for label,a,b in zip(labels,outer,inner)]);np.savez_compressed(out/'frugal_worker_seeds.npz',external=outer,worker=inner,independent_python=reference)
headers=[ROOT/'results/iteration75_frugal_cpu_build_v1/include/fgbs'/x for x in ('cpu/cpu_sampler.hpp','core/random_utils.hpp','core/random_simd.hpp','core/type_utils.hpp')];save(out,'frugal_provenance.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in headers},mapping='UniformRandomInt<size_t>(seed,0,ULLONG_MAX), mt19937_64, first word narrowed to unsigned int; one worker. SIMD lanes consume successive values from that same advancing engine.'))
save(out,'validation.json',dict(passed=True,replay=checks,identical_words=sum(c['words'] for c in checks),invalid_arguments=5,frugal_unique_actual_worker_seeds=len(inner),frugal_independent_mt19937_64_replay=True));manifest(out);print('batch RNG and actual Frugal worker-seed replay passed')
