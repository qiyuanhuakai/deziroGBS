"""Paired unchanged-budget PNR traces, plus new small and 1152-port records."""
import argparse
import subprocess
from iteration94_common_v2 import ROOT, Path, np, H, save, manifest, load93, C, DP, UP, IP, ptr, time, shutil

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
native=ROOT/'native'
source=native/'gaussian_mps_adaptive95_v1.cpp'
archive=ROOT/'results/iteration95_order_extension_v2'
actual=[ROOT/f'results/iteration94_actual_quality_v1/input_{j}.npz' for j in (0,1)]
small=ROOT/'results/iteration93_reference_v2/input_0.npz'
files=[source,native/'gaussian_mps_adaptive95_steps.hpp',native/'gaussian_rng_domain84_v1.hpp',
       Path(__file__),*actual,small,archive/'manifest.json',ROOT/'results/iteration94_build_v2/mps.dll']
save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},
    target_per_projection=.0006,actual_initial=[6,32],actual_caps=[8,48],budgets=[.1,.01],photon_cap=256,
    paired_replay='12 prior records both spectra, original order, fixed and adaptive, common experiment 0x95060000',
    fresh='8 new paired records per method and spectrum, experiment 0x95a00000, 4 workers, same domains only across methods',
    ablation='first prior failure row 3, main spectrum, bond growth alone and Fock growth alone',
    small='four modes, 16384 paired fresh records, 4 workers, cap64, d4 D4 vs caps8/16, budget .02, experiment 0x95a01000',
    small_confidence='two count CDFs share .01 family error by Bonferroni, DKW vs numerical reference on totals through 64 plus failure sentinel',
    trace=['selected_port','charge_before','sqrt_initial_fock_tail','sqrt_final_fock_tail',
           'sqrt_sweep_loss','max_cut_discard_fraction','worst_cut','cuts_grown','peak_rank','sites_grown',
           'charged_distance','largest_leak_site','peak_local_dimension','expanded_hazard_relative_difference']))
cmd=[shutil.which('g++'),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off',
     '-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),
     str(source),'-o',str(out/'adaptive.dll')]
b=subprocess.run(cmd,capture_output=True,text=True,errors='replace')
save(out,'build.json',dict(command=cmd,returncode=b.returncode,stderr=b.stderr))
b.check_returncode()
old,runtime=load93();lib=C.CDLL(str(out/'adaptive.dll'))
cfg=lib.mps93_set_lapack
cfg.argtypes=[C.c_void_p]*3
assert cfg(*[C.cast(getattr(old._blas,f),C.c_void_p) for f in ('scipy_LAPACKE_zgesdd','scipy_LAPACKE_zgeqrf','scipy_LAPACKE_zungqr')])==0
save(out,'backend.json',old._backend)
I=C.c_int
fn=lib.mps95_sample
fn.argtypes=[I,I,DP,DP,DP,DP,I,I,C.c_double,I,I,C.c_uint64,C.c_uint,C.c_uint64,I,
             UP,DP,IP,C.c_char_p,I,I,I,C.c_double,DP]
def run(z,experiment,sector,offset,n,dim,rank,max_dim,max_rank,budget,cap):
    u,t,alpha,noise=[np.ascontiguousarray(z[k]) for k in ('u','t','alpha','noise')]
    counts=np.empty((n,len(u)),np.uint32);stats=np.empty((n,18));errors=np.empty(n,np.int32)
    trace=np.empty((n,cap,14));msg=C.create_string_buffer(4096)
    start=time.perf_counter()
    rc=fn(len(u),len(t),ptr(u.view(float)),ptr(t),ptr(alpha.view(float)),ptr(noise),dim,rank,budget,
          cap,n,experiment,sector,offset,min(n,4),counts.ctypes.data_as(UP),ptr(stats),errors.ctypes.data_as(IP),
          msg,len(msg),max_dim,max_rank,.0006,ptr(trace))
    seconds=time.perf_counter()-start
    assert rc==int(any(errors)) and np.isin(errors,[0,2]).all(),(rc,errors,msg.value)
    assert np.array_equal(stats[:,0],counts.sum(1))
    for c,s,e,tlog in zip(counts,stats,errors,trace):
        accepted=int(s[0]);steps=tlog[:accepted+int(e==2)]
        assert np.array_equal(np.bincount(steps[:accepted,0].astype(int),minlength=len(u)),c)
        if len(steps):
            assert np.max(abs(steps[:,1]-np.r_[0.,np.cumsum(steps[:-1,10])]))<1e-11
            expected=steps[:,3]**2+(1-steps[:,3]**2)*steps[:,4]**2
            assert np.max(abs(expected-steps[:,10]**2))<1e-13
            assert abs(sum(steps[:accepted,10])-s[5])<1e-11 and s[5]<=budget
            if e:assert steps[-1,1]+steps[-1,10]>budget
    return dict(counts=counts,statistics=stats,errors=errors,trace=trace),seconds

timings=[]
for group,experiment,n in [('replay',0x95060000,12),('fresh',0x95a00000,8)]:
    for off in range(0,n,4):
        for sector in (0,1):
            z=dict(np.load(actual[sector]))
            for adaptive in ((0,1) if (off//4+sector)%2==0 else (1,0)):
                dims=(8,48) if adaptive else (6,32)
                raw,elapsed=run(z,experiment,sector,off,4,6,32,*dims,(.1,.01)[sector],256)
                np.savez_compressed(out/f'{group}_{off}_{sector}_{adaptive}.npz',**raw)
                if group=='replay' and not adaptive:
                    before=np.load(archive/f'raw_{sector}_{off}_0.npz')
                    for k in ('counts','errors'):assert np.array_equal(raw[k],before[k]),k
                    assert np.max(abs(raw['statistics']-before['statistics']))<1e-10
                timings.append(dict(group=group,offset=off,sector=sector,adaptive=adaptive,seconds=elapsed))
                save(out,'timings.json',timings)
                print(group,off,sector,adaptive,raw['errors'].tolist(),round(elapsed,2),flush=True)
for tag,md,mr in [('bond',6,48),('site',8,32)]:
    raw,elapsed=run(dict(np.load(actual[0])),0x95060000,0,3,1,6,32,md,mr,.1,256)
    np.savez_compressed(out/f'ablation_{tag}.npz',**raw)
    save(out,f'ablation_{tag}.json',dict(seconds=elapsed,errors=raw['errors'].tolist()))
z=dict(np.load(small));small_quality=[]
for adaptive in (0,1):
    all_counts=[];all_errors=[];seconds=0
    for off in range(0,16384,256):
        raw,elapsed=run(z,0x95a01000,0,off,256,4,4,8 if adaptive else 4,16 if adaptive else 4,.02,64)
        all_counts.append(raw['counts']);all_errors.append(raw['errors']);seconds+=elapsed
    counts=np.concatenate(all_counts);errors=np.concatenate(all_errors)
    values=counts.sum(1).astype(int);values[errors!=0]=65
    cdf=np.bincount(values,minlength=66).cumsum()[:65]/len(values)
    reference=np.load(small.with_name('law_0.npz'))['total'].cumsum()
    stop=min(len(reference),65)
    ks=float(max(abs(cdf[:stop]-reference[:stop])))
    radius=float(np.sqrt(np.log(400)/(2*len(values))))
    small_quality.append(dict(adaptive=adaptive,records=len(values),failures=int(np.count_nonzero(errors)),
        seconds=seconds,max_reference_count=stop-1,empirical_ks=ks,lower=max(0.,ks-radius),upper=min(1.,ks+radius)))
    np.savez_compressed(out/f'small_{adaptive}.npz',counts=counts,errors=errors)
save(out,'small_quality.json',small_quality)
manifest(out)
