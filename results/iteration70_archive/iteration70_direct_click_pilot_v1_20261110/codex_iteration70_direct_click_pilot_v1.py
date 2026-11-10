from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,subprocess,time,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
OUT=ROOT/'.git/iteration70_direct_click_pilot_v1_20261110';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';source=ROOT/'.git/iteration70_direct_click_v1.cpp';sources=[Path(__file__),source,ROOT/'native/gaussian_reset_cpu.cpp']
for p in sources:shutil.copy2(p,OUT/p.name)
command=[str(gcc),'-O3','-march=native','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(source),'-o',str(OUT/'native.dll')]
save('plan.json',dict(command=command,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},samples=1000000,quality_seed_base=708001,threads=16,scope='New direct threshold route, no PNR counts or photon arrival times. Exact Gaussian no-click conditioning, approximate click moment closure. Independent dense physical-space first/second moment checks for every four-mode prefix. Centered full induced law from native branch tree; displaced law assessed empirically. Gaussian-difference identity is prior art: arXiv:1807.01639. No claim that Gaussian moment closure is exact or novel.'))
start=time.perf_counter();build=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(returncode=build.returncode,stdout=build.stdout,stderr=build.stderr,seconds=time.perf_counter()-start));print('direct-click build',build.returncode,flush=True);assert build.returncode==0
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'native.dll'));DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8)
def ptr(a):return np.ascontiguousarray(a,dtype=np.float64).ctypes.data_as(DP)
lib.tc70_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.tc70_create.restype=C.c_void_p;lib.tc70_destroy.argtypes=[C.c_void_p];lib.tc70_probe.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_int,DP,DP,DP,C.c_char_p,C.c_int];lib.tc70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,DP,DP,BP,DP,C.c_char_p,C.c_int];error=C.create_string_buffer(4096)
def sample(handle,count,seed,threads,mean,variance,m):
    output=np.zeros((count,m),np.uint8);stats=np.zeros(5);start=time.perf_counter();rc=lib.tc70_sample(handle,count,seed,threads,ptr(mean),ptr(variance),output.ctypes.data_as(BP),ptr(stats),error,len(error));seconds=time.perf_counter()-start;assert rc==0,error.value;assert stats[4]==count and int(output.sum())==stats[0];return output,stats,seconds
checks=[];rows=[];dependencies={};count=1000000;radius=math.sqrt((16*math.log(2)+math.log(8/.01))/(2*count))
for case in range(4):
    target_path=ROOT/f'results/iteration69_archive/iteration69_quality_pilot_20261110/input_{case}.npz';core_path=ROOT/f'results/iteration69_archive/iteration69_displaced_v6_20261110/input_{case}.npz';dependencies[target_path.relative_to(ROOT).as_posix()]=H(target_path);dependencies[core_path.relative_to(ROOT).as_posix()]=H(core_path);target_data=np.load(target_path);core_data=np.load(core_path);v=np.ascontiguousarray(target_data['v']);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);target=core_data['target'];m,n=v.shape
    for displaced in (False,True):
        label='displaced' if displaced else 'centered';x0=np.ascontiguousarray(core_data['x0'] if displaced else target_data['x0']);variance=np.ascontiguousarray(core_data['noise_variance'] if displaced else np.zeros(2*n));mean=np.zeros(2*n);handle=lib.tc70_create(m,n,v.ctypes.data_as(DP),ptr(x0),error,len(error));assert handle,error.value
        # Dense physical conditional moments, independent of the subspace update.
        fixed_mean=np.random.default_rng(708100+case).normal(size=2*n)*np.sqrt(variance);front=[(x0,fixed_mean,1.,0)];law=np.zeros(16);case_checks=[]
        for mode in range(m):
            new=[]
            for x,w,weight,mask in front:
                sigma=np.eye(2*m)/2+o@x@o.T;mu=o@w;observed=[mode,mode+m];remain=list(range(mode+1,m))+list(range(mode+1+m,2*m));local=sigma[np.ix_(observed,observed)]+np.eye(2)/2;cross=sigma[np.ix_(remain,observed)];inv=np.linalg.inv(local);local_mean=mu[observed];p0=float(np.exp(-.5*local_mean@inv@local_mean)/np.sqrt(np.linalg.det(local)));p1=1-p0;before=sigma[np.ix_(remain,remain)];before_mean=mu[remain];vac_mean=before_mean-cross@inv@local_mean;vac_cov=before-cross@inv@cross.T
                for clicked in (0,1):
                    xx=np.zeros_like(x);ww=np.zeros_like(w);prob=np.zeros(2);rc=lib.tc70_probe(handle,ptr(x),ptr(w),mode,clicked,ptr(xx),ptr(ww),ptr(prob),error,len(error));assert rc==0,error.value
                    if clicked:
                        expected_mean=(before_mean-p0*vac_mean)/p1;delta=before_mean-vac_mean;expected_cov=(before-p0*vac_cov)/p1-p0/(p1*p1)*np.outer(delta,delta)
                    else:expected_mean=vac_mean;expected_cov=vac_cov
                    actual_mean=(o@ww)[remain];actual_cov=(np.eye(2*m)/2+o@xx@o.T)[np.ix_(remain,remain)];row=dict(case=case,label=label,mode=mode,prefix=mask,clicked=clicked,probability_error=float(max(abs(prob-np.array([p0,p1])))),mean_error=float(np.max(abs(actual_mean-expected_mean),initial=0)),covariance_error=float(np.max(abs(actual_cov-expected_cov),initial=0)));row['passed']=max(row[k] for k in ('probability_error','mean_error','covariance_error'))<2e-10;case_checks.append(row);assert row['passed'],row;next_weight=weight*prob[clicked];next_mask=mask|(clicked<<mode)
                    if mode==m-1:law[next_mask]=next_weight
                    else:new.append((xx,ww,next_weight,next_mask))
            front=new
        checks.extend(case_checks);save(f'probes_{case}_{label}.json',case_checks);np.savez_compressed(OUT/f'fixed_mean_law_{case}_{label}.npz',mean=fixed_mean,distribution=law);assert abs(law.sum()-1)<1e-12
        a,_,_=sample(handle,1024,708001+case,1,mean,variance,m);b,_,_=sample(handle,1024,708001+case,16,mean,variance,m);assert np.array_equal(a,b)
        output,stats,seconds=sample(handle,count,708001+case,16,mean,variance,m);hist=np.bincount(output@np.array([1,2,4,8]),minlength=16);tv=float(.5*abs(hist/count-target).sum());row=dict(case=case,label=label,samples=count,seconds=seconds,observed_threshold_tv=tv,nominal_threshold_tv_interval=[max(0,tv-radius),min(1,tv+radius)],mean_clicks=float(output.sum(axis=1).mean()),seed=708001+case);np.savez_compressed(OUT/f'counts_{case}_{label}.npz',clicks=output,stats=stats,histogram=hist,target=target)
        if not displaced:
            row['induced_threshold_tv']=float(.5*abs(law-target).sum());row['empirical_tv_to_induced']=float(.5*abs(hist/count-law).sum());assert row['empirical_tv_to_induced']<radius
        rows.append(row);save(f'case_{case}_{label}.json',row);print(row,flush=True);lib.tc70_destroy(handle)
# Analytic coherent fixture: independent Bernoulli clicks.
v=np.eye(2,dtype=np.complex128);x=np.zeros((4,4));mean=np.array([1.,.7,-.2,.4]);variance=np.zeros(4);handle=lib.tc70_create(2,2,v.ctypes.data_as(DP),ptr(x),error,len(error));assert handle,error.value;output,stats,seconds=sample(handle,100000,708500,8,mean,variance,2);p=-np.expm1(-(mean[:2]**2+mean[2:]**2)/2);assert np.all(abs(output.mean(axis=0)-p)<7*np.sqrt(p*(1-p)/len(output)));vac,_,_=sample(handle,128,708501,4,np.zeros(4),variance,2);assert not vac.any();lib.tc70_destroy(handle);save('analytic.json',dict(passed=True,observed=output.mean(axis=0).tolist(),target=p.tolist()))
save('summary.json',dict(passed=True,rows=rows,checks=checks,nominal_joint_99pct_tv_radius=radius,dependencies=dependencies,dll_sha256=H(OUT/'native.dll')))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
