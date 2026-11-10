from pathlib import Path
import os,ctypes as C,hashlib,json,shutil,time,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
import numpy as np
OUT=ROOT/'.git/iteration70_noise_screen_v1_20261110';OUT.mkdir(exist_ok=False)
TARGET=ROOT/'.git/iteration70_pgf_quad_v1_20261110';BUILD=ROOT/'.git/iteration70_flow_pilot_v1_20261110';assert json.loads((TARGET/'summary.json').read_text())['passed'];assert json.loads((BUILD/'summary.json').read_text())['passed']
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def save(name,obj):
    p=OUT/name;assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
count=100000;theta_values=[0.,.1,.25,.5,.75,1.];runs=5*(1+len(theta_values));delta=.01/(2*runs);upper_radius=math.sqrt(math.log(1/delta)/(2*count));lower_radius=math.sqrt(math.log(1/delta)/(2*(count//2)))
sources=[Path(__file__),ROOT/'.git/iteration70_flow_v1.cpp',ROOT/'native/gaussian_reset_cpu.cpp'];dependencies={p.relative_to(ROOT).as_posix():H(p) for p in [TARGET/'summary.json',BUILD/'native.dll',BUILD/'summary.json']}
for p in sources:shutil.copy2(p,OUT/p.name)
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},dependencies=dependencies,theta_values=theta_values,samples=count,seed_base=702001,selector=3,algorithm=1,nominal_joint_failure_probability=.01,upper_radius=upper_radius,heldout_lower_radius=lower_radius,scope='Screen exact Gaussian input decompositions with two-quadrature classical noise. theta=0 is minimum pure-core squeezing; theta=1 is maximum core squeezing. Same Gaussian target for each theta in exact arithmetic. Joint PNR finite simplex plus full remaining tail; no photon cutoff in the sampler. Dimension-free McDiarmid upper bound plus target tail; lower bound uses an event selected on the first half and tested on the second half. Nominal bounds assume ideal IID draws and exact numerical targets, not formal finite-PRNG/interval certificates. Parameter selection requires a separate holdout run.'))
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(BUILD/'native.dll'));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)
def ptr(a):return a.ctypes.data_as(DP)
lib.gr_create.argtypes=[C.c_int,C.c_int,DP,DP,C.c_char_p,C.c_int];lib.gr_create.restype=C.c_void_p;lib.gr_destroy.argtypes=[C.c_void_p];lib.g70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]
target_summary=json.loads((TARGET/'summary.json').read_text());rows=[];error=C.create_string_buffer(4096)
for case,description in enumerate(target_summary['rows']):
    path=TARGET/f'target_{case}.npz';dependencies[path.relative_to(ROOT).as_posix()]=H(path);data=np.load(path);v=np.ascontiguousarray(data['v']);cov=data['covariance'];r=float(data['r']);eta=float(data['eta']);cutoff=int(data['cutoff']);codes=data['codes'];target=data['probabilities'];tail=float(description['analytic_target_tail']);law=np.r_[target,tail];assert abs(law.sum()-1)<1e-12
    m,n=v.shape;base=cutoff+1;lookup=np.full(base**m,-1,np.int32);lookup[codes]=np.arange(len(codes));powers=np.array([base**j for j in range(m)],np.uint64);o=np.block([[v.real,-v.imag],[v.imag,v.real]]);mean=np.zeros(2*n);vp=.5*(eta*np.exp(2*r)+1-eta);vm=.5*(eta*np.exp(-2*r)+1-eta);lower=-.5*np.log(2*vm);upper=.5*np.log(2*vp)
    for theta in [None]+theta_values:
        if theta is None:
            label='centered';number=np.sinh(r)**2;pair=np.sinh(r)*np.cosh(r);x0=np.diag([eta*(number+pair)]*n+[eta*(number-pair)]*n);variance=np.zeros(2*n)
        else:
            label=f'theta_{round(100*theta):03d}';core=lower+theta*(upper-lower);x0=np.diag([.5*np.expm1(2*core)]*n+[.5*np.expm1(-2*core)]*n);variance=np.r_[np.full(n,vp*(-np.expm1(-2*(1-theta)*(upper-lower)))),np.full(n,vm*(-np.expm1(-2*theta*(upper-lower))))]
        x0=np.ascontiguousarray(x0);variance=np.ascontiguousarray(variance);reconstruction=np.eye(2*m)+2*o@(x0+np.diag(variance))@o.T;reconstruction_error=float(np.max(abs(reconstruction-cov)));assert reconstruction_error<2e-13 and min(variance)>=0
        handle=lib.gr_create(m,n,ptr(v),ptr(x0),error,len(error));assert handle,error.value;counts=np.zeros((count,m),np.uint32);stats=np.zeros(18);start=time.perf_counter();rc=lib.g70_sample(handle,count,702001+case*100,3,1,ptr(mean),ptr(variance),counts.ctypes.data_as(UP),ptr(stats),error,len(error));seconds=time.perf_counter()-start;lib.gr_destroy(handle)
        row=dict(case=case,r=r,eta=eta,theta=theta,label=label,returncode=rc,error=error.value.decode(),seconds=seconds,reconstruction_error=reconstruction_error);np.savez_compressed(OUT/f'case_{case}_{label}.npz',counts=counts,stats=stats,x0=x0,variance=variance)
        if rc==0:
            totals=counts.sum(axis=1);inside=totals<=cutoff;labels=np.full(count,len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];assert min(labels)>=0;hist=np.bincount(labels,minlength=len(law));empirical=hist/count;tv=float(.5*np.sum(abs(empirical-law)));training=np.bincount(labels[:count//2],minlength=len(law))/(count//2);event=training>law;event_target=float(law[event].sum());event_heldout=float(np.mean(event[labels[count//2:]]));lower_tv=max(0.,event_heldout-event_target-lower_radius);upper_tv=min(1.,tv+upper_radius+tail)
            row.update(empirical_joint_binned_tv=tv,nominal_full_pnr_tv_interval=[lower_tv,upper_tv],heldout_event_target=event_target,heldout_event_observed=event_heldout,mean_photons=float(totals.mean()),target_mean_photons=float(n*eta*np.sinh(r)**2),maximum_photons=int(totals.max()),tail_events=int(sum(~inside)),events_ge128=int(sum(totals>=128)),flow_evaluations_per_record=float(stats[11]/count));np.savez_compressed(OUT/f'histogram_{case}_{label}.npz',histogram=hist,target=law,heldout_event=event)
        rows.append(row);save(f'case_{case}_{label}.json',row);print(row,flush=True);assert rc==0,error.value
save('summary.json',dict(passed=all(row['returncode']==0 for row in rows),rows=rows,dependencies=dependencies,selection_scope='Screening only; a fresh validation cohort is required before treating selected theta as improved.')));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
