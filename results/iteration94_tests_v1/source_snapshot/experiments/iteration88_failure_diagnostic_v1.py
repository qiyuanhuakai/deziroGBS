"""Replay every new projected failure; independently evaluate its exported Gram.
No production sampler changes, probability clipping, replacement or new quality
records. A negative probability on a frozen numerical state is distinguished
from cancellation in its final signed-subset arithmetic.
"""
import argparse,subprocess,mpmath as mp
from iteration88_project_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'native/gaussian_click_project_diagnostic88_v1.cpp';dll=out/'diagnostic.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),source,ROOT/'native/gaussian_click_projected88_v2.cpp',inp,ROOT/'research/iteration88_full_quality_plan.json'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},digits=100));gcc=runtime_dirs()[0]/'g++.exe';cmd=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(dll),'-lquadmath'];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,'build.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;z=np.load(inp);v,x=z['v'],z['x'];rows=[];mp.mp.dps=100
for job in read(ROOT/'research/iteration88_full_quality_plan.json')['direct']:
 if job['method']!='projected':continue
 base=ROOT/f"results/iteration88_full_quality_{job['name']}_v1"
 for sector in range(2):
  with NativeStableClickSampler(v,x[sector],dll,branch_budget=job['budget'],runtime_dirs=runtime_dirs()) as model:
   fn=model.lib.tc88_diagnostic;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,DP,DP,IP,C.c_char_p,C.c_int,DP,C.c_char_p,C.c_int]
   for offset in range(0,job['records'],64):
    frozen=np.load(base/f'raw_{sector}_{offset}.npz')
    for local in np.flatnonzero(frozen['errors'][:,0]):
     record=offset+int(local);mode=int(frozen['errors'][local,1]);tag=f"{job['name']}_{sector}_{record}";draws=uniforms(int(job['experiment'],16),sector,record,1,len(v));numeric,nmeta=projected(model,draws,job['budget'],guarded=0,threads=1);guarded,gmeta=projected(model,draws,job['budget'],threads=1);np.savez_compressed(out/f'numeric_{tag}.npz',**numeric);np.savez_compressed(out/f'guarded_{tag}.npz',**guarded);identical=all(np.array_equal(guarded[key][0],frozen[key][local],equal_nan=True) for key in guarded);assert identical
     covariance=np.zeros_like(x[sector]);ports=np.zeros(8,np.int32);text=C.create_string_buffer(65536);stats=np.zeros(8);err=C.create_string_buffer(4096);rc=fn(model.handle,mode,job['budget'],ptr(draws),ptr(covariance),ports.ctypes.data_as(IP),text,len(text),ptr(stats),err,len(err));assert rc==0,err.value;G=json.loads(text.value.decode());save(out,f'gram_{tag}.json',G);np.savez_compressed(out/f'state_{tag}.npz',covariance=covariance,ports=ports,statistics=stats);k=int(stats[0]);A=mp.matrix([[mp.mpf(v) for v in row] for row in G]);terms=[]
     for mask in range(1<<(k+1)):
      ids=[i for j in range(k+1) if mask>>j&1 for i in (2*j,2*j+1)];terms.append(1/mp.sqrt(mp.det(mp.matrix([[A[i,j] for j in ids] for i in ids]))) if ids else mp.mpf(1))
     mass=mp.fsum((-1)**mask.bit_count()*terms[mask] for mask in range(1<<k));zero=mp.fsum((-1)**mask.bit_count()*terms[mask|(1<<k)] for mask in range(1<<k));prob=(mass-zero)/mass;d=len(covariance);n=d//2;omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);quantum_min=float(np.linalg.eigvalsh(covariance+np.eye(d)/2+1j*omega/2).min());row=dict(method=job['name'],sector=sector,record=record,mode=mode,budget=job['budget'],identical_replay=identical,numeric=nmeta,guarded=gmeta,retained=k,native_probability=stats[3],independent100_digit_probability=mp.nstr(prob,85),absolute_native_difference=float(abs(mp.mpf(stats[3])-prob)),base_quantum_uncertainty_min_eigenvalue=quantum_min,scope='100-digit determinant sums use42-digit export of the actual long-double Gram. Base covariance is exported as double; its uncertainty eigenvalue is diagnostic, not an interval proof.');rows.append(row);save(out,f'row_{tag}.json',row);print('failure diagnostic',row,flush=True)
save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
