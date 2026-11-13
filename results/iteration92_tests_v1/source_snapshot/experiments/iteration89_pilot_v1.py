"""Build both C++ candidates; physical probes and matched exploratory paths.
Paired diagnostic records are never counted as fresh independent quality data.
"""
import argparse,subprocess,mpmath as mp
from iteration89_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);sources=[ROOT/'native/gaussian_click_physical89_v1.cpp',ROOT/'native/gaussian_pnr_puretree89_v1.cpp'];files=[Path(__file__),ROOT/'experiments/iteration89_common_v1.py',*sources,ROOT/'results/iteration83_loss_pilot_v1/input.npz',ROOT/'results/iteration88_project_gates_v1/summary.json'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},threads=16,full_records=32,click_experiment=0x89002001,pnr_experiment=0x89002002));gcc=runtime_dirs()[0]/'g++.exe'
for name,source in zip(('physical','puretree'),sources):
 cmd=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(source),'-o',str(out/f'{name}.dll'),'-lquadmath'];r=subprocess.run(cmd,capture_output=True,text=True,errors='replace');save(out,f'build_{name}.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('built',name,flush=True)
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(out/'physical.dll'));fn=lib.tc89_physical_probe;fn.argtypes=[C.c_int,DP,DP,C.c_char_p,C.c_int];probes=[]
cases=[('thermal',np.eye(4)*.1,True),('vacuum_boundary',np.zeros((4,4)),False),('negative',np.diag([-.2,.1,.1,.1]),False),('frozen_failure',np.load(ROOT/'results/iteration88_failure_diagnostic_v1/state_project16_1_685.npz')['covariance'],False)]
for name,x,expected in cases:
 x=np.ascontiguousarray(x);stats=np.zeros(4);err=C.create_string_buffer(4096);rc=fn(len(x),ptr(x),ptr(stats),err,len(err));n=len(x)//2;omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);eigen=float(np.linalg.eigvalsh(x+np.eye(len(x))/2+1j*omega/2).min());row=dict(case=name,returncode=rc,accepted=bool(stats[0]),shift=stats[1],residual=stats[2],independent_min_eigenvalue=eigen);probes.append(row);assert rc==0 and bool(stats[0])==expected,row
inv=lib.tc89_project_probe;inv.argtypes=[C.c_int,DP,DP,DP,DP,C.c_char_p,C.c_int]
for r in read(ROOT/'results/iteration88_project_pilot_v2/projection.json')['rows']:
 z=np.load(ROOT/f"results/iteration88_project_pilot_v2/projection_{r['dimension']}_{r['scale']}.npz");target=z['target'];F=z['forms'];base=np.zeros_like(target);stats=np.zeros(2);err=C.create_string_buffer(4096);rc=inv(len(target),ptr(target),ptr(F),ptr(base),ptr(stats),err,len(err));row=dict(dimension=len(target),scale=r['scale'],returncode=rc,reason=stats[0],residual=stats[1],base_error=float(abs(base-z['base']).max()));probes.append(row);assert rc==0 and stats[0]==0 and row['base_error']<2e-12,row
save(out,'physical_probes.json',dict(passed=True,rows=probes));print('physical probes passed',flush=True)
inp=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');v,x=inp['v'],inp['x'];rows=[]
with NativeStableClickSampler(v,x[1],out/'physical.dll',branch_budget=16,runtime_dirs=runtime_dirs()) as model:
 draws=uniforms(0x88004002,1,685,1,len(v));raw,meta=physical(model,draws,16);np.savez_compressed(out/'old_failure_replay.npz',**raw);save(out,'old_failure_replay.json',meta);assert meta['returncode']==0,meta;print('old failing address now completes',flush=True)
for budget in (16,64):
 for method in ('block','unchecked','physical'):
  dll={'block':ROOT/'results/iteration87_guarded_gates_v2/guarded.dll','unchecked':ROOT/'results/iteration88_project_pilot_v2/projected.dll','physical':out/'physical.dll'}[method];parts=[];good=[];metas=[];total=None
  for sector in range(2):
   draws=uniforms(0x89002001,sector,0,32,len(v))
   with NativeStableClickSampler(v,x[sector],dll,branch_budget=budget,runtime_dirs=runtime_dirs()) as model:raw,meta=(sample if method=='block' else projected if method=='unchecked' else physical)(model,draws,budget)
   np.savez_compressed(out/f'click_{method}_{budget}_{sector}.npz',**raw);save(out,f'click_{method}_{budget}_{sector}.json',meta);parts.append(raw['clicks']);good.append((raw['errors'][:,0]==0)&(raw['statistics'][:,5]==len(v)));metas.append(meta);total=raw['statistics'].sum(0) if total is None else total+raw['statistics'].sum(0)
  bits=np.bitwise_or(*parts);valid=good[0]&good[1];row=dict(kind='click',method=method,budget=budget,failed=int((~valid).sum()),mean_successful=float(bits[valid].sum(1).mean()),seconds=sum(m['seconds'] for m in metas),statistics_sum=total.tolist());rows.append(row);save(out,f'click_{method}_{budget}.json',row);print('click pilot',row,flush=True)
for case in range(5):
 sectors=range(2) if case==4 else range(1)
 for R in (3,5):
  for sector in sectors:
   if case==4:
    z=np.load(ROOT/f'results/iteration84_correlated_pnr_pilot_v1/input_{sector}.npz');t=np.ascontiguousarray(z['t']);q=dict(v=z['v'],t=t,L=np.ascontiguousarray(z['noise_factor']),x=np.diag(np.r_[-t/(1+t),t/(1-t)]),mean=np.zeros(2*len(t)))
   else:q={k:np.ascontiguousarray(a) for k,a in np.load(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{case}.npz').items() if k in ('v','t','x','L','mean')}
   outputs=[];metas=[]
   with NativeDirectClickSampler(q['v'],q['x'],out/'puretree.dll',runtime_dirs=runtime_dirs()) as model:
    handle,destroy=tree(model)
    try:
     for method in ('old','new'):
      raw,meta=(pnr if method=='old' else puretree)(model,handle,q,R,32 if case==4 else 256,0x89002002+case,sector=sector);np.savez_compressed(out/f'pnr_{case}_{R}_{sector}_{method}.npz',**raw);outputs.append(raw);metas.append(meta)
    finally:destroy(handle)
   equal=all(np.array_equal(outputs[0][key],outputs[1][key]) for key in ('counts','errors','valid')) and np.array_equal(outputs[0]['statistics'][:,:16],outputs[1]['statistics'][:,:16]);row=dict(kind='pnr',case=case,retained=R,sector=sector,identical_original_output=equal,old=metas[0],new=metas[1]);rows.append(row);save(out,f'pnr_{case}_{R}_{sector}.json',row);print('PNR pilot',row,flush=True);assert equal,row
save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__));manifest(out)
