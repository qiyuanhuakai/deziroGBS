"""Analytic actual-output probes and matched exploratory full trajectories.
No exploratory record is included in future quality samples.
"""
import argparse
from iteration90_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration90_build_v1/optical.dll';files=[Path(__file__),ROOT/'experiments/iteration90_common_v1.py',dll,ROOT/'results/iteration89_latent_physicality_v1/summary.json',ROOT/'results/iteration89_remaining_state_v1/state.npz',ROOT/'results/iteration83_loss_pilot_v1/input.npz'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},experiment=0x90002001,records=32,threads=16));runtime=os.add_dll_directory(str(runtime_dirs()[0]));lib=C.CDLL(str(dll));fn=lib.tc90_optical_probe;fn.argtypes=[C.c_int,C.c_int,DP,DP,DP,C.c_char_p,C.c_int];rows=[]
for r in read(files[3])['rows']:
 squeeze=float(r['squeezing']);eta=float(r['measured_fraction']);noise=float(r['added_classical_variance']);q=-squeeze/(1+squeeze)+noise;p=squeeze/(1-squeeze)+noise;x=np.diag([q/(1+eta*q),p/(1+eta*p)]);v=np.array([[np.sqrt(1-eta)+0j]]);stats=np.zeros(6);err=C.create_string_buffer(4096);rc=fn(1,1,ptr(x),ptr(v.view(float)),ptr(stats),err,len(err));row=dict(squeezing=squeeze,measured=eta,noise=noise,returncode=rc,statistics=stats.tolist());rows.append(row);assert rc==0,err.value
 if noise>0:assert stats[0]==1,row
state=np.load(ROOT/'results/iteration88_failure_diagnostic_v1/state_project16_1_685.npz');v=np.load(files[4])['optical'];x=state['covariance'];stats=np.zeros(6);err=C.create_string_buffer(4096);rc=fn(len(x)//2,len(v),ptr(x),ptr(np.ascontiguousarray(v).view(float)),ptr(stats),err,len(err));assert rc==0 and stats[0]==0,(stats,err.value);save(out,'probes.json',dict(passed=True,analytic_rows=rows,old_failure_statistics=stats.tolist(),limits='Strict interior test can reject pure/low-rank boundary states. No eigenvalue cutoff or state noise is added.'))
z=np.load(files[5]);full=[];cache_meta=[]
for sector in (0,1):
 with NativeStableClickSampler(z['v'],z['x'][sector],dll,branch_budget=128,runtime_dirs=runtime_dirs()) as model:
  cache,destroy,setup=optical_cache(model);cache_meta.append(setup);print('optical cache',sector,setup,flush=True)
  try:
   if sector==1:
    raw,meta=optical_sample(model,cache,uniforms(0x88004002,1,685,1,len(z['v'])),16);np.savez_compressed(out/'old_failure_replay.npz',**raw);save(out,'old_failure_replay.json',meta);assert meta['returncode']==0,meta
   for budget in (16,64):
    draws=uniforms(0x90002001,sector,0,32,len(z['v']));raw,meta=optical_sample(model,cache,draws,budget);np.savez_compressed(out/f'optical_{budget}_{sector}.npz',**raw);save(out,f'optical_{budget}_{sector}.json',meta);full.append(dict(method='optical',sector=sector,budget=budget,failed=int((raw['errors'][:,0]!=0).sum()),mean=float(raw['clicks'].sum(1).mean()),statistics_sum=raw['statistics'].sum(0).tolist(),**meta));print('optical pilot',full[-1]['budget'],sector,meta,flush=True)
  finally:destroy(cache)
 with NativeStableClickSampler(z['v'],z['x'][sector],ROOT/'results/iteration89_pilot_v1/physical.dll',branch_budget=128,runtime_dirs=runtime_dirs()) as model:
  for budget in (16,64):
   raw,meta=physical(model,uniforms(0x90002001,sector,0,32,len(z['v'])),budget);np.savez_compressed(out/f'canonical_{budget}_{sector}.npz',**raw);full.append(dict(method='canonical',sector=sector,budget=budget,failed=int((raw['errors'][:,0]!=0).sum()),mean=float(raw['clicks'].sum(1).mean()),statistics_sum=raw['statistics'].sum(0).tolist(),**meta))
combined=[]
for method in ('optical','canonical'):
 for budget in (16,64):
  parts=[np.load(out/f'{method}_{budget}_{sector}.npz') for sector in (0,1)];valid=(parts[0]['errors'][:,0]==0)&(parts[1]['errors'][:,0]==0);bits=parts[0]['clicks']|parts[1]['clicks'];combined.append(dict(method=method,budget=budget,failed=int((~valid).sum()),mean_successful=float(bits[valid].sum(1).mean()),sampling_seconds=sum(r['seconds'] for r in full if r['method']==method and r['budget']==budget)))
save(out,'summary.json',dict(completed=True,rows=full,combined=combined,cache=cache_meta,scope=__doc__));manifest(out);print(json.dumps(combined,indent=2),flush=True)
