"""Independently reconstruct every packed suffix Gram; compare full replay paths.
No replay record is counted as new quality data. The superseded v1 rotation is
preserved and explicitly tested to demonstrate its adjoint-sign defect.
"""
import argparse
from iteration91_common_v1 import *
from iteration90_common_v1 import optical_cache as old_cache,optical_sample as old_sample
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
new=ROOT/'results/iteration91_build_v3/optical.dll';old=ROOT/'results/iteration90_build_v1/optical.dll';bad=ROOT/'results/iteration91_build_v1/optical.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),ROOT/'experiments/iteration91_common_v1.py',new,old,bad,inp]
save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},experiment=hex(0x85001002),records=64,budget=16))
z=np.load(inp);audits=[]
with NativeStableClickSampler(z['v'],z['x'][0],bad,branch_budget=16,runtime_dirs=runtime_dirs()) as model:
 h,d,setup=optical_cache(model);d(h);assert setup['gram_residual']>1e-3;save(out,'rejected_v1.json',dict(reason='Used G rather than G adjoint for row updates; detected before any quality run',**setup))
for case in ('full','haar_rectangular','rank_deficient_suffix','zero_suffix','identity','permutation'):
 rng=np.random.default_rng(910010);v=z['v'] if case=='full' else np.linalg.qr(rng.normal(size=(7,4))+1j*rng.normal(size=(7,4)))[0]
 if case=='rank_deficient_suffix':v=np.vstack([np.eye(4),np.zeros((3,4))]).astype(complex)
 if case=='identity':v=np.eye(4,dtype=complex)
 if case=='permutation':v=np.eye(5,dtype=complex)[[3,0,4,1,2]]
 if case=='zero_suffix':v=np.vstack([np.eye(3),np.zeros((2,3))]).astype(complex)
 x=z['x'][0] if case=='full' else np.eye(2*v.shape[1])*.2
 with NativeStableClickSampler(v,x,new,branch_budget=16,runtime_dirs=runtime_dirs()) as model:
  h,d,setup=optical_cache(model);fn=model.lib.tc91_optical_export;fn.argtypes=[C.c_void_p,C.c_int,DP,C.c_char_p,C.c_int];errors=[];maps=[];offsets=[0]
  for j in range(len(v)):
   k=min(v.shape[1],len(v)-j);raw=np.zeros((k,v.shape[1],2));msg=C.create_string_buffer(4096);rc=fn(h,j,ptr(raw),msg,len(msg));assert rc==0,msg.value;b=raw[:,:,0]+1j*raw[:,:,1];gram=v[j:].conj().T@v[j:];error=float(np.linalg.norm(b.conj().T@b-gram)/(1+np.linalg.norm(gram)));errors.append(error);maps.append(b.ravel());offsets.append(offsets[-1]+b.size);assert error<2e-14,(case,j,error)
  np.savez_compressed(out/f'maps_{case}.npz',v=v,maps=np.concatenate(maps),offsets=np.array(offsets),errors=np.array(errors));d(h);audits.append(dict(case=case,all_suffixes=len(v),maximum_independent_gram_error=max(errors),setup=setup));print('cache audit',audits[-1],flush=True)
rows=[]
for sector in (0,1):
 draws=uniforms(0x85001002,sector,0,64,len(z['v']));parts=[]
 for name,dll,cache_fn,sample_fn in (('old',old,old_cache,old_sample),('new',new,optical_cache,optical_sample)):
  with NativeStableClickSampler(z['v'],z['x'][sector],dll,branch_budget=16,runtime_dirs=runtime_dirs()) as model:
   h,d,setup=cache_fn(model);raw,meta=sample_fn(model,h,draws,16);d(h);assert meta['returncode']==0,meta;np.savez_compressed(out/f'{name}_{sector}.npz',**raw);parts.append(raw)
 row=dict(sector=sector,equal_clicks=bool(np.array_equal(parts[0]['clicks'],parts[1]['clicks'])),equal_errors=bool(np.array_equal(parts[0]['errors'],parts[1]['errors'])),equal_probabilities=bool(np.array_equal(parts[0]['conditional'],parts[1]['conditional'])),equal_fit_counters=bool(np.array_equal(parts[0]['statistics'][:,12:15],parts[1]['statistics'][:,12:15])),maximum_probability_difference=float(np.max(abs(parts[0]['conditional']-parts[1]['conditional']))));rows.append(row);assert row['equal_clicks'] and row['equal_errors'] and row['equal_fit_counters'];print('replay',row,flush=True)
save(out,'summary.json',dict(passed=True,audits=audits,replays=rows,scope=__doc__));manifest(out)
