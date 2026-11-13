"""Fresh fixed512 full1152-output B64 shared-base direct-click records.
New disjoint experiment/sector/record/purpose domains; retain every failure as
an extra outcome. Separate nine-event nominal joint99 quality family; combining
with the PNR99 family gives at least98%, unless bounds are reallocated explicitly.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,argparse,json,hashlib,ctypes as C,shutil,time,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
from iteration86_joint_quality_v1 import witness
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);U64P=C.POINTER(C.c_uint64)
def ptr(x):return x.ctypes.data_as(DP)
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 gates=ROOT/'results/iteration86_shared_gates_v4';replay=ROOT/'results/iteration86_shared_replay_v2';assert read(gates/'summary.json')['passed'] and read(replay/'summary.json')['passed'];dll=gates/'shared.dll';rngdll=ROOT/'results/iteration84_rng_gates_v1/rng.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';prior=ROOT/'results/iteration85_joint_quality_v1';realpath=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz';profile=ROOT/'results/iteration83_bias_profile_v1/summary.json';files=[Path(__file__),ROOT/'experiments/iteration86_joint_quality_v1.py',dll,rngdll,inp,gates/'summary.json',replay/'summary.json',profile,realpath,*[prior/f'reference_{j}.npz' for j in range(3)]];N=512;pilot=128;experiment=0x86004001;delta=.01/9;groups=[list(range(8)),list(range(1144,1152)),list(range(0,1152,144))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},experiment=experiment,records=N,global_offset=0,sectors=[0,1],purpose=5,budget=64,batch_size=32,threads=16,groups=groups,regions=[[0,1152],[1024,1152]],confidence=dict(events=9,delta_per_event=delta,pilot_records=pilot)))
 dirs=[Path(shutil.which('g++')).parent];runtime=os.add_dll_directory(str(dirs[0]));lib=C.CDLL(str(rngdll));rng=lib.rng84_stream;rng.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,U64P,C.c_char_p,C.c_int];rng.restype=C.c_int;d=np.load(inp);v,x=d['v'],d['x'];m=len(v);parts=[];valids=[];rows=[]
 for sector in range(2):
  output=[];completed=[]
  with NativeStableClickSampler(v,x[sector],dll,branch_budget=64,runtime_dirs=dirs) as model:
   fn=model.lib.tc86_shared_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.c_int,DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.restype=C.c_int
   for offset in range(0,N,32):
    count=min(32,N-offset);words=np.zeros((count,m),np.uint64);err=C.create_string_buffer(4096);start=time.perf_counter()
    for r in range(count):assert rng(experiment,sector,5,offset+r,0,m,words[r].ctypes.data_as(U64P),err,len(err))==0,err.value
    draws=np.ascontiguousarray((words>>11).astype(float)*2**-53);bits=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));errors=np.zeros((count,2),np.int32);stats=np.zeros((count,6));rc=fn(model.handle,count,16,64,ptr(draws),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err));seconds=time.perf_counter()-start;valid=(errors[:,0]==0)&(stats[:,5]==m);row=dict(sector=sector,offset=offset,records=count,returncode=rc,error=err.value.decode(),failed=int((~valid).sum()),seconds=seconds);rows.append(row);save(f'row_{sector}_{offset}.json',row);np.savez_compressed(out/f'raw_{sector}_{offset}.npz',clicks=bits,conditional=prob,errors=errors,statistics=stats);output.append(bits);completed.append(valid);print('shared64',sector,offset,'failed',row['failed'],'seconds',seconds,flush=True)
  parts.append(np.concatenate(output));valids.append(np.concatenate(completed))
 bits=np.bitwise_or(*parts);valid=valids[0]&valids[1];np.savez_compressed(out/'combined.npz',clicks=bits,valid=valid);quality=dict(records=N,failed_records=int((~valid).sum()),mean_clicks_successful=float(bits[valid].sum(1).mean()) if valid.any() else None,local_click=[],click_regions=[])
 for group,ids in enumerate(groups):
  target=np.r_[np.load(prior/f'reference_{group}.npz')['target'],0.];labels=(bits[:,ids].astype(np.uint64)@np.array([1<<j for j in range(8)],np.uint64)).astype(int);labels[~valid]=256;hist=np.bincount(labels,minlength=257);ph=np.bincount(labels[:pilot],minlength=257);vh=np.bincount(labels[pilot:],minlength=257);event=ph/pilot>target;tv=float(abs(hist/N-target).sum()/2);gap=float(vh[event].sum()/(N-pilot)-target[event].sum());quality['local_click'].append(dict(group=group,empirical_tv=tv,nominal_joint99_marginal_tv_lower=max(0.,gap-math.sqrt(math.log(1/delta)/(2*(N-pilot)))),nominal_joint99_marginal_tv_upper=min(1.,tv+math.sqrt(math.log(1/delta)/(2*N)))));np.savez_compressed(out/f'hist_{group}.npz',histogram=hist,pilot_histogram=ph,validation_histogram=vh,event=event)
 moments=read(profile)['reference']
 for first,end in [(0,1152),(1024,1152)]:
  ref=next(r for r in moments if (r['first'],r['end'])==(first,end));values=bits[:,first:end].sum(1).astype(int);values[~valid]=end-first+1;quality['click_regions'].append(dict(first=first,end=end,**witness(values,ref['mean'],ref['variance'],end-first+1,delta)))
 realhist=np.load(realpath)['total_click_histogram'];realN=int(realhist.sum());values=bits.sum(1).astype(int);values[~valid]=1153;hist=np.bincount(values,minlength=1154);distance=float(abs(np.cumsum(hist)/N-np.r_[np.cumsum(realhist)/realN,1.]).max());radius=math.sqrt(math.log(2/delta)/(2*N))+math.sqrt(math.log(2/delta)/(2*realN));quality['experimental_click_count']=dict(empirical_cdf_distance=distance,nominal_joint99_cdf_lower=max(0.,distance-radius),nominal_joint99_cdf_upper=min(1.,distance+radius),experimental_records=realN)
 save('summary.json',dict(completed=True,rows=rows,quality=quality,unique_domains=N*2,sampling_seconds=sum(r['seconds'] for r in rows),confidence=read(out/'plan.json')['confidence'],scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}));print('QUALITY',json.dumps(quality),flush=True)

if __name__=='__main__':main()
