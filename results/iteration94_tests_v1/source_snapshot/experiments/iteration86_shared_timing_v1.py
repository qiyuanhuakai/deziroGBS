"""Three serial four/64-record paired timings of frozen84 B64 versus shared86 B64.
Use frozen85 experiment0x85001002 records0..63 in both spectra, with four-record prefixes and64-record batches. The
first64 frozen rows have no old failures; these are not unbiased failure-rate trials. These are repeated correctness/timing
records, not new quality samples. Include model construction, explicit native
uniforms, sampling, OR and destruction; exclude loss preparation, DLL load and IO.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,argparse,json,hashlib,ctypes as C,shutil,time,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_stable_click import NativeStableClickSampler
DP=C.POINTER(C.c_double);BP=C.POINTER(C.c_uint8);IP=C.POINTER(C.c_int);U64P=C.POINTER(C.c_uint64)
def ptr(x):return x.ctypes.data_as(DP)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 paths=dict(old=ROOT/'results/iteration84_exactsum_pilot_v1/exactsum.dll',shared=ROOT/'results/iteration86_shared_gates_v4/shared.dll');rngdll=ROOT/'results/iteration84_rng_gates_v1/rng.dll';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),inp,rngdll,*paths.values()];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files},records=[4,64],experiment=0x85001002,global_offset=0,purpose=5,sectors=[0,1],budget=64,threads=16,repetitions=3))
 dirs=[Path(shutil.which('g++')).parent];runtime=os.add_dll_directory(str(dirs[0]));loaded=[C.CDLL(str(p)) for p in paths.values()];lib=C.CDLL(str(rngdll));rng=lib.rng84_stream;rng.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,U64P,C.c_char_p,C.c_int];d=np.load(inp);v,x=d['v'],d['x'];m=len(v);rows=[];results={}
 for rep in range(3):
  for count in (4,64):
   for method in (['old','shared'] if rep%2==0 else ['shared','old']):
    parts=[];raw=[];sectors=[];start=time.perf_counter()
    for sector in range(2):
     words=np.zeros((count,m),np.uint64);err=C.create_string_buffer(4096)
     for record in range(count):assert rng(0x85001002,sector,5,record,0,m,words[record].ctypes.data_as(U64P),err,len(err))==0,err.value
     draws=np.ascontiguousarray((words>>11).astype(float)*2**-53);bits=np.zeros((count,m),np.uint8);prob=np.zeros((count,m));errors=np.zeros((count,2),np.int32);stats=np.zeros((count,8 if method=='old' else 6))
     with NativeStableClickSampler(v,x[sector],paths[method],branch_budget=64,runtime_dirs=dirs) as model:
      fn=getattr(model.lib,'tc84e_sample' if method=='old' else 'tc86_shared_sample');tail=[DP,BP,DP,IP,DP,C.c_char_p,C.c_int];fn.argtypes=[C.c_void_p,C.c_int,*([C.c_uint64] if method=='old' else []),C.c_int,C.c_int,*tail];args=[model.handle,count,*([0] if method=='old' else []),16,64,ptr(draws),bits.ctypes.data_as(BP),ptr(prob),errors.ctypes.data_as(IP),ptr(stats),err,len(err)];rc=fn(*args)
     success=rc==0 and not errors[:,0].any() and bool(np.all(stats[:,5]==m));sectors.append(dict(sector=sector,returncode=rc,error=err.value.decode(),success=success));parts.append(bits);raw.append(dict(clicks=bits,conditional=prob,errors=errors,statistics=stats))
    combined=np.bitwise_or(*parts);seconds=time.perf_counter()-start;row=dict(rep=rep,method=method,records=count,seconds=seconds,success=all(s['success'] for s in sectors),sectors=sectors);rows.append(row);save(f'row_{rep}_{method}_{count}.json',row)
    for sector,z in enumerate(raw):np.savez_compressed(out/f'raw_{rep}_{method}_{count}_{sector}.npz',**z)
    np.savez_compressed(out/f'combined_{rep}_{method}_{count}.npz',clicks=combined);results[rep,method,count]=raw;print(rep,method,count,seconds,row['success'],flush=True)
 repeats=[];cross=[];medians=[]
 for count in (4,64):
  for method in paths:
   equal=all(np.array_equal(results[0,method,count][s][field],results[rep,method,count][s][field],equal_nan=True) for rep in [1,2] for s in range(2) for field in ['clicks','conditional','errors','statistics']);repeats.append(dict(method=method,records=count,identical_repetitions=equal))
   medians.append(dict(method=method,records=count,median_seconds=statistics.median(r['seconds'] for r in rows if r['method']==method and r['records']==count)))
  for sector in range(2):
   old,new=results[0,'old',count][sector],results[0,'shared',count][sector];cross.append(dict(records=count,sector=sector,bits_equal=bool(np.array_equal(old['clicks'],new['clicks'])),maximum_conditional_difference=float(np.nanmax(abs(old['conditional']-new['conditional'])))))
 save('summary.json',dict(passed=all(r['success'] for r in rows),rows=rows,medians=medians,repetitions=repeats,cross=cross,scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}))


if __name__=='__main__':main()
