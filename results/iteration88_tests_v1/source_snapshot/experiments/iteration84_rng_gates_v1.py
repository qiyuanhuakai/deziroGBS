"""Domain-separation gates before regenerated statistical experiments.
Pinned upstream Philox KATs, actual native address uniqueness, boundary rejection,
and complete PNR replay across worker counts and chunking. This verifies input
domain separation, not true iid output or absence of ordinary random value collisions.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,shutil,subprocess,ctypes as C
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_direct_click import NativeDirectClickSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));ref=ROOT/'results/iteration84_rng_reference_v1';sources=[ROOT/'native'/name for name in ('gaussian_rng_probe84_v1.cpp','gaussian_pnr_domain84_v1.cpp','gaussian_pnr_correlated84_v2.cpp')];files=[Path(__file__),*[p for p in (ROOT/'native').iterdir() if p.suffix in ('.cpp','.cu','.hpp','.h','.txt')],*[p for p in ref.iterdir() if p.is_file()],ROOT/'src/gbslab/native_direct_click.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},experiments=[0x84001001,0x84001002],sectors=[0,1],records=4096,purposes=[1,2,3,4],draws=[0,1,2,3]))
for name,src in zip(('rng','mixed','correlated'),sources):
 command=[str(gcc),'-O3','-std=c++20','-shared','-DEIGEN_DONT_PARALLELIZE','-ffp-contract=off','-fno-fast-math','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(src),'-o',str(OUT/f'{name}.dll')];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save(f'build_{name}.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr;print('built',name,flush=True)
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(OUT/'rng.dll'));U32=C.POINTER(C.c_uint32);U64=C.POINTER(C.c_uint64);DP=C.POINTER(C.c_double);IP=C.POINTER(C.c_int);ptr=lambda x:x.ctypes.data_as(DP);block=lib.rng84_block;block.argtypes=[U32,U32,U32];block.restype=C.c_int;stream=lib.rng84_stream;stream.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,C.c_int,U64,C.c_char_p,C.c_int];stream.restype=C.c_int;address=lib.rng84_address;address.argtypes=[C.c_uint64,C.c_uint,C.c_uint,C.c_uint64,C.c_uint64,U32,IP,C.c_char_p,C.c_int];address.restype=C.c_int
kat=[]
for path in (ref/'kat_vectors',ref/'old_kat_vectors'):
 for line in path.read_text().splitlines():
  words=line.split()
  if words[:2]!=['philox4x32','10']:continue
  values=[int(w,16) for w in words[2:]];assert len(values)==10;ctr=np.array(values[:4],np.uint32);key=np.array(values[4:6],np.uint32);out=np.zeros(4,np.uint32);assert block(ctr.ctypes.data_as(U32),key.ctypes.data_as(U32),out.ctypes.data_as(U32))==0;assert out.tolist()==values[6:];kat.append(dict(source=path.name,counter=values[:4],key=values[4:6],expected=values[6:],actual=out.tolist()))
save('known_answer_checks.json',dict(passed=True,rows=kat));assert len(kat)>=3;print('known answers',len(kat),flush=True)
seen=set();digest=hashlib.sha256();fields=np.zeros(6,np.uint32);lane=np.zeros(1,np.int32);err=C.create_string_buffer(4096)
for experiment in (0x84001001,0x84001002):
 for sector in range(2):
  for row in range(4096):
   for purpose in range(1,5):
    for draw in range(4):
     assert address(experiment,sector,purpose,row,draw,fields.ctypes.data_as(U32),lane.ctypes.data_as(IP),err,len(err))==0;actual=(*map(int,fields),int(lane[0]));expected=(experiment,0,draw//2,0,row,(sector<<16)|purpose,draw%2);assert actual==expected and actual not in seen;seen.add(actual);digest.update(np.array(actual,dtype='<u4').tobytes())
save('domain_checks.json',dict(passed=True,addresses=len(seen),duplicates=0,ordered_addresses_sha256=digest.hexdigest(),proof='Key identifies the 64-bit experiment. Counter fields encode disjoint 64-bit block, 32-bit global record, 16-bit sector and 16-bit purpose; lane distinguishes the two 64-bit words. Each coordinate is recoverable.'))
boundary=[]
for sector,purpose,row,draw in [(65535,65535,2**32-1,2**64-2),(0,1,0,2**33+1)]:
 assert address(2**64-1,sector,purpose,row,draw,fields.ctypes.data_as(U32),lane.ctypes.data_as(IP),err,len(err))==0;assert fields.tolist()==[2**32-1,2**32-1,(draw//2)&(2**32-1),(draw//2)>>32,row,(sector<<16)|purpose];boundary.append(dict(sector=sector,purpose=purpose,row=row,draw=str(draw),valid=True))
for sector,purpose,row in [(65536,1,0),(0,65536,0),(0,1,2**32)]:
 assert address(1,sector,purpose,row,0,fields.ctypes.data_as(U32),lane.ctypes.data_as(IP),err,len(err))!=0;boundary.append(dict(sector=sector,purpose=purpose,row=row,error=err.value.decode()))
out=np.zeros(8,np.uint64);assert stream(1,0,1,0,2**64-1,1,out.ctypes.data_as(U64),err,len(err))!=0;boundary.append(dict(draw='uint64 max',error=err.value.decode()));save('boundary_checks.json',boundary)
# Real stream words reproduce their native block addresses and odd starts.
allwords=np.zeros(32,np.uint64);assert stream(0x84001001,1,3,77,0,32,allwords.ctypes.data_as(U64),err,len(err))==0
for start in (0,1,7,16):
 out=np.zeros(8,np.uint64);assert stream(0x84001001,1,3,77,start,8,out.ctypes.data_as(U64),err,len(err))==0;assert np.array_equal(out,allwords[start:start+8])
np.savez_compressed(OUT/'stream_words.npz',words=allwords)
def invoke(kind,count,offset,workers):
 v=np.eye(2,dtype=complex);t=np.array([.2,.3]);x=np.eye(4)*.3 if kind=='mixed' else np.diag(np.r_[-t/(1+t),t/(1-t)]);mean=np.array([.4,-.1,.2,.3]);variance=np.array([.1,.2,.3,.1]);L=np.array([[.3,0,0,0],[.1,.4,0,0],[0,.1,.3,0],[.1,0,.2,.3]]);counts=np.zeros((count,2),np.uint32);stats=np.zeros((count,16));errors=np.zeros((count,2),np.int32);err=C.create_string_buffer(4096)
 with NativeDirectClickSampler(v,x,OUT/f'{kind}.dll',runtime_dirs=[gcc.parent]) as model:
  if kind=='mixed':
   fn=model.lib.pnr84_domain_sample;fn.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,DP,DP,U32,DP,IP,C.c_char_p,C.c_int];fn.restype=C.c_int;rc=fn(model.handle,count,0x84000001,7,offset,workers,3,ptr(mean),ptr(variance),counts.ctypes.data_as(U32),ptr(stats),errors.ctypes.data_as(IP),err,len(err))
  else:
   fn=model.lib.pnr84d_correlated_sample;fn.argtypes=[C.c_void_p,DP,DP,C.c_int,C.c_uint64,C.c_uint,C.c_uint64,C.c_int,C.c_int,C.c_int,DP,DP,U32,DP,IP,C.c_char_p,C.c_int];fn.restype=C.c_int;rc=fn(model.handle,ptr(t),ptr(L),count,0x84000002,7,offset,workers,3,1,ptr(mean),None,counts.ctypes.data_as(U32),ptr(stats),errors.ctypes.data_as(IP),err,len(err))
 assert rc==0 and not errors[:,0].any() and np.all(stats[:,1]==1),err.value;assert np.array_equal(counts.sum(1),stats[:,0]) and np.array_equal(stats[:,12],stats[:,0]+1) and np.array_equal(stats[:,13],stats[:,0]);return counts,stats,errors
replays=[]
for kind in ('mixed','correlated'):
 a=invoke(kind,97,0,1);b=invoke(kind,97,0,8);first=invoke(kind,31,0,4);second=invoke(kind,66,31,4)
 for i in range(3):assert np.array_equal(a[i],b[i]) and np.array_equal(a[i],np.concatenate([first[i],second[i]]))
 np.savez_compressed(OUT/f'replay_{kind}.npz',counts=a[0],statistics=a[1],errors=a[2]);replays.append(dict(method=kind,records=97,thread_replay_exact=True,chunk_replay_exact=True,consumption_counters_match=True));print('PNR replay',kind,flush=True)
save('replay_checks.json',dict(rows=replays));save('validation.json',dict(passed=True,known_answers=len(kat),native_addresses=len(seen),domain_duplicates=0,boundary_cases=len(boundary),pnr_replay_methods=2,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('all domain gates passed',flush=True)
