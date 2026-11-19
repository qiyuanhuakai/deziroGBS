"""Fresh timing-only confirmation of both nine-component 1152-port models."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np

root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser()
p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
source=root/'results/iteration95_ensemble_v2'
dll=source/'ensemble.dll';layout=root/'results/iteration83_loss_pilot_v1/input.npz'
optics=[root/f'results/iteration94_actual_quality_v1/input_{j}.npz' for j in range(2)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,v):(out/name).write_text(json.dumps(v,indent=2)+'\n',encoding='utf-8')
config=[(v,n,t) for v in (2,5) for n in (8192,32768) for t in (1,8,16)]
save('plan.json',dict(configs=config,repetitions=7,rotation=5,worker_runtimes='CPU threads; full byte outputs in physical order',
    frozen_quality='two nine-node configurations; no tuning',experiment='0x95900000+256*(rep+1)+job, latency 0x95910000+variant',
    hashes={p.relative_to(root).as_posix():sha(p) for p in [Path(__file__),dll,layout,*optics,source/'quality_4_2.npz',source/'quality_4_5.npz']}))
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
lib=C.CDLL(str(dll))
I,P,U=C.c_int,C.c_void_p,C.c_uint64
double,integer,byte=[np.ctypeslib.ndpointer(dtype=t,flags='C_CONTIGUOUS') for t in (np.float64,np.int32,np.uint8)]
def binding(name,args,result=I):
    fn=getattr(lib,name);fn.argtypes=args;fn.restype=result;return fn
mom=binding('window95_moments',[I,I,double,double,double,double,double,double])
prepare=binding('window95_prepare',[I,I,I,I,double,double,double],P)
tilt=binding('window95_tilt',[P,I,C.c_double],P)
make=binding('ensemble95_prepare',[I,C.POINTER(P),double],P)
order=binding('ensemble95_output_order',[P,integer])
draw=binding('ensemble95_sample',[P,I,U,U,I,byte])
release=binding('window95_destroy',[P],None)
release_fast=binding('ensemble95_destroy',[P],None)
m=1152
detector=np.load(layout)['order']
q=np.empty((2,2*m,2*m));mu=np.empty((2,2*m))
for j,path in enumerate(optics):
    z=np.load(path);u=np.ascontiguousarray(z['u'])
    assert mom(m,len(z['t']),u.view(float),z['t'],z['alpha'].view(float),z['noise'],q[j],mu[j])==0
handles={}
for v in (2,5):
    z=np.load(source/f'quality_4_{v}.npz')
    perm=z['order'];internal=np.argsort(detector)[perm];ix=np.r_[internal,internal+m]
    base=prepare(m,4,4,2,np.ascontiguousarray(q[:,ix][:,:,ix]),np.ascontiguousarray(mu[:,ix]),np.zeros(4))
    assert base
    plans=[tilt(base,4,s) for s in z['slopes']]
    assert all(plans)
    ptr=make(9,(P*9)(*plans),z['weights'])
    assert ptr and order(ptr,perm)==0
    sample=np.empty((16,m),np.uint8)
    assert draw(ptr,16,0x95600000+64*4+v,333,3,sample)==0
    assert np.array_equal(np.packbits(sample,axis=1,bitorder='little'),z['packed_bits'][333:349])
    for plan in plans+[base]:release(plan)
    handles[v]=ptr
times=[]
for rep in range(-1,7):
    for k in range(len(config)):
        job=(k+5*rep)%len(config);v,n,t=config[job]
        if rep<0:n=256
        start=time.perf_counter()
        bits=np.empty((n,m),np.uint8)
        rc=draw(handles[v],n,0x95900000+256*(rep+1)+job,0,t,bits)
        seconds=time.perf_counter()-start
        assert rc==0
        if rep>=0:times.append(dict(variant=v,n=n,threads=t,rep=rep,seconds=seconds,
             sha256=hashlib.sha256(bits.tobytes()).hexdigest()))
latencies={}
for v in handles:
    times1=[]
    for record in range(101):
        start=time.perf_counter();bits=np.empty((1,m),np.uint8)
        rc=draw(handles[v],1,0x95910000+v,record,1,bits)
        elapsed=time.perf_counter()-start;assert rc==0
        if record:times1.append(elapsed)
    latencies[v]=times1
    release_fast(handles[v])
save('summary.json',dict(timings=times,latencies=latencies))
save('manifest.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})
