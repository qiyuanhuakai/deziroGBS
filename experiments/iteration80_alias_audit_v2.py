"""Exact integer alias audit against normalized binary64 input tables.
This certifies the discrete table law under independent uniform uint64 words,
not the PRNG, calibration, or target quantum distribution.
"""
from pathlib import Path
import os,json,hashlib,shutil,sys,argparse,subprocess,ctypes as C
from fractions import Fraction
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_cached_transport_pnr import NativeCachedTransportPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
gcc=Path(shutil.which('g++'));runtime=[gcc.parent];eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
sources=[Path(__file__),*sorted((ROOT/'native').glob('*.cpp')),ROOT/'src/gbslab/native_cached_transport_pnr.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources}))
command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','-DEIGEN_DONT_PARALLELIZE','-I',str(eigen),str(ROOT/'native/gaussian_transport_alias_export80_v1.cpp'),'-o',str(OUT/'export.dll')]
r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
rows=[]
for case in ('strong4','identity2'):
    d=np.load(ROOT/f'results/iteration80_ipf_pilot_v1/{case}.npz')
    with NativeCachedTransportPNRSampler(d['v'],.5,OUT/'export.dll',couplings=d['couplings'],latent_weights=d['weights'],exact_total_budget=0,workers=16,runtime_dirs=runtime) as model:
        fn=model.lib.transport80b_alias;fn.argtypes=[C.c_void_p,C.c_int,C.c_int,C.POINTER(C.c_uint64),C.POINTER(C.c_int),C.POINTER(C.c_int),C.c_char_p,C.c_int];fn.restype=C.c_int
        audits=[];deltas=[]
        for label in range(-1,len(model.latent_weights)):
            weights=model.latent_weights if label<0 else model.pair_masses[label];actual=model.actual_latent_weights if label<0 else model.actual_pair_masses[label]
            capacity=1<<max(1,(len(weights)-1).bit_length());threshold=np.zeros(capacity,np.uint64);alias=np.zeros(capacity,np.int32);meta=np.zeros(3,np.int32);error=C.create_string_buffer(4096)
            rc=fn(model._plan,label,capacity,threshold.ctypes.data_as(C.POINTER(C.c_uint64)),alias.ctypes.data_as(C.POINTER(C.c_int)),meta.ctypes.data_as(C.POINTER(C.c_int)),error,len(error));assert rc==0,error.value
            bits,size,categories=map(int,meta);assert size==capacity and categories==len(weights);levels=1<<(64-bits);counts=[0]*size
            for i,(t,j) in enumerate(zip(threshold,alias)):
                t,j=int(t),int(j);assert 0<=t<=levels and 0<=j<size;counts[i]+=t;counts[j]+=levels-t
            assert sum(counts)==2**64 and not any(counts[categories:])
            ratios=[float(w).as_integer_ratio() for w in weights];den=max(b for _,b in ratios);integers=[a*(den//b) for a,b in ratios];total=sum(integers)
            delta=Fraction(sum(abs(c*total-i*2**64) for c,i in zip(counts,integers)),2*2**64*total);deltas.append(delta)
            assert all(c==0 for c,w in zip(counts,weights) if w==0)
            assert np.max(abs(np.array([float(Fraction(c,2**64)) for c in counts[:categories]])-actual))<=2**-52
            audits.append(dict(label=label,bits=bits,thresholds=[int(v) for v in threshold],aliases=alias.tolist(),word_counts=[str(v) for v in counts],tv_numerator=str(delta.numerator),tv_denominator=str(delta.denominator),tv=float(delta),positive_inputs=int(sum(weights>0)),positive_outputs=sum(c>0 for c in counts)))
        bound=deltas[0]+7*max(deltas[1:]);sample=model.sample(1024,0xb0000000000000,fixed_total=14);other=model.sample(1024,0xb0000000000000,workers=1,fixed_total=14);assert np.array_equal(sample['counts'],other['counts'])
        save(case+'.json',dict(audits=audits,N14_bound_numerator=str(bound.numerator),N14_bound_denominator=str(bound.denominator),N14_bound=float(bound)));np.savez_compressed(OUT/(case+'.npz'),counts=sample['counts']);rows.append(dict(case=case,tables=len(audits),N14_bound=float(bound)));print(rows[-1],flush=True)
save('summary.json',dict(passed=True,rows=rows));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
