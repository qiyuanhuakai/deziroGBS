"""Reconstruct every 16-bit outcome with independent binary128 linear algebra."""
import argparse
import ctypes as ct
from math import comb
from pathlib import Path
from time import perf_counter
from iteration96_group_common_v1 import ROOT,np,shutil,subprocess,I,D,bind,H,save,manifest,Native
from iteration96_group_benchmark_v1 import BUILD
p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False)
dirs=[ROOT/'results'/('iteration96_'+s) for s in ['optical_v1','component_v1','frugal16_v2','herald_v1']]
source=ROOT/'native/click_oracle96_v1.cpp';files=[source,Path(__file__),*(d/'manifest.json' for d in dirs)]
save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},scope='full 16-bit binary128 oracle; all-occupancy geometric twin bound'))
cmd=[shutil.which('g++'),'-shared','-std=c++20','-O3','-ffp-contract=off',str(source),'-o',str(out/'oracle.dll'),'-lquadmath']
run=subprocess.run(cmd,capture_output=True,text=True);save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr))
run.check_returncode()
runtime=Native(BUILD);lib=ct.CDLL(cmd[-2]);fn=bind(lib,'oracle96',[I,I,D,D,D,D])
rows=[]
for name in ['twins','weak','mixed','actual16']:
    ref=np.load(dirs[0]/f'{name}_reference.npz');q,mu=ref['q'],ref['mu']
    law=np.zeros(65536);audit=np.empty(2);start=perf_counter()
    assert fn(16,len(q),q,mu,law,audit)==0 and abs(audit[1]-1)<1e-14
    assert audit[0]<2e-13
    row=dict(case=name,negative_mass=float(audit[0]),seconds=perf_counter()-start)
    for key,arr in [('direct',np.load(dirs[1]/f'{name}.npz')['law']),
                    ('tree',np.load(dirs[0]/f'{name}_3.npz')['law']),
                    ('k2',np.load(dirs[2]/f'{name}_2.npz')['law']),
                    ('k4',np.load(dirs[2]/f'{name}_4.npz')['law']),('first_oracle',ref['click'])]:
        row[key]=float(np.abs(law-arr).sum()/2)
    if name=='twins':
        a=np.arange(256);t=np.zeros_like(law)
        n=np.array([int(i).bit_count() for i in a]);t[a+(a<<8)]=(.25**n)*(.75**(8-n))
        row['analytic_tv']=float(abs(t-law).sum()/2);assert row['analytic_tv']<1e-12
    assert row['first_oracle']<1e-10 and row['direct']<1e-10
    np.savez_compressed(out/f'{name}.npz',reference=law)
    rows.append(row);print(row,flush=True)
save(out,'summary.json',rows);manifest(out)
