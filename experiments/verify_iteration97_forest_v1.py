"""Verify individual PNR paths, archived optical probabilities, public CLI and both detector routes."""
import argparse,ctypes as ct,subprocess,sys
from fractions import Fraction as F
from pathlib import Path
from threadpoolctl import threadpool_limits
from iteration97_forest_v1 import local_tables,edge_tables
from iteration97_bridges_v1 import covariance,network
from iteration97_field_v1 import Field
from iteration96_group_benchmark_v1 import fock
from iteration97_conditionals_v1 import ROOT,np,H,save,manifest,Kernel
from iteration96_group_common_v1 import read
p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
data=ROOT/'results';names=['build','panel','field','validation','bridges','bridge_baselines','forest'];dirs=[data/f'iteration97_{n}_v1' for n in names]
cli=ROOT/'experiments/iteration97_sample_cells_v1.py';files=[Path(__file__),cli,*(d/'manifest.json' for d in dirs)]
save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},scope='archive bytes; full Fock probabilities through 8 photons; classical detection control; CLI replay'))
hashes=0;plans=0
for d in dirs:
    m=read(d/'manifest.json');assert set(m)=={f.name for f in d.iterdir() if f.name!='manifest.json'}
    for k,v in m.items():assert H(d/k)==v,(d,k);hashes+=1
    for k,v in read(d/'plan.json')['hashes'].items():assert H(ROOT/k)==v,(d,k);plans+=1
kernel=Kernel();lib=ct.CDLL(str(dirs[-1]/'forest.dll'))
tables,surv=local_tables(lib,7,F(36,625));_,_,_,keys,_,prob=tables[1]
weights=np.zeros(0);lookup={}
for key,x in zip(keys,prob):lookup[int(key)]=lookup.get(int(key),0.)+x
ref=np.load(dirs[4]/'input_1.npz');states,target=fock(kernel.prior,ref['barg'],ref['q'],8)
r=F.from_float(float(np.sin(.18)**2));edge,_=edge_tables(7,r)
model=np.zeros(len(states));balanced=states[:,:8].sum(1)==states[:,8:].sum(1)
s=states[balanced].astype(int);a=s[:,8:12].sum(1)-s[:,:3].sum(1);b=s[:,12:].sum(1)-s[:,5:8].sum(1)
allowed=(a>=0)&(b>=0)&(a+b==s[:,3:5].sum(1));s=s[allowed];a=a[allowed];b=b[allowed]
first=np.column_stack([s[:,:3],a,s[:,8:12]]);second=np.column_stack([b,s[:,5:8],s[:,12:16]])
def val(x):
    packed=x@(16**np.arange(8,dtype=np.int64));return np.array([lookup.get(int(k),0.) for k in packed])
k=s[:,3];cdf=edge[1];mass=(cdf[a,b,k]-np.where(k>0,cdf[a,b,np.maximum(k-1,0)],0))/float(2**53)
model[np.flatnonzero(balanced)[allowed]]=val(first)*val(second)*mass
delta=abs(model-target/float(surv**2))
assert delta.max()<2e-13 and delta.sum()<2e-12,(delta.max(),delta.sum())
for route in ['click','pnr']:
    def launch(suffix,n,off,workers):
        path=out/f'{route}_{suffix}.npz';subprocess.run([sys.executable,str(cli),str(path),'--route',route,'--cells','5','--seed','97160000','--reflectivity','1/16','--shots',str(n),'--offset',str(off),'--workers',str(workers)],check=True)
        return np.load(path)['records']
    assert np.array_equal(launch('all',128,0,4)[17:],launch('part',111,17,1))
field=Field(dirs[2]/'field.dll',np.zeros((2,1)),np.array([1.,1.]),1);poisson=[]
with threadpool_limits(limits=1,user_api='blas'):
    for route in (0,1):
        x,err=field.sample(131072,route,4,97160100+route,0);assert not err.any()
        observed=float(np.mean(x==0));assert abs(observed-np.exp(-1))<.005
        poisson.append(observed)
save(out,'validation.json',dict(passed=True,hashes=hashes,input_hashes=plans,reference_fock_patterns=len(target),
    fock_probability_max_error=float(delta.max()),fock_probability_l1_error=float(delta.sum()),full_cell_survival=str(surv),
    coherent_vacuum=np.exp(-1),observed_zero=poisson,cli_replay=True))
manifest(out)
