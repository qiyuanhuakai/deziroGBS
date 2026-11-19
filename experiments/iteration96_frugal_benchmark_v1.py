"""Compare identical zero-mean 16-port optical inputs and complete click laws."""
import argparse
import ctypes as ct
from time import perf_counter
from iteration96_group_common_v1 import ROOT,Path,np,D,B,I,bind,Native,H,save,manifest
from iteration96_group_benchmark_v1 import BUILD
import sys
sys.path.insert(0,str(ROOT/'src'))
from gbslab.frugal_law import uniform_model_transfer_bound

def or_law(laws):
    if len(laws)==1:
        return np.asarray(laws[0],float)
    # Rare 16-bit events lose significant digits in a binary64 inverse zeta transform.
    from decimal import Decimal, getcontext
    getcontext().prec=64
    cdf=np.full(len(laws[0]),Decimal(1),dtype=object)
    for law in laws:
        x=np.array([Decimal.from_float(float(p)) for p in law],dtype=object)
        for i in range(16):
            v=x.reshape(-1,2,1<<i);v[:,1]+=v[:,0]
        cdf*=x
    for i in range(16):
        v=cdf.reshape(-1,2,1<<i);v[:,1]-=v[:,0]
    assert cdf.min()>-1e-14 and abs(cdf.sum()-1)<1e-12
    return cdf.astype(float)
p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args()
out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
native=Native(BUILD)
dll=ROOT/'results/iteration96_frugal_build_v1/frugal.dll'
inputs=ROOT/'results/iteration96_optical_v1';names=['twins','weak','mixed','actual16']
files=[Path(__file__),dll,ROOT/'src/gbslab/frugal_law.py',*(inputs/f'{n}_reference.npz' for n in names)]
save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},
    scope='complete continuous-uniform tree with explicit transfer bound to independent uint64 words on pinned GCC',K=[2,4],batch=2048,reps=5))
lib=ct.CDLL(str(dll));msg=ct.create_string_buffer(2048)
tree=bind(lib,'fc96_full16',[I,D,D,D,D,D,ct.c_char_p,I]);sample=bind(lib,'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I])
rows=[]
for c,name in enumerate(names):
    data=np.load(inputs/f'{name}_reference.npz')
    assert not data['mu'].any()
    covs=2*data['q']-np.eye(32)
    for cutoff in [2,4]:
        laws=[];payload={};prefix_checks=[]
        for j,cov in enumerate(covs):
            q=np.empty(65535);prev=q.copy();prop=q.copy();st=np.zeros(3)
            assert tree(cutoff,cov,q,prev,prop,st,msg,len(msg))==0,msg.value
            assert st[1]==16 and st[2]==16*st[0]
            prefix_checks.append(int(st[2]))
            prob=np.ones(1,np.longdouble)
            for d in range(16):
                start=2**d-1;z=np.clip(np.nan_to_num(q[start:start+len(prob)],nan=1),0,1).astype(np.longdouble)
                prob=np.r_[prob*z,prob*(1-z)]
            laws.append(prob);payload[f'threshold_{j}']=q
        law=or_law(laws);payload['law']=law
        times=[];wall=[];phases=[]
        for rep in range(5):
            start=perf_counter();recs=[];stats=[]
            for j,cov in enumerate(covs):
                x=np.empty((2048,16),np.uint8);st=np.zeros(7)
                assert sample(cutoff,2048,96020000+100*c+10*rep+j,cov,x,st,msg,len(msg))==0,msg.value
                assert st[5]==64
                recs.append(x);stats.append(st)
            merge=perf_counter();records=np.bitwise_or.reduce(recs,axis=0) if len(recs)>1 else recs[0]
            merge_s=perf_counter()-merge;wall.append((perf_counter()-start)*1e6/2048)
            times.append((sum(s[3]+s[4] for s in stats)+merge_s)*1e6/2048);phases.append(np.array(stats).tolist())
        tv=float(np.abs(law-data['click']).sum()/2)
        rows.append(dict(case=name,K=cutoff,full_click_tv=tv,us=times,us_median=float(np.median(times)),wall_us=wall,
            phases=phases,prefix_checks=prefix_checks,uint64_transfer_bound=len(covs)*float(uniform_model_transfer_bound(16))))
        payload['last_batch']=records;np.savez_compressed(out/f'{name}_{cutoff}.npz',**payload)
        print(name,cutoff,tv,np.median(times),flush=True);save(out,'summary.json',rows)
manifest(out)
