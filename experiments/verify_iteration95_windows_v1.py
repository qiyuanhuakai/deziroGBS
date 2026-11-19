"""Independent implementation of Gaussian subset laws, window counts and local witnesses."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
results=root/'results'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
verified=0
for suffix in ('order_extension_v2','window_panel_v1','ensemble_v1','ensemble_v2','ensemble_timing_v1'):
    folder=results/('iteration95_'+suffix)
    manifest=read(folder/'manifest.json')
    assert set(manifest)=={p.name for p in folder.iterdir() if p.name!='manifest.json'}
    for name,expected in manifest.items():
        assert digest(folder/name)==expected
        verified+=1
    plan=read(folder/'plan.json')
    for name,expected in (plan.get('hashes') or plan.get('files')).items():
        assert digest(root/name)==expected

pnr=results/'iteration95_order_extension_v2'
complete=np.ones((2,12),bool)
for j,budget in enumerate((.1,.01)):
    for k in (0,1):
        for offset in (0,4,8):
            z=np.load(pnr/f'raw_{j}_{offset}_{k}.npz')
            ok=z['errors']==0
            assert np.isin(z['errors'],[0,2]).all() and np.all(z['statistics'][:,5]<=budget)
            assert np.array_equal(z['counts'].sum(1),z['statistics'][:,0])
            complete[k,offset:offset+4]&=ok
assert np.array_equal(complete[0],complete[1]) and complete.sum(1).tolist()==[7,7]
first=np.load(results/'iteration95_order_extension_run1/raw_0_0_0.npz')
reused=np.load(pnr/'raw_0_0_0.npz')
assert all(np.array_equal(first[k],reused[k]) for k in first.files)

folder=results/'iteration95_ensemble_v2'
summary=read(folder/'summary.json')
hwdir=results/'iteration81_jiuzhang_accuracy_v3'
hw=np.load(hwdir/'histograms.npz')
layout=np.load(results/'iteration83_loss_pilot_v1/input.npz')['order']
hardware_to_internal=np.argsort(layout)
factors=[]
for j in (0,1):
    z=np.load(results/f'iteration94_actual_quality_v1/input_{j}.npz')
    u=z['u'][hardware_to_internal]
    a=np.block([[u.real,-u.imag],[u.imag,u.real]])
    t=z['t'];noise=z['noise']
    assert not z['alpha'].any()
    core=np.diag(np.r_[t/(1-t),-t/(1+t)])+2*noise@noise.T
    factors.append((a,core))

def marginal(ports):
    n=len(ports); masks=np.arange(2**n)
    indices=np.r_[ports,ports+1152]
    covs=[np.eye(2*n)+a[indices]@core@a[indices].T for a,core in factors]
    vac=np.ones(len(masks),np.longdouble)
    for k in range(1,n+1):
        active=[int(x) for x in masks if int(x).bit_count()==k]
        selected=np.array([[b for b in range(n) if x&(1<<b)] for x in active])
        selected=np.concatenate([selected,selected+n],axis=1)
        for q in covs:
            sign,logdet=np.linalg.slogdet(q[selected[:,:,None],selected[:,None,:]])
            assert (sign>0).all()
            vac[active]*=np.exp(-logdet.astype(np.longdouble)/2)
    prob=vac[::-1].copy()
    for b in range(n):
        mask=masks[(masks&(1<<b))!=0];prob[mask]-=prob[mask^(1<<b)]
    assert min(prob)>-1e-13 and abs(sum(prob)-1)<1e-12
    return np.maximum(prob,0.).astype(float)

groups=hw['ports']
ideal=np.stack([np.load(hwdir/f'input_{i}.npz')['two_common_sectors'] for i in range(3)])
ideal_error=max(float(abs(marginal(g)-a).max()) for g,a in zip(groups,ideal))
assert ideal_error<1e-12
outcomes=np.arange(16)
pop=np.array([i.bit_count() for i in range(16)])
selector=(3,32,255)
audit=[]
for v0 in (0,3):
    saved=[np.load(folder/f'quality_4_{v0+k}.npz') for k in range(3)]
    order=saved[0]['order']
    nodes=np.concatenate([z['slopes'] for z in saved])
    # Forward exponential weights, independent of native backward tilt construction.
    event=np.ones((len(nodes),1,10))
    total=np.ones((1,1))
    for start in range(0,1152,4):
        ports=order[max(0,start-4):start+4]
        cond=marginal(ports).reshape(16,-1).T
        assert cond.sum(1).min()>0
        cond/=cond.sum(1)[:,None]
        raw=cond.T@total
        new=np.zeros((16,raw.shape[1]+4))
        for word,k in enumerate(pop):new[word,k:k+raw.shape[1]]=raw[word]
        total=new
        allow=np.ones((16,10),bool)
        for i,g in enumerate(groups):
            for j,mask in enumerate(selector):
                watched=set(g[b] for b in range(8) if mask&(1<<b))
                for bit,port in enumerate(order[start:start+4]):
                    if port in watched:allow[:,1+3*i+j]&=(outcomes&(1<<bit))==0
        event=(cond.T@event)*np.exp(nodes[:,None,None]*pop[None,:,None])*allow
        event/=event[:,:,0].sum(1)[:,None,None]
    base=total.sum(0)
    assert abs(base.sum()-1)<1e-11
    support=base>0
    x=np.arange(1153);idx=0
    for k,z in enumerate(saved):
        row=next(r for r in summary['rows'] if r['case']==4 and r['variant']==v0+k)
        w=z['weights'];slopes=z['slopes'];length=len(w)
        logp=np.log(base[support])[None,:]+slopes[:,None]*x[support][None,:]
        p=np.exp(logp-logp.max(1,keepdims=True));p/=p.sum(1)[:,None]
        dist=np.zeros(1153);dist[support]=w@p
        count_error=float(abs(dist-z['count']).sum()/2)
        pred=w@event[idx:idx+length].sum(1);idx+=length
        local=z['marginal']
        event_error=max(abs(pred[1+3*i+j]-local[i,(np.arange(256)&mask)==0].sum())
                        for i in range(3) for j,mask in enumerate(selector))
        assert count_error<1e-10 and event_error<1e-10
        tv=abs(local-ideal).sum(1)/2
        assert np.allclose(tv,row['eight_port_optical_tv'],atol=1e-12,rtol=0)
        bits=np.unpackbits(z['packed_bits'],axis=1,count=1152,bitorder='little')
        empirical=np.bincount(bits.sum(1).astype(int),minlength=1153).cumsum()/len(bits)
        assert abs(max(abs(empirical-dist.cumsum()))-row['sample_count_ks'])<1e-11
        true_ks=max(abs(dist.cumsum()-hw['total_click_histogram'].cumsum()/hw['total_click_histogram'].sum()))
        assert abs(true_ks-row['hardware_count_ks'])<1e-11
        audit.append(dict(variant=v0+k,count_law_tv_residual=count_error,max_local_event_residual=event_error))
timing=read(results/'iteration95_ensemble_timing_v1/summary.json')
assert len(timing['timings'])==84 and len({r['hash'] if 'hash' in r else r['sha256'] for r in timing['timings']})==84
data=dict(verified_manifest_entries=verified,independent_physical_eight_port_max_error=ideal_error,
          independent_count_laws_and_nine_events=audit,pnr_completed=complete.sum(1).tolist(),timing_batches=84)
(out/'validation.json').write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8')
print(data,flush=True)
