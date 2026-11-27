"""Same observed 16 multiplexed channels for optical reference, tilted readouts, Frugal and PE."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration115_tilt_v1 import Device,setup
from iteration115_visible_v1 import DATA,P,I,U,J,hardware
from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration100_baselines_v1 import purification,pair_reference
from iteration92_pe_common_v1 import pe_sample
from iteration112_coarse_v1 import Native,alias


def lookup(law,core):
    start=perf_counter();prob,alt=alias(law)
    prep=perf_counter()-start
    fn=core.lib.categorical112;fn.argtypes=[I,I,I,U,J,P,P,P]
    n=4096;a=np.empty((n,16),np.uint8);times=[]
    for i in range(16):
        start=perf_counter()
        assert fn(n,16,1,115612000+i,0,alt.ctypes.data,prob.ctypes.data,a.ctypes.data)==0
        if i>1:times.append((perf_counter()-start)*1e6/n)
    return dict(alias_setup_s=prep,us=float(np.median(times)),times=times)


def panel_law(model,a,b):
    split=np.diff(model['split'],axis=1,prepend=0)
    masks=(np.arange(256)[:,None]>>np.arange(8))&1
    w1=(masks@split[a])[np.arange(65536)&255]
    w2=(masks@split[b])[np.arange(65536)>>8]
    w=np.stack([w1,w1,w2,w2],axis=1)
    i=[2*a,2*a+1,2*b,2*b+1]
    vacuum=np.zeros(65536)
    for weight,x in zip(model['weights'],model['marginal']):
        v=np.ones(65536)
        for mat in x:
            q=mat[np.ix_(i,i)]
            v*=np.exp(-.5*np.linalg.slogdet(np.eye(4)+q*w[:,None,:])[1])
        vacuum+=weight*v
    law=vacuum[::-1].astype(np.longdouble)
    for j in range(16):
        view=law.reshape(-1,2,1<<j);view[:,1]-=view[:,0]
    assert law.min()>-1e-10 and abs(law.sum()-1)<1e-10
    return np.asarray(law,float)


def run(out, count, block):
    if count < 2*block or block < 1 or count % block:
        raise ValueError("shots must be a multiple of block with at least two blocks")
    out.mkdir(exist_ok=True,parents=True);torch.set_float32_matmul_precision('highest')
    raw,r=actual_input();t,ss=hardware()
    args,_,model=setup(t,ss)
    banks=[34,35];ids=(8*np.array(banks)[:,None]+np.arange(8)).ravel()
    x=optical_x(raw[ids],r)
    quad=oracle(True);start=perf_counter();exact=gaussian_click(quad,x,ONE);enum=perf_counter()-start
    law=panel_law(model,*banks)
    arrays=dict(optical=exact,model=law)
    info=dict(full_model_tv=float(abs(law-exact).sum()/2),exact_enumeration_s=enum,
              optical_cached=lookup(exact,Native(out)),rows=[])
    # Also compare the first 16 detectors with the archived full-width Frugal prefix.
    prefix=panel_law(model,0,1)
    exact_prefix=gaussian_click(quad,optical_x(raw[:16],r),ONE)
    info['full_device_first16_model_tv']=float(abs(prefix-exact_prefix).sum()/2)
    arrays.update(prefix=prefix,prefix_exact=exact_prefix)
    mat,split,trans,scales,tab,gates,cdf=args
    local=[mat.reshape(len(mat),144,4)[:,banks].reshape(len(mat),-1),split[banks],trans,scales[:,banks],
           tab[:,banks],gates[:,banks],cdf]
    lib=ct.CDLL(str(DATA/'iteration115_tilt_v1/gpu.dll'));lib.normal103_launch.argtypes=[I,I,U,J,I,P,P]
    lib.tilt115_latent.argtypes=[I]*4+[U,J]+[P]*6;lib.tilt115_banks.argtypes=[I]*5+[U,J]+[P]*9
    gpu=Device(lib,local,False);pairref=pair_reference(np.eye(4)+x[:,[0,8,16,24]][:,:,[0,8,16,24]],8)
    def save(name,v,times,pnr):
        hh=np.bincount((v>0)@(1<<np.arange(16)),minlength=65536)
        row=dict(name=name,shots=count,us=float(np.median(times)),times=times,click_empirical_tv=float(abs(hh/count-exact).sum()/2))
        if pnr:
            ph=np.bincount(np.minimum(v[:,0],8)+9*np.minimum(v[:,8],8),minlength=81)
            row['pair_empirical_tv']=float(abs(ph/count-pairref).sum()/2)
            arrays[name+'_pair']=ph
        arrays[name+'_click']=hh
        info['rows'].append(row)
    for pn in (0,1):
        ts=[]
        for rep in range(10):
            start=perf_counter();a=gpu.draw(count,pn,115680000+pn,rep*count)
            if rep:ts.append((perf_counter()-start)*1e6/count)
        save('gpu_pnr' if pn else 'gpu_direct',a,ts,pn)
    assert gpu.error.item()==0
    prepared=[purification(np.eye(32)+xx) for xx in x]
    info['max_purification_error']=max(v[2] for v in prepared)
    for device in ('cpu','cuda'):
        buf=np.empty((count,16),np.int32);ts=[]
        for off in range(0,count,block):
            begin=perf_counter();acc=np.zeros((block,16),np.int32)
            for j,(state,unitary,_) in enumerate(prepared):
                rows,stat=pe_sample(state,unitary,block,device=device,seed=115710000+1000*(device=='cuda')+100*j+off//block,
                    window=4,impl='v2',max_photons=100,return_stats=True)
                assert not np.any(stat.censored)
                acc+=rows[:,:16].astype(np.int32)
            buf[off:off+block]=acc
            if off:ts.append((perf_counter()-begin)*1e6/block)
        save('photonic_engines_'+device,buf,ts,True)
        print(device,ts[-1],flush=True)
    # FP64 field and tilted latent computation on the complete 1152-bin record.
    wide=Device(lib,args,True);timings=[]
    for pn in (0,1):
        ts=[]
        for rep in range(9):
            start=perf_counter();wide.draw(8192,pn,115720000+pn,rep*8192)
            if rep>1:ts.append((perf_counter()-start)*1e6/8192)
        timings.append(dict(pnr=bool(pn),batch=8192,us=float(np.median(ts)),times=ts))
    assert wide.error.item()==0
    info['full_device_fp64']=timings
    np.savez_compressed(out/'panel.npz',pairref=pairref,**arrays)
    (out/'summary.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--shots',type=int,default=4096)
    p.add_argument('--block',type=int,default=128)
    args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output.resolve(),args.shots,args.block)
