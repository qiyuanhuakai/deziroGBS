"""Held-out optical and hardware checks, full-batch latency and two-port PNR TV."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import HIST,save,manifest
from iteration100_projection_v1 import physical
from iteration100_baselines_v1 import pair_reference,frugal
from iteration104_validate_v1 import interval
from iteration105_jiuzhang_v1 import inputs,xpair
from iteration106_groups_v1 import Groups,actual,synthetic,DATA
from iteration106_reference_v1 import Marked,OUT as REF,CAPS,realify
from iteration106_field_v1 import Field,actual_input,factors,OUT as FIELD


def read(path):return json.loads(path.read_text())


def panel_hist(x,ids):
    codes=(x[:,ids]>0).astype(np.uint32)@(np.uint32(1)<<np.arange(len(ids),dtype=np.uint32))
    return np.bincount(codes,minlength=1<<len(ids))


def benchmark(draw,batch,seed,reps):
    times=[]
    for i in range(reps+2):
        start=perf_counter();draw(batch,seed,batch*i)
        elapsed=(perf_counter()-start)*1e6/batch
        if i>=2:times.append(elapsed)
    return dict(batch=batch,us=float(np.median(times)),quartiles=np.quantile(times,[.25,.75]).tolist(),repeats=times)


def pair_group(marked,ids,clip):
    length=32
    points=np.exp(-2j*np.pi*np.arange(length)/length)
    z=np.stack(np.meshgrid(points,points,indexing='xy'),axis=-1).reshape(-1,2)
    p=np.fft.ifft2(marked.evaluate(ids,z).reshape(length,length)).real
    assert p.min()>-1e-12 and abs(p.sum()-1)<1e-10
    out=np.zeros((clip+1,clip+1))
    i=np.minimum(np.arange(length),clip)
    np.add.at(out,(i[:,None],i[None,:]),p)
    # Marginal Markov bounds on P(n_i>=32), with real evaluation points in the convergent range.
    high=2.
    bound=0.
    for j in range(2):
        v=np.ones((1,2));v[0,j]=high
        mgf=marked.evaluate(ids,v)[0]
        assert abs(mgf.imag)<1e-10 and 1<mgf.real<5
        bound+=mgf.real/high**length
    return out.ravel(),float(bound)


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    t,r,nodes=actual(CAPS)
    start=perf_counter();group=Groups(t,r,nodes);gp=perf_counter()-start
    t0,r0=actual_input();profile=read(FIELD/'summary.json')['selected']
    tstart=perf_counter();field=Field(t0,r0,profile);torch.cuda.synchronize();cold=perf_counter()-tstart
    tstart=perf_counter();field=Field(t0,r0,profile);torch.cuda.synchronize();warm=perf_counter()-tstart
    wide=Field(t0,r0,profile,True)
    refs=[np.load(REF/f'jiuzhang_{j}.npz') for j in range(4)]
    actual_panels=[v['ports'] for v in refs]
    field_refs=[np.load(FIELD/f'field_{profile["d"]}_panel_{j}.npz')['model'] for j in range(4)]
    hw=np.load(HIST)['total_click_histogram']
    peer=read(DATA/'iteration103_baselines_v1/summary.json')['pnr'][0]
    ids=np.array(peer['ports'])
    truth=pair_reference(physical(inputs()[0],ids),12)
    fp=np.zeros_like(truth);sub=np.r_[ids,len(t0)+ids]
    for weight,v in zip(profile['weights'],profile['values']):
        q=np.array([np.eye(4)+2*v*(f[:,sub].T@f[:,sub]) for f in factors(t0,r0)])
        fp+=weight*pair_reference(q,12)
    group_pair,alias=pair_group(Marked(REF/'laws.dll',t,r,nodes),ids,12)
    tail=float(truth.reshape(13,13)[-1].sum()+truth.reshape(13,13)[:-1,-1].sum())
    perf=[]
    methods=[('group1',lambda n,p,s,o:group.draw(n,p,s,o,1)),('group16',lambda n,p,s,o:group.draw(n,p,s,o,16)),('field',field.draw)]
    for j,(name,draw) in enumerate(methods):
        for pnr in (False,True):
            for k,n in enumerate(([4,2048] if j<2 else [4,64,2048,16384])):
                row=benchmark(lambda n,s,o:draw(n,pnr,s,o),n,106040000+100*j+10*pnr+k,9)
                row.update(method=name,pnr=pnr)
                perf.append(row)
    # Same addresses are intentional for replay, distinct from scientific runs.
    for j,(name,draw) in enumerate(methods):
        x=draw(1024,False,106041000+j,11)
        y=draw(891,True,106041000+j,144)
        assert np.array_equal(x[133:],y>0),name
    fp32=field.draw(4096,True,106042000);fp64=wide.draw(4096,True,106042000)
    delta=int(np.count_nonzero(fp32!=fp64))
    runs=[]
    for ii,(name,draw) in enumerate(methods[1:]):
        laws=field_refs if name=='field' else [z['model'] for z in refs]
        pair=fp if name=='field' else group_pair
        photons=np.load(FIELD/f'field_{profile["d"]}_total.npz' if name=='field' else REF/'total.npz')['model']
        for pnr in (False,True):
            n=262144 if name=='group16' else 1048576
            hs=[np.zeros(1<<len(x),np.int64) for x in actual_panels]
            c=np.zeros(1153,np.int64);pn=np.zeros(1024,np.int64);pairs=np.zeros(169,np.int64)
            for off in range(0,n,2048):
                x=draw(2048,pnr,106043000+2*ii+pnr,off)
                for h,ids0 in zip(hs,actual_panels):h+=panel_hist(x,ids0)
                c+=np.bincount(np.count_nonzero(x,axis=1),minlength=len(c))
                if pnr:
                    pn+=np.bincount(x.sum(1).astype(np.int64),minlength=len(pn))
                    pairs+=np.bincount(np.minimum(x[:,ids],12)@np.array([1,13]),minlength=169)
            biases=[float(h[law>z['target']].sum()/n-law[law>z['target']].sum()) for h,law,z in zip(hs,laws,refs)]
            assert max(map(abs,biases))<.008,(name,biases)
            row=dict(method=name,pnr=pnr,n=n,panel_event_gaps=biases,hardware_ks=float(abs(c.cumsum()/n-hw.cumsum()/hw.sum()).max()))
            if pnr:
                row['total_ks_to_model']=float(abs(pn.cumsum()/n-photons.cumsum()).max())
                row['pair_empirical_tv_to_model']=float(abs(pairs/n-pair).sum()/2)
                assert row['total_ks_to_model']<.009 and row['pair_empirical_tv_to_model']<.006,(name,row)
            runs.append(row)
            np.savez_compressed(out/f'{name}_{int(pnr)}.npz',click=c,photon=pn,pair=pairs,**{f'panel_{i}':v for i,v in enumerate(hs)})
            print(row,flush=True)
    np.savez_compressed(out/'pair.npz',target=truth,field=fp,groups=group_pair,ports=ids)
    save(out,'summary.json',dict(timings=perf,runs=runs,hardware_records=int(hw.sum()),prep_s=dict(groups=gp,field_first=cold,field_warm=warm),
         fp32_vs_fp64_cells=delta,fp32_vs_fp64_total=fp32.size,pair=dict(ports=ids.tolist(),groups_tv=float(abs(group_pair-truth).sum()/2),field_tv=float(abs(fp-truth).sum()/2),reference_tail=tail,groups_alias_bound=alias,photonic_engines=peer)))
    manifest(out)


if __name__=='__main__':
    torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
