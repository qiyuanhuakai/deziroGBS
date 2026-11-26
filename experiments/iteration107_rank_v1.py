"""Complete click projections of disjoint local transport matrices and increasing occupied source rank."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import I,D,bind,save,manifest,Native
from iteration100_baselines_v1 import frugal,pair_reference,purification,pe_sample
from iteration106_field_v1 import factors,BASE
from iteration106_width_v1 import case
from iteration106_sample_v1 import benchmark,read
from iteration107_cliques_v1 import Generic,prepare,optical_x,DATA


def gaussian_click(oracle,x,profile):
    d=x.shape[1]//2;p=np.zeros(1<<d)
    for w,g in zip(profile['weights'],profile['values']):
        part=np.empty_like(p)
        assert oracle(d,len(x),np.eye(2*d)[None,:,:]+g*x,np.zeros((len(x),2*d)),part,np.empty(2))==0
        p+=w*part
    return p


def projection(t,r,config,ids,oracle):
    selected=set(map(int,ids));tiles=[];union=[];natural=[]
    for row in config['plan']:
        d=row[8];ports=list(map(int,row[:d]));keep=[j for j,k in enumerate(ports) if k in selected]
        if not keep:continue
        union.extend(ports);natural.extend(ports[j] for j in keep)
        states=np.arange(1<<d)
        subgroup=sum(((states>>j)&1)<<k for k,j in enumerate(keep))
        move=config['click_move'][row[9]+states].astype(float)
        cdf=config['click_cdf'][row[9]+states]
        excess=np.diff(cdf,prepend=np.uint64(0))/2**52
        pr=np.bincount(subgroup,weights=excess,minlength=1<<len(keep))
        mat=move[:,None]*pr
        mat[states,subgroup]+=1-move
        tiles.append(mat)
    if len(union)>18:raise ValueError(f'Projection touches {len(union)} raw bits')
    f=factors(t[union],r);xc=np.stack([2*a.T@a for a in f])
    law=gaussian_click(oracle,xc,config)
    shape=tuple(mat.shape[0] for mat in tiles[::-1]);a=law.reshape(shape)
    for j,mat in enumerate(tiles):
        axis=len(tiles)-1-j
        a=np.moveaxis(np.tensordot(a,mat,axes=([axis],[0])),-1,axis)
    words=np.arange(a.size);indices=sum(((words>>natural.index(int(i)))&1)<<j for j,i in enumerate(ids))
    return np.bincount(indices,weights=a.ravel(),minlength=len(words)),len(union)


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration105_source_groups_v1/extended.dll')),'oracle96',[I,I,D,D,D,D])
    quad=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    old=read(DATA/'iteration106_width_v1/summary.json')['rows'];rows=[];peers=[]
    one=dict(values=[1.],weights=[1.])
    for j,(m,k) in enumerate([(4,2),(16,2),(64,8),(256,16),(1024,32),(4096,64)]):
        t,r=case(m,k);start=perf_counter();config=prepare(t,r);model=Generic(t,r,config);torch.cuda.synchronize();prep=perf_counter()-start
        prev=next(z for z in old if z['m']==m and z['sources']==k);ids=np.array(prev['panel'])
        result=[]
        for ix in [ids,np.arange(min(16,m))]:
            target=gaussian_click(quad,optical_x(t[ix],r),one)
            f=factors(t[ix],r);base=gaussian_click(oracle,np.stack([2*x.T@x for x in f]),config)
            p,u=projection(t,r,config,ix,oracle)
            assert np.isfinite(p).all() and p.min()>-1e-10 and abs(p.sum()-1)<1e-9
            result.append(dict(ids=ix.tolist(),raw_union=u,base_tv=float(abs(target-base).sum()/2),local_tv=float(abs(target-p).sum()/2)))
        assert abs(result[0]['base_tv']-prev['tv'])<1e-9
        n=prev['batch']
        speed=[benchmark(lambda n,s,o:model.draw(n,pnr,s,o),n,107020000+20*j+pnr,7)['us'] for pnr in (False,True)]
        replay=model.draw(81,True,107021000+j)
        assert np.array_equal(replay[17:],model.draw(64,True,107021000+j,17))
        rows.append(dict(m=m,sources=k,prep_s=prep,batch=n,click_us=speed[0],pnr_us=speed[1],base_click_us=prev['click_us'],base_pnr_us=prev['pnr_us'],panels=result,number_states=len(config['pnr_move']),
                largest_clique_count_tv_before=float(config['coverage'][:,1].max()),largest_count_rule_rounding=float(config['coverage'][:,3].max()),whole_tail_budget=float(config['tail_budget'])))
        if m in (16,4096):
            lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'));xx=optical_x(t[:16],r)
            truth=gaussian_click(quad,xx,one)
            for K in (2,4):
                pr,ts=frugal(lib,np.eye(32)[None,:,:]+xx,K,107025000+100*j+10*K)
                peers.append(dict(m=m,K=K,observed=16,us=float(np.median(ts)),tv=float(abs(truth-pr).sum()/2)))
        if m==4096:
            np.savez_compressed(out/'case4096.npz',**config)
            np.savez_compressed(out/'input4096.npz',transfer=t,squeezing=r)
            ix=np.array([0,1]);q=np.eye(4)+optical_x(t[ix],r)
            target=pair_reference(q,12)
            h=np.zeros(169,np.int64);b=h.copy();baseline=h.copy()
            for off in range(0,262144,1024):
                x,v=model.draw(1024,True,107027000,off,paired=True)
                for a,z in [(x,h),(v,baseline)]:z+=np.bincount(a[:,ix].clip(0,12)@np.array([1,13]),minlength=169)
            st,u,_=purification(q[0]);tim=[]
            ph=np.zeros(169,np.int64);equal=np.zeros_like(ph)
            for rep in range(16):
                ti=perf_counter()
                x,stats=pe_sample(st,u,2048,seed=107028000+rep,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
                if rep:tim.append((perf_counter()-ti)*1e6/2048)
                assert not np.any(stats.censored)
                ph+=np.bincount(np.minimum(x[:,:2],12)@np.array([1,13]),minlength=169)
                xx=model.draw(2048,True,107029000,rep*2048)
                equal+=np.bincount(np.minimum(xx[:,ix],12)@np.array([1,13]),minlength=169)
            rows[-1]['pnr_pair']=dict(ids=ix.tolist(),native_records=int(h.sum()),empirical_tv=float(abs(h/h.sum()-target).sum()/2),base_empirical_tv=float(abs(baseline/baseline.sum()-target).sum()/2),equal_n=int(ph.sum()),equal_native_tv=float(abs(equal/equal.sum()-target).sum()/2),pe_empirical_tv=float(abs(ph/ph.sum()-target).sum()/2),pe_us=float(np.median(tim)))
            np.savez_compressed(out/'pnr4096.npz',truth=target,local=h,base=baseline,pe=ph,local_equal=equal)
        print(m,rows[-1],flush=True)
    save(out,'summary.json',dict(rows=rows,frugal=peers))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
