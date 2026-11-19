"""Hardware validation of count-conditioned adaptive trees and a forced residual test."""
import argparse
import hashlib
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration96_group_common_v1 import ROOT,Path,np,Native,H,save,manifest,Optical,construct,block_weights,partition,pair_information,joint_values
from iteration96_group_benchmark_v1 import BUILD,fixture,HIST,DATA
p=argparse.ArgumentParser();p.add_argument('output',type=Path);out=p.parse_args().output.resolve()
out.mkdir(parents=True,exist_ok=False);lib=Native(BUILD)
clickfile=DATA/'iteration95_third_click_v1/quality_3.npz';phfile=DATA/'iteration84_photon_cdf_v1/reference.npz'
files=[Path(__file__),BUILD,HIST,clickfile,phfile,DATA/'iteration96_optical_v1/manifest.json',ROOT/'experiments/iteration96_group_benchmark_v1.py',
       ROOT/'experiments/iteration96_group_common_v1.py']
save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},sample=262144,cap=32,
    count_limits=[512,384],accounting='unsupported target mass separately recorded; raw and calibrated estimates distinguished',bound=1.25,
    intervals='two-sided DKW, alpha=.01/4 each for control and three empirical CDFs; conditional on iid complete records'))
def shot(h,m,domain,off,n,workers):
    x=np.empty((n,m),np.uint8);tries=np.zeros(n,np.uint32);tail=np.zeros(n,np.uint8);res=np.zeros(n,np.uint8)
    assert lib.draw(h,n,domain,off,1,workers,x,tries,tail,res)==0
    return x,tail,res
def prepare(opt,info,width,base,cap,limit):
    groups=partition(opt.m,width,True,info);table,diag=construct(opt,groups,block_weights(groups,info),base,cap,True)
    proposal=np.zeros(limit+1);h=lib.prepare(opt.m,len(groups),table['transition'].shape[-1],cap,int(base==3),limit,*table.values(),proposal)
    assert h and abs(proposal.sum()-1)<1e-8
    return h,table,proposal
def calibrate(h,proposal,source):
    target=source[:len(proposal)].copy()
    removed=float(source[len(proposal):].sum()+target[proposal<=0].sum());target[proposal<=0]=0;target/=target.sum()
    diag=np.empty(2);assert lib.target(h,target,1.25,diag)==0
    accept=np.floor(np.minimum(1,np.divide(target,1.25*proposal,out=np.zeros_like(target),where=proposal>0))*2**53)/2**53
    recovered=1.25*proposal*accept+np.maximum(0,target-1.25*proposal*accept)
    assert abs(recovered-target).max()<1e-12
    return target,diag,removed

with threadpool_limits(limits=1,user_api='blas'):
    q,mu,_,_=fixture('mixed');ids=[1,4,8,10,14];ix=np.r_[ids,np.array(ids)+16]
    q=q[:,ix][:,:,ix].copy();mu=mu[:,ix].copy();opt=Optical(lib,q,mu);info,_=pair_information(q)
    h,table,proposal=prepare(opt,info,2,2,2,5)
    try:
        target=np.zeros(6);target[3]=.7;target+=.3*proposal
        target,diag,_=calibrate(h,proposal,target)
        states=((np.arange(32)[:,None]>>np.arange(5))&1).astype(np.uint8);nc=states.sum(1).astype(int)
        raw=joint_values(table,2,states);exact=raw*target[nc]/proposal[nc]
        n=131072;x,tail,res=shot(h,5,96050000,0,n,16)
        hist=np.bincount(x@(2**np.arange(5)),minlength=32)/n
        assert abs(exact.sum()-1)<1e-12 and abs(hist-exact).sum()/2<.015 and abs(res.mean()-diag[0])<.01
        save(out,'control.json',dict(residual=float(diag[0]),observed=float(res.mean()),tv=float(abs(hist-exact).sum()/2),
            records=n,exact=exact.tolist(),observed_pmf=hist.tolist()))
    finally:lib.free(h)
    q,mu,_,_=fixture('actual1152');opt=Optical(lib,q,mu);info,_=pair_information(q)
    hw=np.load(HIST);hist=hw['total_click_histogram'];hardware=hist/hist.sum();groups=hw['ports'];rows=[]
    for j,(base,width,cap,limit,source) in enumerate([(2,4,4,512,np.load(clickfile)['count']),(3,2,32,384,np.load(phfile)['photon_pmf'])]):
        start=perf_counter();h,table,proposal=prepare(opt,info,width,base,cap,limit)
        try:
            target,diag,removed=calibrate(h,proposal,source);setup=perf_counter()-start;exp=96050010+j
            n=8192;allclick=np.zeros(1153,int);allph=np.zeros(1153,int);projection=np.zeros((3,256),int)
            tails=rare=0;hashes=[];times=[]
            for rep in range(32):
                x,tail,res=shot(h,1152,exp,rep*n,n,16);click=(x>0);tot=click.sum(1)
                allclick+=np.bincount(tot,minlength=1153);allph+=np.bincount(x.sum(1).astype(int),minlength=1153)
                for g,idx in enumerate(groups):projection[g]+=np.bincount(click[:,idx]@(2**np.arange(8)),minlength=256)
                tails+=int(tail.sum());rare+=int(res.sum());hashes.append(hashlib.sha256(x).hexdigest())
            for threads in [1,16]:
                stamps=[]
                for rep in range(5):
                    start=perf_counter();shot(h,1152,exp,2**20+(threads*5+rep)*32768,32768,threads)
                    stamps.append((perf_counter()-start)*1e6/32768)
                times.append(dict(threads=threads,us=stamps,median=float(np.median(stamps))))
            start_x=shot(h,1152,exp,0,32,1)[0];assert np.array_equal(start_x[8:],shot(h,1152,exp,8,24,4)[0])
            sim=allclick/allclick.sum();law=np.pad(target,(0,len(hardware)-len(target))) if base==2 else sim
            eps_hw=np.sqrt(np.log(2/.0025)/(2*hist.sum()));eps_sim=np.sqrt(np.log(2/.0025)/(2*allclick.sum()))
            hwks=float(abs(law.cumsum()-hardware.cumsum()).max())
            projtv=[float(abs(p/p.sum()-opt.law(tuple(idx),2)).sum()/2) for p,idx in zip(projection,groups)]
            row=dict(base=base,width=width,records=int(allclick.sum()),setup_s=setup,counts_target_removed=removed,
                avg_excess=float(diag[0]),acceptance=float(diag[1]),observed_rare=rare,tails=tails,
                hardware_ks=hwks,hardware_ks_halfwidth=float(eps_hw+(eps_sim if base==3 else 0)),
                empirical_projected_tv=projtv,empirical_count_target_ks=float(np.max(abs(allph.cumsum()/allph.sum()-np.pad(target.cumsum(),(0,len(allph)-len(target)),constant_values=1)))),
                us=times,hashes=hashes)
            rows.append(row);np.savez_compressed(out/f'case_{base}.npz',target=target,proposal=proposal,click=allclick,photon=allph,
                projections=projection,ports=groups,first=start_x)
            print(base,hwks,projtv,diag,times[-1]['median'],flush=True)
        finally:lib.free(h)
save(out,'summary.json',rows);manifest(out)
