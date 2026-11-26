"""Estimate preparation, full-law accuracy and hardware generalization of optical conditionals."""
import argparse,hashlib
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration97_conditionals_v1 import (Kernel,Optical,layout,fit,sample,density,ROOT,Path,np,H,
    save,manifest,BUILD,PREVIOUS,pair_information)
from iteration96_group_benchmark_v1 import fixture,fock,HIST,ORDER,ACTUAL
def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('cases',nargs='+');a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);data=ROOT/'results';ref=data/'iteration96_optical_v1'
    files=[Path(__file__),ROOT/'experiments/iteration97_conditionals_v1.py',BUILD,PREVIOUS,HIST,ORDER,
        *(ACTUAL/f'input_{i}.npz' for i in (0,1)),ROOT/'experiments/iteration96_group_benchmark_v1.py',
        *(data/'iteration96_quad_v1'/f'{n}.npz' for n in ['mixed','weak','actual16','twins'])]
    save(out,'plan.json',dict(hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},cases=a.cases,seed=97010000,
        small_history=[0,4,8,12],hardware_history=[0,4,8],hardware_n=131072,cap=32,replicates=5,
        dkw='10 empirical distributions plus one independent hardware empirical law, Bonferroni family alpha=.01; iid assumption'))
    kernel=Kernel();rows=[];names=['weak','mixed','actual16','twins','actual1152'];hw=np.load(HIST);n_hw=hw['total_click_histogram'].sum()
    for name in a.cases:
        start=perf_counter();large=name=='actual1152';q,mu,barg,_=fixture(name);opt=Optical(kernel.prior,q,mu)
        info,_=pair_information(q);m=opt.m;opt_s=perf_counter()-start
        if not large:
            st=((np.arange(1<<m)[:,None]>>np.arange(m))&1).astype(np.uint8)
            target=np.load(data/'iteration96_quad_v1'/f'{name}.npz')['reference'];target=np.maximum(target,0);target/=target.sum()
            if barg is not None:ns,fp=fock(kernel.prior,barg,q,8)
        groups_hw=hw['ports'];choices=[(2,4),(3,2)]
        for base,width in choices:
            for physical in ([False,True] if large else [False]):
                groups=layout(info,width,physical)
                for history in ([0,4,8] if large and not physical else ([8] if large else [0,4,8,12])):
                    tag=f'{name}_{base}_{int(physical)}_{history}';dom=97010000+10000*names.index(name)+1000*base+100*int(physical)+history
                    model=fit(kernel,opt,info,groups,base,history)
                    row=dict(case=name,base=base,width=width,physical=physical,history=history,table_bytes=model['bytes'],
                        input_seconds=opt_s,setup_s=model['seconds'],negative=max(x['neg'] for x in model['diagnostics']))
                    payload=dict(groups=np.array(groups),history=np.full((len(groups),history),-1,np.int32))
                    for i,h in enumerate(model['past']):payload['history'][i,:len(h)]=h
                    try:
                        if not large:
                            if base==2:
                                law=density(model,base,st);assert abs(law.sum()-1)<2e-12
                                row['click_tv']=float(abs(law-target).sum()/2)
                                payload['law']=law
                            elif barg is not None:
                                law=density(model,base,ns);tail=1-fp.sum();high=max(0.,1-np.minimum(law,fp).sum())
                                row['pnr_tv']=[max(0.,high-min(max(0.,tail),max(0.,1-law.sum()))),high]
                        x,_=sample(kernel,model,m,64,dom,0,1);y,_=sample(kernel,model,m,48,dom,16,4)
                        assert np.array_equal(y,x[16:]);payload['first64']=x
                        if large:
                            n=8192;counts=np.zeros(m+1,int);proj=np.zeros((3,256),int);hashes=[];overflow=0
                            for rep in range(16):
                                x,t=sample(kernel,model,m,n,dom,256+rep*n,16);click=x>0
                                counts+=np.bincount(click.sum(1),minlength=m+1);overflow+=int(t.sum())
                                for j,ids in enumerate(groups_hw):proj[j]+=np.bincount(click[:,ids]@(2**np.arange(8)),minlength=256)
                                hashes.append(hashlib.sha256(x).hexdigest())
                            row.update(sample_n=int(counts.sum()),hardware_ks=float(abs((counts/counts.sum()).cumsum()-(hw['total_click_histogram']/n_hw).cumsum()).max()),
                                hardware_halfwidth=float(np.sqrt(np.log(22/.01)/2)*(1/np.sqrt(counts.sum())+1/np.sqrt(n_hw))),
                                projected_empirical_tv=[float(abs(p/p.sum()-opt.law(tuple(ids),2)).sum()/2) for p,ids in zip(proj,groups_hw)],
                                overflows=overflow,hashes=hashes)
                            payload.update(counts=counts,projections=proj)
                        timing=[];n=32768 if large else 2048
                        for workers in ([1,16] if large else [1]):
                            stamps=[]
                            for rep in range(5):
                                start=perf_counter();sample(kernel,model,m,n,dom,(1<<23)+(workers*5+rep)*n,workers)
                                stamps.append((perf_counter()-start)*1e6/n)
                            timing.append(dict(workers=workers,us=stamps,median=float(np.median(stamps)),batch=n))
                        row['timing']=timing
                    finally:kernel.free(model['handle'])
                    rows.append(row);np.savez_compressed(out/f'{tag}.npz',**payload);save(out,'summary.json',rows)
                    print(tag,row.get('click_tv',row.get('pnr_tv',row.get('hardware_ks'))),row['setup_s'],timing[-1]['median'],flush=True)
        Optical.law.cache_clear();Optical.local_entropy.cache_clear();Optical.photon_split.cache_clear()
    manifest(out)
if __name__=='__main__':
    with threadpool_limits(limits=1,user_api='blas'):main()
