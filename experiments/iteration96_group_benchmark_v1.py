"""Audit full small-system laws and time the native tree on physical port records."""
import argparse
import hashlib
import math
import sys
from pathlib import Path
from time import perf_counter
from threadpoolctl import threadpool_limits
from iteration96_group_common_v1 import (ROOT,H,save,manifest,Native,Optical,partition,
    construct,pair_information,block_weights,joint_values,gaussian_entropy,entropy,np)

DATA=ROOT/'results'
ACTUAL=DATA/'iteration94_actual_quality_v1'
ORDER=DATA/'iteration83_loss_pilot_v1/input.npz'
HIST=DATA/'iteration81_jiuzhang_accuracy_v3/histograms.npz'
BUILD=DATA/'iteration96_tree_build_v1/tree.dll'
def moments(u,t,alpha,noise):
    m,n=u.shape
    transfer=np.block([[u.real,-u.imag],[u.imag,u.real]])
    core=2*noise@noise.T+np.diag(np.r_[t/(1-t),-t/(1+t)])
    return np.eye(2*m)+transfer@core@transfer.T,transfer@(np.sqrt(2)*np.r_[alpha.real,alpha.imag])

def fixture(name):
    if name.startswith('actual'):
        perm=np.argsort(np.load(ORDER)['order'])
        ports=np.arange(16)*8 if name=='actual16' else np.arange(1152)
        cov=[];means=[];global_entropy=0.
        for j in range(2):
            z=np.load(ACTUAL/f'input_{j}.npz')
            u=z['u'][perm][ports];args=[z[k] for k in ('t','alpha','noise')]
            q,mu=moments(u,*args);cov.append(q);means.append(mu)
            if len(ports)>100:
                reduced=np.linalg.qr(u,mode='r')
                global_entropy+=gaussian_entropy(moments(reduced,*args)[0][None])
            else:global_entropy+=gaussian_entropy(q[None])
        return np.array(cov),np.array(means),None,global_entropy
    if name=='pure4':
        z=np.load(DATA/'iteration93_reference_v2/input_0.npz')
        u,t,alpha,noise=[z[k] for k in ('u','t','alpha','noise')]
    else:
        d=8;u=np.zeros((16,16),complex)
        for j in range(d):
            u[j,2*j:2*j+2]=[1,1];u[d+j,2*j:2*j+2]=[1,-1]
        u/=np.sqrt(2)
        angle={'twins':0.,'weak':.025,'mixed':.30}[name]
        t=np.tile([1.,-1.],d)*(.5 if name=='twins' else .24)
        for step in range(2):
            for a in range(step,d,2):
                b=(a+1)%d
                phase=np.exp(1j*(a+1)*.317)
                u[[a,b]]=np.array([[np.cos(angle),np.sin(angle)*phase],
                    [-np.sin(angle)*phase.conjugate(),np.cos(angle)]])@u[[a,b]]
        alpha=np.zeros(2*d,complex);noise=np.zeros((4*d,4*d))
    q,mu=moments(u,t,alpha,noise)
    assert np.max(np.abs(u.conj().T@u-np.eye(len(u))))<1e-12 and not np.any(alpha) and not np.any(noise)
    return q[None],mu[None],(u*t)@u.T,0.

def draw(lib,ptr,m,n,domain,offset,workers):
    x=np.empty((n,m),np.uint8);tries=np.empty(n,np.uint32);tail=np.empty(n,np.uint8);excess=np.empty(n,np.uint8)
    assert lib.draw(ptr,n,domain,offset,0,workers,x,tries,tail,excess)==0
    assert np.all(tries==1) and not np.any(excess)
    return x,tail

def fock(lib,barg,q,cutoff):
    m=len(barg);size=sum(math.comb(m+n-1,n) for n in range(0,cutoff+1,2))
    keys=np.empty(size,np.uint64);p=np.empty(size)
    p0=np.exp(-.5*np.linalg.slogdet(q[0])[1])
    assert lib.fock(m,cutoff,size,np.ascontiguousarray(barg).view(float),p0,keys,p)==size
    states=((keys[:,None]>>(4*np.arange(m,dtype=np.uint64)))&15).astype(np.uint8)
    return states,p

def projected(table,base,ids):
    nodes=table['parents'];tr=table['transition'];k=tr.shape[-1]
    values=np.ones((len(nodes),k,1<<len(ids)))
    codes=np.arange(k);masks=np.arange(1<<len(ids))
    for i,ports in enumerate(table['ports']):
        for j,port in enumerate(ports):
            if port in ids:
                bit=ids.index(port)
                values[i]*=(((codes//base**j)%base>0)[:,None]==((masks>>bit)&1)[None,:])
    for i in reversed(range(1,len(nodes))):
        values[nodes[i]]*=tr[i]@values[i]
    law=tr[0,0]@values[0];assert abs(law.sum()-1)<2e-10
    return law

def run_case(out,lib,name,caseid):
    then=perf_counter();q,mu,barg,hglobal=fixture(name);m=q.shape[1]//2
    opt=Optical(lib,q,mu);info,_=pair_information(q);input_seconds=perf_counter()-then
    large=m>16
    if not large:
        states=((np.arange(1<<m)[:,None]>>np.arange(m))&1).astype(np.uint8)
        truth=opt.law(tuple(range(m)),2)
        vals={'q':q,'mu':mu,'click':truth}
        if barg is not None:
            vals['barg']=barg
            ns,fp=fock(lib,barg,q,12 if m==4 else 8)
            labels=(ns>0)@(2**np.arange(m))
            partial=np.bincount(labels,weights=fp,minlength=len(truth))
            assert (partial-truth).max()<2e-10
        np.savez_compressed(out/f'{name}_reference.npz',**vals)
    projections=np.load(HIST)['ports'].tolist() if large else [list(range(min(8,m)))]
    configs=[(2,2,False,False),(2,2,True,False),(2,2,True,True),(2,4,True,True),
             (3,2,False,False),(3,2,True,False),(3,2,True,True)]
    if large:configs=[(2,4,True,True),(3,2,True,True)]
    rows=[]
    for j,(base,width,auto,tree) in enumerate(configs):
        start=perf_counter()
        groups=partition(m,width,auto,info)
        cap=32 if base==3 else width
        table,diag=construct(opt,groups,block_weights(groups,info),base,cap,tree)
        k=table['transition'].shape[-1];limit=(384 if large else 96) if base==3 else min(m,512)
        proposal=np.empty(limit+1)
        ptr=lib.prepare(m,len(groups),k,cap,int(base==3),limit,*table.values(),proposal)
        assert ptr and abs(proposal.sum()-1)<1e-9
        prep=perf_counter()-start;exp=96001000+100*caseid+j
        row=dict(case=name,m=m,base=base,width=width,adaptive=auto,tree=tree,setup_s=prep,
                 input_and_pair_s=input_seconds,global_entropy=hglobal,**diag)
        payload=dict(ports=table['ports'],parents=table['parents'],proposal=proposal,transition=table['transition'])
        try:
            for v,ids in enumerate(projections):
                p=projected(table,base,ids);target=opt.law(tuple(ids),2)
                row.setdefault('projected_click_tv',[]).append(float(abs(p-target).sum()/2))
                payload[f'projection_{v}']=p
            if not large:
                if base==2:
                    model=joint_values(table,2,states);assert abs(model.sum()-1)<1e-10
                    row['full_click_tv']=float(abs(model-truth).sum()/2)
                    row['kl']=max(0.,diag['block_classical']-diag['tree_mi']-entropy(truth))
                    row['pinsker']=min(1.,np.sqrt(row['kl']/2))
                    assert row['full_click_tv']<=row['pinsker']+1e-7
                    count=np.bincount(states.sum(1).astype(int),weights=model,minlength=m+1)
                    assert np.max(abs(count-proposal))<1e-10
                    payload['law']=model
                elif barg is not None:
                    model=joint_values(table,3,ns)
                    tail=max(0.,1-fp.sum());high=float(1-np.minimum(fp,model).sum())
                    low=max(0.,high-min(tail,max(0.,1-model.sum())))
                    row['full_pnr_tv']=[low,max(low,high)]
                    row['fock_states']=len(fp);row['fock_tail']=tail
            quantum=max(0.,diag['block_entropy']-hglobal-diag['tree_mi'])
            row['quantum_pinsker_with_tail']=min(1.,np.sqrt(quantum/2)+diag['tail_union'])
            n=8192 if large else 2048
            x,_=draw(lib,ptr,m,64,exp,0,1)
            replay,_=draw(lib,ptr,m,48,exp,16,4)
            assert np.array_equal(x[16:],replay)
            times=[];totals=np.zeros(m+1,int);alltail=0;hashes=[]
            for workers in ([1,16] if large else [1]):
                stamps=[]
                for rep in range(5):
                    start=perf_counter()
                    x,tail=draw(lib,ptr,m,n,exp,128+(workers*5+rep)*n,workers)
                    stamps.append((perf_counter()-start)*1e6/n)
                    totals+=np.bincount((x>0).sum(1),minlength=m+1)
                    alltail+=int(tail.sum())
                    hashes.append(hashlib.sha256(x).hexdigest())
                times.append(dict(threads=workers,us=stamps,median=float(np.median(stamps)),batch=n))
            row.update(timing=times,tail_flags=alltail,record_hashes=hashes)
            payload['counts']=totals;payload['first64']=draw(lib,ptr,m,64,exp,0,1)[0]
            if large:
                hw=np.load(HIST)['total_click_histogram'];hw=hw/hw.sum()
                cdf=np.cumsum(totals/totals.sum()) if base==3 else np.cumsum(np.pad(proposal,(0,len(hw)-len(proposal))))
                row['hardware_count_ks']=float(abs(cdf-np.cumsum(hw)).max())
                if base==3:
                    ref=np.load(DATA/'iteration84_photon_cdf_v1/reference.npz')['photon_pmf']
                    row['total_photon_ks']=float(abs(proposal.cumsum()-ref[:len(proposal)].cumsum()).max())
                row['hardware_n']=int(np.load(HIST)['total_click_histogram'].sum())
        finally:lib.free(ptr)
        np.savez_compressed(out/f'{name}_{j}.npz',**payload)
        rows.append(row)
        print(name,base,width,auto,tree,row.get('full_click_tv',row.get('full_pnr_tv')),flush=True)
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('cases',nargs='+')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    files=[Path(__file__),ROOT/'experiments/iteration96_group_common_v1.py',BUILD,HIST,ORDER,
           *(ACTUAL/f'input_{j}.npz' for j in range(2)),DATA/'iteration93_reference_v2/input_0.npz',
           DATA/'iteration84_photon_cdf_v1/reference.npz']
    save(out,'plan.json',dict(cases=args.cases,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},
        cutoffs=[8,12],cap=32,angles=[0,.025,.30],workers=[1,16],replicates=5,
        scope='raw trees; independent integer-word population laws; optical parameters only'))
    lib=Native(BUILD);rows=[]
    ids=['twins','weak','mixed','actual16','pure4','actual1152']
    with threadpool_limits(limits=1,user_api='blas'):
        for name in args.cases:
            rows.extend(run_case(out,lib,name,ids.index(name)))
            save(out,'summary.json',rows)
            Optical.law.cache_clear();Optical.local_entropy.cache_clear();Optical.photon_split.cache_clear()
    manifest(out)
if __name__=='__main__':main()
