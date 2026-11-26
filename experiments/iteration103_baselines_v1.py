"""Same-input same-observed-width Frugal and photonic-engines baselines on the corrected sites."""
import argparse
import ctypes as ct
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_baselines_v1 import frugal,pair_reference,physical,purification,pe_sample
from iteration100_projection_v1 import Projector
from iteration100_spatial_v1 import ORDER,Tree,Native,save,manifest
from iteration102_sample_v1 import Sampler as Pair
from iteration103_sample_v1 import DATA,BASE,ENGINE


class Projected(Projector):
    def response(self,groups,feature):
        a,b=sorted(groups)
        left,ids=feature(a)
        right,more=feature(b)
        return (right.T@self.pair.T@left).ravel(),ids+more


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    native=Native(BASE/'spatial.dll')
    lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    proj=Projected(Tree(native,dict(np.load(BASE/'prepared.npz'))))
    spec=dict(np.load(ORDER))
    click=[]
    pnr=[]
    for i,(a,b) in enumerate(((36,107),(25,139),(34,35),(0,1))):
        z=np.load(DATA/f'iteration103_projection_v1/pair_{a}_{b}.npz')
        corrected='patched' in z
        joint=z['patched'] if corrected else z['raw'].clip(0)
        joint=joint/joint.sum()
        t=perf_counter()
        model=Pair(ENGINE/'forest.dll',[a,b],joint)
        prep=perf_counter()-t
        q=physical(spec,z['ports'])
        truth=z['truth']
        bits=(proj.mask[b].T@joint.T@proj.mask[a]).ravel()
        med=[]
        for j in range(8):
            t=perf_counter()
            model.sample(2048,False,False,1,103090000+i,2048*j)
            if j:med.append((perf_counter()-t)*1e6/2048)
        row=dict(groups=[a,b],patched=corrected,us=float(np.median(med)),timing_us=med,load_s=prep,complete_tv=float(abs(bits-truth).sum()/2),frugal=[])
        for k in (2,4):
            law,times=frugal(lib,q,k,103091000+100*i+20*k)
            row['frugal'].append(dict(K=k,complete_tv=float(abs(law-truth).sum()/2),us=float(np.median(times)),timing_us=times))
        click.append(row)
        print('click',a,b,row,flush=True)
        if i:continue
        ids=[int(np.argmax(proj.model.ports[g]))+8*g for g in (a,b)]
        small=physical(spec,ids)
        ref=pair_reference(small,12)
        proj.pair=joint
        law=proj.photons(ids,12)
        tail=float(ref.reshape(13,13)[-1].sum()+ref.reshape(13,13)[:-1,-1].sum())
        states=[purification(s) for s in small]
        def peer(n,seed):
            total=np.zeros((n,2),np.uint32)
            for h,(state,modes,_) in enumerate(states):
                x,st=pe_sample(state,modes,n,seed=seed+h,device='cpu',return_stats=True,impl='v2',window=4,max_photons=64)
                assert not np.any(st.censored)
                total+=np.asarray(x[:,:2],np.uint32)
            return total
        indices=[ids[0]%8,8+ids[1]%8]
        def ours(n,off):
            return model.sample(n,True,False,1,103092000,off)[0][:,indices].copy()
        ts,ps=[],[]
        for j in range(6):
            t=perf_counter()
            ours(2048,2048*j)
            if j:ts.append((perf_counter()-t)*1e6/2048)
            t=perf_counter()
            peer(2048,103093000+2*j)
            if j:ps.append((perf_counter()-t)*1e6/2048)
        x=np.concatenate([peer(2048,103094000+2*j) for j in range(16)])
        y=ours(len(x),65536)
        def hist(v):return np.bincount(v.clip(0,12)@np.array([1,13]),minlength=169)/len(v)
        v=float(abs(law-ref).sum()/2)
        pnr.append(dict(ports=ids,records=len(x),model_full_tv=[v,v+tail],model_empirical_tv=float(abs(hist(y)-ref).sum()/2),pe_empirical_tv=float(abs(hist(x)-ref).sum()/2),model_us=float(np.median(ts)),pe_us=float(np.median(ps)),model_timing=ts,pe_timing=ps,pe_censored=0,purification_error=max(c for _,_,c in states)))
        np.savez_compressed(out/'pnr.npz',reference=ref,model=law,observed=hist(y),pe=hist(x))
    save(out,'summary.json',dict(click=click,pnr=pnr,scope='Projected two-group CPU sampler, 2048 shots, one thread. Frugal includes target construction twice per batch. PE timing excludes purification.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
