"""All-port photon-number laws, optical edge validation and independent recurrence checks."""
import argparse
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration100_baselines_v1 import pair_reference
from iteration106_field_v1 import DATA,actual_input,optical
from iteration106_packed_v1 import compress
from iteration106_sample_v1 import benchmark,panel_hist
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration109_model_v1 import build,panel
from iteration109_sample_v1 import tv,Tree
from iteration109_tree_v1 import prepare,minors

def total(eig):
    n=4096;z=np.exp(-2j*np.pi*np.arange(n//2+1)/n)
    p=np.fft.irfft(np.exp(-np.log1p((1-z[:,None])*eig).sum(1)/2),n=n)
    # Positive PGF beyond the unit circle bounds periodic aliasing.
    s=1+.6/max(1.,eig.max());tail=float(np.exp(-.5*np.log1p((1-s)*eig).sum()-n*np.log(s)))
    assert tail<1e-12 and p.min()>-1e-13
    return p,tail

def check_recurrence(params):
    errors=[];vacuum_errors=[];rng=np.random.default_rng(109440000)
    for s,g in ((0,0),(1,0),(0,34),(1,107)):
        pars=params['meta'][s,g];inv=pars[:16].reshape(4,4);c=np.linalg.inv(inv)-np.eye(4);d=rng.normal(0,.6,4);shift=inv@d
        eta=rng.uniform(0,1,2);Z=np.diag(eta.repeat(2));mat=np.eye(4)+Z@c
        p_true=np.exp(-d@np.linalg.solve(mat,Z@d))/np.sqrt(np.linalg.det(mat))
        pol,num=minors(c);basis=np.outer(eta[0]**np.arange(3),eta[1]**np.arange(3))
        det=(basis*pol).sum();p=np.exp(-np.einsum('ab,abij,i,j',basis,num,d,d)/det)/np.sqrt(det)
        vacuum_errors.append(float(abs(p-p_true)))
        # Independent complex FFT of the displaced Gaussian generating function.
        # A tilted contour suppresses periodic aliases in the low counts.
        z=.70*np.exp(-2j*np.pi*np.arange(64)/64)
        u,v=np.meshgrid(1-z,1-z,indexing='ij');zz=np.stack((u,u,v,v),axis=-1)
        m=np.eye(4)+zz[...,None]*c;rhs=(zz*d)[...,None]
        pgf=np.exp(-(np.linalg.solve(m,rhs)[...,0]*d).sum(2))/np.sqrt(np.linalg.det(m))
        reference=np.fft.ifft2(pgf).real[:9,:9]/.7**np.add.outer(np.arange(9),np.arange(9))
        co=params['photon'][s,g];prod=np.array([shift[i]*shift[j] for i in range(4) for j in range(i,4)])
        S=co[...,0];R=co[...,1]+co[...,2:]@prod;law=np.zeros((9,9));law[0,0]=np.exp(-pars[16]-d@shift)
        for a,b in np.ndindex(law.shape):
            if not a+b:continue
            law[a,b]=sum(law[a-i,b-j]*(R[i,j]-(a+b-i-j)*S[i,j]) for i in range(min(a,4)+1) for j in range(min(b,4)+1) if i+j)/(a+b)
        errors.append(float(abs(reference-law).max()))
    assert max(errors)<1e-11 and max(vacuum_errors)<1e-13
    return dict(pnr_max_cell_error=max(errors),vacuum_max_error=max(vacuum_errors),pnr_cells=4*81)

def run(out):
    out.mkdir(parents=True,exist_ok=False);t,r=actual_input();ref=oracle()
    law,tail=total(optical(t,r));arrays=dict(optical_total=law);scans=[]
    for alpha in (0,.25,.5,.75,1.):
        model=build(t,r,8,alpha);single=dict(model,fan=1,weights=np.ones((len(t)//8,1)))
        xx=panel(single,np.arange(len(t)//8));eig=np.linalg.eigvalsh(xx).ravel()
        p,alias=total(eig);arrays[f'model_total_{alpha}']=p
        scans.append(dict(alpha=alpha,total_photon_tv=tv(p,law),total_photon_ks=float(abs((p-law).cumsum()).max()),alias_bound=alias,mean=eig.sum()/2))
    required=((0,1),(36,107),(34,35))
    params=prepare(t,r,required=required);tree=Tree(params,DATA/'iteration109_tree_v1/tree.dll')
    _,_,proj=compress(t,(np.sinh(r)**2).sum(0),1)
    # Check one independent four-detector marginal on every edge. Enumerate
    # all sixteen detectors on the three frozen high-priority pairs as well.
    small=[];edges=[]
    for child,parent in enumerate(params['parent']):
        if parent<0:continue
        edges.append([int(parent),child]);ids=np.r_[8*parent,8*parent+5,8*child+1,8*child+7]
        p=gaussian_click(ref,optical_x(proj[ids],r),ONE);q=gaussian_click(ref,optical_x(t[ids],r),ONE)
        small.append(tv(p,q))
    selected=[]
    for idx,key in enumerate((0,1,5)):
        frozen=np.load(DATA/f'iteration106_packed_v1/panel_{key}.npz');ids=frozen['ports'];truth=frozen['target'];x=optical_x(proj[ids],r)
        p=gaussian_click(ref,x,ONE);sel=build(t,r,8,1)
        independent=gaussian_click(ref,panel(sel,ids),ONE)
        selected.append(dict(ports=ids.tolist(),edge_tv=tv(p,truth),local_tv=tv(independent,truth)))
        arrays[f'edge{idx}']=p;arrays[f'edge_target{idx}']=truth
    counts=pair_reference(np.eye(4)+optical_x(proj[[289,861]],r),12)
    truth=np.load(DATA/'iteration107_projection_v1/pnr.npz')['target']
    measurements=[];shots=8192
    for pn in (False,True):
        batch=256;tm=benchmark(lambda n,s,o:tree.draw(n,pn,s,o),batch,109445000+pn,7)
        h=np.zeros(169,int);hh=np.zeros((3,65536),np.int64)
        for off in range(0,shots,batch):
            v=tree.draw(batch,pn,109446000+pn,off)
            if pn:h+=np.bincount(v[:,[289,861]].clip(0,12)@np.array([1,13]),minlength=169)
            for j,sel in enumerate(selected):hh[j]+=panel_hist(v,sel['ports'])
        measurements.append(dict(route='pnr' if pn else 'click',timing=tm,shots=shots))
        arrays[f'prioritized_hist{int(pn)}']=hh
        if pn:arrays['prioritized_count_hist']=h
    # Pure twin beam has perfect equal-number and equal-click support even
    # when photons bunch. This exercises the conditional step beyond vacuum.
    zz=prepare(np.eye(2,dtype=complex),np.array([[.87]]),1,share=0)
    tw=Tree(zz,DATA/'iteration109_tree_v1/tree.dll');controls=[]
    for pn in (False,True):
        a=tw.draw(32768,pn,109447000+pn);assert np.all(a[:,0]==a[:,1])
        if pn:assert a.max()>6
        controls.append(dict(pnr=pn,shots=len(a),unequal=int(np.count_nonzero(a[:,0]!=a[:,1])),max_readout=int(a.max())))
    arrays.update(count_target=truth,count_model=counts,edge_four_tv=small)
    np.savez_compressed(out/'global.npz',**arrays)
    save(out,'summary.json',dict(total_photon=scans,optical_alias=tail,priority=required,edges=edges,
         four_bit_edge_median_tv=float(np.median(small)),four_bit_edge_max_tv=max(small),edge_sixteen=selected,
         two_port_pnr_tv=tv(counts,truth),focused=measurements,controls=controls,recurrence=check_recurrence(params)))
    manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
