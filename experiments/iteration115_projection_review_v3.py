"""Full-circuit bright-mode projection and conditional spatial tests at fixed global N."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from iteration115_visible_v1 import DATA, hardware
from iteration111_memory_v1 import realify
from iteration115_panel_v1 import panel_law
from iteration115_panel_accuracy_v2 import bank_counts, tv, dark_projection
from iteration106_field_v1 import actual_input
from iteration107_fast_v1 import gaussian_click,oracle,ONE
from iteration107_cliques_v1 import optical_x

OUT=DATA/'iteration115_review_v3'
TEST_N=np.array([80,100,116,140,160])
PANELS=[(34,35),(0,1),(17,89),(45,98),(71,72),(102,143)]


def factor(t, ss):
    """Return orthonormal real axes and spectrum of X for both spectra."""
    q,r=np.linalg.qr(t,mode='reduced')
    real_q,real_r=realify(q),realify(r)
    ans=[]
    for s in ss:
        d=np.expm1(np.r_[-2*s,2*s])/2
        e,u=eigh((real_r*d)@real_r.T)
        ans.append((real_q@u,e))
    return ans


def project(t,ss,width):
    banks=t.reshape(144,8,50)
    u=np.linalg.svd(banks,full_matrices=False)[0][:,:,:width]
    bright=(u.conj().transpose(0,2,1)@banks)
    dark=(banks-u@bright).reshape(1152,-1)
    gram=realify(dark.conj().T@dark)
    qb,rb=np.linalg.qr(bright.reshape(-1,50),mode='reduced')
    basis=realify(qb)
    rb=realify(rb)
    logp=0.; models=[]
    for s in ss:
        diag=np.expm1(np.r_[-2*s,2*s])/2
        mat=np.eye(len(diag))+diag[:,None]*gram
        sign,ld=np.linalg.slogdet(mat)
        assert sign>0
        logp-=ld/2
        effective=np.linalg.solve(mat,np.diag(diag))
        x=rb@effective@rb.T
        e,v=eigh((x+x.T)/2)
        models.append((basis@v,e))
    return u,models,-np.expm1(logp)


def sector_vacuum(models,sites):
    # Coefficients of G(z) times the tilted vacuum probability on 'sites'.
    # The ratio of the two coefficients is conditional on the total for the entire circuit.
    grid=512
    z=np.exp(-2j*np.pi*np.arange(grid//2+1)/grid)
    total=np.ones(len(z),complex);joint=np.ones_like(total)
    for u,e in models:
        den=1+(1-z[:,None])*e
        total*=np.exp(-.5*np.log(den).sum(1))
        b=u[sites]
        # Evaluate the vacuum by a small determinant. Unwrap phase from z=1.
        x=(b[None,:,:]*(z[:,None]*e/den)[:,None,:])@b.T
        phase,logdet=np.linalg.slogdet(np.eye(len(b))+x)
        angle=np.unwrap(np.angle(phase))
        joint*=np.exp(-.5*(logdet+1j*angle))
    p=np.fft.irfft(total,n=grid);joint_p=np.fft.irfft(joint*total,n=grid)
    assert p[TEST_N].min()>1e-4 and np.all(joint_p[TEST_N]<=p[TEST_N]+1e-12)
    # Chernoff bound on the discarded total-N aliases.
    log_tail=[]
    for r in np.linspace(1.002,1.3,150):
        ds=[1+(1-r)*e for _,e in models]
        if min(d.min() for d in ds)>0:log_tail.append(-grid*np.log(r)-sum(np.log(d).sum() for d in ds)/2)
    tail=np.exp(min(log_tail))
    return joint_p[TEST_N]/p[TEST_N],float(tail)


def run():
    OUT.mkdir(exist_ok=True)
    t,ss=hardware(); raw,r=actual_input()
    full=factor(t,ss); widths={}
    for width in [1,2,4,8]:
        u,model,eps=project(t,ss,width)
        widths[width]=dict(modes_per_spectrum=144*width,dark_probability=float(eps),
                           state_distance_upper=float(np.sqrt(max(0,eps))))
        if width==1: first_u,first_model=u,model
    split=abs(first_u[:,:,0])**2;idx=np.column_stack((np.arange(144),144+np.arange(144))).ravel()
    mats=np.array([((u*e)@u.T)[np.ix_(idx,idx)] for u,e in first_model])
    compact=dict(weights=[1.],marginal=[mats],split=split.cumsum(1))
    old=np.load(DATA/'iteration115_tilt_v1/model.npz')
    # Mixture and its base have exactly the same conditional PNR law at fixed global N.
    from iteration115_banks_v1 import bank_model
    args,_,_=bank_model(t,ss,8)
    a,e,_,_=args;old_factors=[]
    for j in range(2):
        f=a.reshape(len(a),144,4)[:,:,2*j:2*j+2].reshape(len(a),-1)
        vals,axes=eigh(2*f.T@f+np.diag(e[:,2*j:2*j+2].ravel()))
        old_factors.append((axes,vals))
    panelstats=[]; arrays={}; full_oracle=oracle(True)
    for ia,ib in PANELS:
        panels=[ia,ib]
        ports=(8*np.array(panels)[:,None]+np.arange(8)).ravel()
        optical=gaussian_click(full_oracle,optical_x(raw[ports],r),ONE)
        bright=panel_law(compact,ia,ib)
        prev=panel_law(old,ia,ib)
        ids=(2*np.array(panels)[:,None]+np.arange(2)).ravel()
        projected=bank_counts(mats[:,ids][:,:,ids],[0,1])
        x=optical_x(raw[ports],r)
        target=bank_counts(x,[*range(8),*range(16,24)])
        previous=sum(w*bank_counts(c[:,ids][:,:,ids],[0,1]) for w,c in zip(old['weights'],old['marginal']))
        # Independent, local dark projection supplies a much tighter panel upper bound.
        block,local_leak,_=dark_projection(raw[ports].reshape(2,8,50),r)
        local=bank_counts(block,[0,1])
        from iteration115_panel_accuracy_v2 import tail
        pnl=tv(projected,target)
        up=tv(projected,local)+np.sqrt(local_leak)+tail(block)+tail(mats[:,ids][:,:,ids])
        assert pnl<=up+1e-12
        site=np.r_[ports,ports+1152]
        ref,err=sector_vacuum(full,site)
        new,err2=sector_vacuum(first_model,np.r_[panels,np.array(panels)+144])
        base,err3=sector_vacuum(old_factors,ids)
        panelstats.append(dict(banks=panels,old_word_tv=tv(prev,optical),projected_word_tv=tv(bright,optical),
            old_bank_pnr_tv=tv(previous,target),projected_bank_pnr_tv=pnl,panel_pnr_upper=float(up),
            optical_fixed_N_vacuum=ref.tolist(),base_fixed_N_vacuum=base.tolist(),projected_fixed_N_vacuum=new.tolist(),
            global_N_alias_tail=max(err,err2,err3)))
        arrays[f'truth_{ia}_{ib}']=optical
        arrays[f'projected_{ia}_{ib}']=bright
        print(panels, panelstats[-1]['projected_word_tv'],max(abs(ref-new)),flush=True)
    np.savez_compressed(OUT/'projection_panels.npz',**arrays)
    info=dict(widths=widths,total_photons=TEST_N.tolist(),panels=panelstats)
    (OUT/'projection.json').write_text(json.dumps(info,indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=3):run()
