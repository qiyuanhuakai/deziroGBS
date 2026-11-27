"""Complete 16-channel click law and a full-PNR bracket for the same quartic sampler."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits

from iteration106_field_v1 import actual_input
from iteration115_panel_accuracy_v2 import bank_counts,dark_projection,tail
from iteration117_quartic_laws import physical_models,marginal,contraction,void
from iteration117_quartic import OUT
from iteration115_visible_v1 import DATA


def word_law(mods,ports):
    subsets=np.arange(1<<16)
    values=np.ones(len(subsets),np.longdouble)
    for model in mods:
        S,norm,t,x=marginal(model,ports)
        for mask in range(1,len(values)):
            sites=np.flatnonzero(mask & (1<<np.arange(16)))
            ids=np.r_[sites,16+sites]
            den=np.eye(len(ids))+x[np.ix_(ids,ids)]
            z=contraction(S,norm,t[sites],np.linalg.inv(den),np.exp(-.5*np.linalg.slogdet(den)[1]))
            assert abs(z.imag)<1e-11
            values[mask]*=z.real
    law=values[::-1].copy()
    for j in range(16):
        block=law.reshape(-1,2,1<<j);block[:,1]-=block[:,0]
    assert abs(law.sum()-1)<1e-12 and np.minimum(law,0).sum()>-1e-8
    return np.array(law,float)


def bank_grid(mods,ports,rotation,projected,grid):
    # Rotation is block-unitary within each bank. The first row selects its bright mode.
    z=np.exp(2j*np.pi*np.arange(grid)/grid)
    all_weights=1-np.stack(np.broadcast_arrays(z[:,None],z[None,:]),-1)
    axis=np.repeat([0,1],8)
    joint=np.ones((grid,grid),complex)
    for model in mods:
        q,S,noise,norm=model
        t=rotation@q[ports]
        r=np.block([[t.real,-t.imag],[t.imag,t.real]])
        x=r@noise@r.T
        sign=np.empty_like(joint);raw=np.empty_like(joint)
        for i in range(grid):
            for j in range(grid):
                w=all_weights[i,j,axis].copy()
                if projected:w[np.arange(16)%8!=0]=1.
                w=np.tile(w,2)
                den=np.eye(len(w))+w[:,None]*x
                phase,lg=np.linalg.slogdet(den);sign[i,j]=phase
                metric=np.linalg.solve(den,np.diag(w))
                raw[i,j]=contraction(S,norm,t,metric,np.exp(-.5*lg))
        ph=np.unwrap(np.unwrap(np.angle(sign),axis=0),axis=1)
        assert max(abs(np.diff(ph,axis=k)).max() for k in (0,1))<1.5
        joint*=raw*np.exp(-.5j*ph)
    vacuum=joint[0,0].real
    coeff=np.fft.fft2(joint/vacuum)/grid**2
    assert abs(coeff.imag).max()<1e-12 and coeff.real.min()>-2e-12
    return coeff.real,vacuum


def chernoff(mods,ports,radius):
    g=1.
    for model in mods:
        S,norm,t,x=marginal(model,ports)
        e,u=np.linalg.eigh(x);w=1-radius
        assert min(1+w*e)>0
        g*=contraction(S,norm,t,(u*(w/(1+w*e)))@u.T,np.exp(-.5*np.log1p(w*e).sum()))
    return float(g.real/radius**128)


def run(pair,transport=False):
    mods=physical_models(transport=transport);raw,r=actual_input()
    ports=(8*np.array(pair)[:,None]+np.arange(8)).ravel()
    tr=raw[ports].reshape(2,8,-1)
    u=np.linalg.svd(tr,full_matrices=False)[0].conj().transpose(0,2,1)
    rot=np.zeros((16,16),complex);rot[:8,:8],rot[8:,8:]=u
    tag="_".join(map(str,pair))
    word=word_law(mods,ports)
    exact=np.load(DATA/"iteration115_review_v3/projection_panels.npz")["truth_"+tag]
    click=float(abs(word-exact).sum()/2)
    if pair==[34,35]:
        target=np.load(DATA/"iteration115_panel_accuracy_v2/pnr_laws.npz")
        old=json.loads((DATA/"iteration115_panel_accuracy_v2/pnr_bounds.json").read_text())
        real,proj=target["optical_bank"],target["projected_bank"]
        leak,ptail=old["dark_nonvacuum"],old["projected_tail"]
    else:
        from iteration107_cliques_v1 import optical_x
        real=bank_counts(optical_x(raw[ports],r),[*range(8),*range(16,24)])
        comp,leak,_=dark_projection(tr,r)
        proj,ptail=bank_counts(comp,[0,1]),tail(comp)
    q,v=bank_grid(mods,ports,rot,False,128)
    assert abs(v-1)<2e-12
    qp,keep=bank_grid(mods,ports,rot,True,128)
    # The bank-PNR Chernoff tail also bounds joint bright-count tail with vacuum dark.
    ub=min(chernoff(mods,ports,R) for R in (1.15,1.2,1.25))
    lo=float(abs(real-q).sum()/2)
    hi=float(abs(proj-qp).sum()/2+np.sqrt(leak)+np.sqrt(max(0,1-keep))+ptail+ub/keep)
    # The all-vacuum event is identical for PNR and threshold up to modulo aliasing.
    assert abs(q[0,0]-word[0])<ub+5e-13
    prefix="transport_" if transport else ""
    result=dict(banks=pair,click_tv=click,full_pnr_lower=lo,full_pnr_upper=hi,
                candidate_dark=float(1-keep),optical_dark=float(leak),candidate_alias=ub,
                click_negative_mass=float(-np.minimum(word,0).sum()))
    np.savez_compressed(OUT/f"{prefix}panel_{tag}.npz",click=word,pnr=q,projected_pnr=qp)
    (OUT/f"{prefix}panel_{tag}.json").write_text(json.dumps(result,indent=2)+"\n",encoding="utf8")
    print(result,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("banks",type=int,nargs=2)
    p.add_argument("--transport",action="store_true");opt=p.parse_args()
    with threadpool_limits(limits=1):run(opt.banks,opt.transport)
