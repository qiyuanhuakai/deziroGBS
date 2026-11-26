"""Fit recordwise rejection rules and a numerical CDF lower bound for the optical-input/hardware gap."""
import argparse
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import least_squares,linprog,minimize
from scipy.sparse import coo_matrix
from scipy.special import expit
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import HIST,save,manifest
from iteration104_optical_v1 import OPT
from iteration104_sample_v1 import PILOT,MAX


def optical_weights(hist,ref,first,second,cap):
    # Fit on the discrete row law in collision coordinates, where adjacent
    # photon totals have similar conditional moments.
    m=len(hist)
    mass=hist.sum(1)
    collision=np.zeros_like(hist)
    for n in range(m):collision[n,:n+1]=hist[n,:n+1][::-1]
    smooth=gaussian_filter1d(collision,.9,axis=0)
    features=np.zeros((m,3))
    good=np.flatnonzero((mass>2e-5)&(ref>2e-5))
    for n in good:
        if not 0<n<m-1:continue
        q=smooth[n,:n+1].copy()
        q/=q.sum()
        mean=first[n]/ref[n]
        var=(first[n]+second[n])/ref[n]-mean**2
        t=(n-np.arange(n+1)-mean)/np.sqrt(max(var,.05))
        f=np.array([np.ones_like(t),t,t*t]).T
        r=ref[n]/(.5*mass[n]+.5*gaussian_filter1d(mass,1.)[n])
        target=np.array([r,0.,r])
        def calc(theta):
            u=expit(np.log(1/(cap-1))+f@theta)
            return cap*u
        def residual(theta):return (q*calc(theta))@f-target
        def jac(theta):
            u=calc(theta)
            return (f.T*(q*u*(1-u/cap)))@f
        features[n]=least_squares(residual,np.zeros(3),jac=jac,bounds=(-12,12),max_nfev=45).x
    th=np.array([np.interp(np.arange(m),good,features[good,j]) for j in range(3)]).T
    th=gaussian_filter1d(th,.9,axis=0)
    w=np.empty_like(hist)
    for n in range(m):
        near=np.clip(n,good.min(),good.max())
        mu=first[near]/ref[near]+(n-near)*.8
        var=max(.05,(second[near]+first[near])/ref[near]-(first[near]/ref[near])**2)
        t=(np.arange(m)-mu)/np.sqrt(var)
        w[n]=cap*expit(np.log(1/(cap-1))+th[n,0]+th[n,1]*t+th[n,2]*t*t)
    return w


def hardware_weights(hist,ph,hw,cap):
    # Bounded logistic density ratio, fitted to two margins with a convex dual.
    n=len(hw)
    ii,jj=np.nonzero(hist)
    q=hist[ii,jj]
    prior=-np.log(cap-1)
    def objective(v):
        s=prior+v[ii]+v[n+jj]
        weight=cap*expit(s)
        cost=cap*(q@np.logaddexp(0.,s))-ph@v[:n]-hw@v[n:]
        grad=np.r_[np.bincount(ii,weights=q*weight,minlength=n)-ph,
                   np.bincount(jj,weights=q*weight,minlength=n)-hw]
        return cost+1e-8*(v@v),grad+2e-8*v
    fit=minimize(objective,np.zeros(2*n),jac=True,method='L-BFGS-B',bounds=[(-30,30)]*(2*n),options=dict(maxiter=1500,ftol=2e-13,gtol=1e-8,maxcor=16))
    return cap*expit(prior+fit.x[:n,None]+fit.x[None,n:]),dict(iterations=fit.nit,gradient_max=float(abs(fit.jac).max()),message=str(fit.message))


def witness(prob,first,second,hardware):
    # Optimize over monotone [0,1] tests. For each photon number choose a
    # quadratic that upper-bounds the test at every admissible click count.
    selected=np.flatnonzero(prob>1e-6)
    width=int(selected.max()+1)
    ix=[];cols=[];vals=[];rhs=[]
    def add(cs,vs,limit=0):
        ix.extend([len(rhs)]*len(cs));cols.extend(cs);vals.extend(vs);rhs.append(limit)
    terms=[]
    for i,n in enumerate(selected):
        mu=first[n]/prob[n]
        sd=np.sqrt(max(.02,(second[n]+first[n])/prob[n]-mu*mu))
        ts=(np.arange(n+1)-mu)/sd
        terms.append((mu,sd,ts))
        vec=width+3*i
        for c,t in enumerate(ts):add([c,vec,vec+1,vec+2],[1.,-1.,-t,-t*t])
    for j in range(width-1):add([j,j+1],[1.,-1.])
    hw=np.r_[hardware[:width-1],hardware[width-1:].sum()]
    obj=np.r_[-hw,np.column_stack((prob[selected],np.zeros(len(selected)),prob[selected])).ravel()]
    mat=coo_matrix((vals,(ix,cols)),shape=(len(rhs),len(obj))).tocsr()
    ans=linprog(obj,A_ub=mat,b_ub=rhs,bounds=[(0.,1.)]*width+[(None,None)]*(3*len(selected)),method='highs',options={'dual_feasibility_tolerance':1e-8,'primal_feasibility_tolerance':1e-8})
    if not ans.success:raise RuntimeError(ans.message)
    h=np.maximum.accumulate(np.clip(ans.x[:width],0.,1.))
    coeff=ans.x[width:].reshape(-1,3)
    upper=1-prob[selected].sum()
    slack=0.
    for n,(_,_,t),v in zip(selected,terms,coeff):
        # Repair any floating LP feasibility residual on the actual discrete support.
        err=max(0.,float(np.max(h[:n+1]-v@[np.ones_like(t),t,t*t])))
        slack=max(slack,err)
        upper+=prob[n]*(v[0]+v[2]+err)
    return h,float(upper),dict(n_range=[int(selected.min()),int(selected.max())],omitted_mass=float(1-prob[selected].sum()),largest_lp_repair=slack,training_gap=float(hw@h-upper)),selected,coeff


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    raw=np.load(PILOT/'joint.npz')['hist']
    hist=raw/raw.sum()
    z=np.load(OPT/'moments.npz')
    ph,first,second=[np.r_[z[k][:MAX-1],z[k][MAX-1:].sum()] for k in ('prob','click_mass','factorial_mass')]
    hw=np.load(HIST)['total_click_histogram']
    train=np.random.default_rng(104003000).multivariate_hypergeometric(hw,hw.sum()//2,method='marginals')
    test=hw-train
    goal=np.r_[train[:MAX-1],train[MAX-1:].sum()]/train.sum()
    table={}
    diagnostics={}
    for label,cap in [('optical',1.65),('hardware',8.)]:
        if label=='optical':
            w=optical_weights(hist,ph,first,second,cap)
        else:
            w,stats=hardware_weights(hist,ph,goal,cap)
            diagnostics['bounded_fit']=stats
        table[label]=w/cap
        mass=hist*table[label]
        rate=mass.sum()
        table[label+'_rate']=rate
        diagnostics[label]=dict(rate=rate,train_photon_ks=float(abs(mass.sum(1).cumsum()/rate-ph.cumsum()).max()),train_click_ks=float(abs(mass.sum(0).cumsum()/rate-goal.cumsum()).max()))
    h,upper,diag,selected,co=witness(z['prob'],z['click_mass'],z['factorial_mass'],train/train.sum())
    obs=np.r_[test[:len(h)-1],test[len(h)-1:].sum()]@h/test.sum()
    allobs=np.r_[hw[:len(h)-1],hw[len(h)-1:].sum()]@h/hw.sum()
    # A DKW band for the full hardware CDF remains valid for an adaptively
    # selected monotone witness, conditional on the iid hardware assumption.
    radius=float(np.sqrt(np.log(200)/(2*hw.sum())))
    diagnostics['cdf_witness']=dict(**diag,optical_upper=upper,heldout_hw=float(obs),all_hw=float(allobs),empirical_gap=float(allobs-upper),hardware_dkw_radius99=radius,population_lower99=float(allobs-upper-radius),train_records=int(train.sum()),test_records=int(test.sum()))
    np.savez_compressed(out/'tables.npz',**table)
    np.savez_compressed(out/'witness.npz',h=h,upper=upper,train=train,test=test,selected=selected,coeff=co)
    save(out,'summary.json',diagnostics)
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
