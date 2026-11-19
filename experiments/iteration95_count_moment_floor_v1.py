"""Sharp numerical count-CDF floor for a finite support and prescribed moments."""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import diags, eye, hstack, vstack, csr_matrix

def moment_floor(hist,moments):
    empirical=hist/hist.sum()
    size=len(hist)-1
    cdf=empirical.cumsum()[:-1]
    x=(np.arange(size+1)-moments[0])/np.sqrt(moments[1])
    targets=[0.,1.,moments[2]/moments[1]**1.5]
    outputs=[]
    for degree in (1,2,3):
        # Variables are the candidate CDF minus the empirical CDF, and its sup norm.
        basis=np.array([x**k for k in range(1,degree+1)])
        eq=np.zeros((degree,size+1))
        eq[:,:-1]=np.diff(basis)
        rhs=basis@empirical-targets[:degree]
        # Rescale the moment constraints to avoid an ill-conditioned cubic row.
        scale=np.max(abs(eq),axis=1)
        eq/=scale[:,None];rhs/=scale
        mono=hstack([diags([1.,-1.],[0,1],shape=(size-1,size)),
                     csr_matrix((size-1,1))])
        last=-np.ones((size,1))
        ineq=vstack([mono,hstack([eye(size),last]),hstack([-eye(size),last])]).tocsr()
        upper=np.r_[np.diff(cdf),np.zeros(2*size)]
        low=np.r_[-cdf,0.];high=np.r_[1-cdf,1.]
        objective=np.zeros(size+1);objective[-1]=1
        sol=linprog(objective,A_ub=ineq,b_ub=upper,A_eq=eq,b_eq=rhs,
                    bounds=list(zip(low,high)),method='highs',
                    options={'dual_feasibility_tolerance':1e-9,'primal_feasibility_tolerance':1e-9})
        assert sol.success,sol.message
        law=np.diff(np.r_[0.,cdf+sol.x[:-1],1.])
        assert law.min()>-1e-8 and np.max(abs(basis@law-targets[:degree]))<1e-6
        # Weak dual lower bound with the box infimum, including any stationarity residual.
        y=np.minimum(0.,sol.ineqlin.marginals)
        z=sol.eqlin.marginals
        residual=objective-ineq.T@y-eq.T@z
        lower=upper@y+rhs@z+np.minimum(residual*low,residual*high).sum()
        assert lower<=sol.fun+1e-8 and sol.fun-lower<1e-7
        outputs.append((dict(degree=degree,minimum_ks=float(sol.fun),dual_lower=float(lower),
                       max_moment_residual=float(np.max(abs(basis@law-targets[:degree])))),
                       dict(law=law,inequality_multipliers=y,equality_multipliers=z)))
    return outputs
