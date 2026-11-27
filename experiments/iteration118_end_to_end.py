"""Check actual rank-one readout draws against independent two-mode oscillator probabilities."""
import json
import math
import numpy as np
from scipy.linalg import expm
from threadpoolctl import threadpool_limits
from iteration118_auxiliary import OUT,draw,prepare


def reference(model,cap,order):
    q,S,_,field,_=model
    pair=q@S@q.T;dim=32
    create=np.diag(np.sqrt(np.arange(1,dim)),-1)
    power=np.zeros((dim,dim),complex);power[0,0]=1.;psi=power.copy()
    for j in range(1,cap+1):
        power=(pair[0,0]*(create@create@power)/2+pair[0,1]*(create@power@create.T)
               +pair[1,1]*(power@create.T@create.T)/2)/j
        psi+=power
    psi/=np.linalg.norm(psi)
    nodes,weights=np.polynomial.hermite.hermgauss(order);prob=np.zeros((4,4))
    bins=np.minimum(np.arange(dim),3)
    for i in range(order):
        for j in range(order):
            alpha=field@(np.sqrt(2)*nodes[[i,j]])
            a,b=[expm(x*create-x.conjugate()*create.T) for x in alpha]
            joint=abs(a@psi@b.T)**2
            np.add.at(prob,(bins[:,None],bins[None,:]),joint*weights[i]*weights[j]/np.pi)
    assert abs(prob.sum()-1)<3e-14
    return prob


def run():
    t=np.array([[.54+.17j,.12+.08j],[.03-.21j,.57+.23j]])
    cap=7;model=prepare(t,np.array([.92,.62]),cap)
    p=reference(model,cap,24);old=reference(model,cap,18)
    clicks=np.zeros((2,2))
    for i,j in np.ndindex(p.shape):clicks[int(i>0),int(j>0)]+=p[i,j]
    n=2048;rows=[]
    for direct,ref in enumerate([p,clicks]):
        obs=np.zeros(ref.shape,dtype=int);attempts=0
        for k in range(n):
            bits,tries=draw(model,cap,direct,[118,744,direct,k])
            obs[tuple(np.minimum(bits,ref.shape[0]-1))]+=1;attempts+=tries
        # Independent per-record draws, Hoeffding union over both finite histograms.
        maximum=float(abs(obs/n-ref).max())
        limit=np.sqrt(np.log(2*(p.size+clicks.size)/.01)/(2*n))
        assert maximum<limit
        rows.append(dict(direct=bool(direct),shots=n,empirical_tv=float(abs(obs/n-ref).sum()/2),
                         max_cell_error=maximum,simultaneous99_cell_radius=float(limit),
                         mean_auxiliary_attempts=attempts/n,expected_attempts=(cap+1)/model[-1]))
    info=dict(routes=rows,reference_quadrature_max_change=float(abs(p-old).max()),oscillator_dimension=32,
              pnr_reference=p.tolist(),direct_click_reference=clicks.tolist())
    (OUT/"end_to_end.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    print(rows,flush=True)


if __name__=="__main__":
    with threadpool_limits(limits=1):run()
