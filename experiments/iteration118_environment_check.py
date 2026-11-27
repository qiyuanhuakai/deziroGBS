"""Independent oscillator-diagonalized test of the conjugate-environment kernel."""
import json
import numpy as np
from threadpoolctl import threadpool_limits
from iteration115_displacement_v1 import conditional
from iteration118_auxiliary import OUT, prepare
from iteration118_environment import evaluate,setup


def run():
    t=np.array([[.64+.31j]]);s=np.array([.95]);cap=2;dim=80
    _,V,_,R,_=conditional(t,s)
    _,S,_,_,_=prepare(t,s,cap)
    T=(-R.T@np.array([[0,1],[-1,0]])).ravel()
    a=np.diag(np.sqrt(np.arange(1,dim)),1)
    eig,U=np.linalg.eigh(T[0]*(a+a.T)-1j*T[1]*(a-a.T))
    psi=np.zeros(dim,complex);psi[0]=1
    for n in range(2,dim,2):psi[n]=psi[n-2]*S[0,0]*np.sqrt((n-1)/n)
    psi/=np.linalg.norm(psi);cut=psi.copy();cut[2*cap+1:]=0;cut/=np.linalg.norm(cut)
    g=U.conj().T@psi;c=U.conj().T@cut
    z=np.linspace(-4,4,37);y=z*np.sqrt(1+T@V@T)
    weight=np.exp(-.5*(y[:,None]-eig)**2)
    target=weight@abs(g)**2
    truth=weight@abs(c)**2/target
    overlap=weight@(g.conj()*c)/target
    est,cross=evaluate(setup(t,s,cap),z[:,None])
    errors=[float(abs(est-truth).max()),float(abs(cross-overlap).max())]
    assert max(errors)<2e-11,errors
    info=dict(fock_dimension=dim,real_nodes=len(z),max_likelihood_error=errors[0],max_cross_error=errors[1])
    (OUT/"environment_check.json").write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    print(info)


if __name__=="__main__":
    with threadpool_limits(limits=1):run()
