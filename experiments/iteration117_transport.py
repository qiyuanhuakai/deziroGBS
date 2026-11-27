"""Fix the entire first-order optical coherence matrix after truncating the pure core."""
import numpy as np
from scipy.linalg import eigh


def moments(S):
    k=S@S.conj().T;eye=np.eye(len(S))
    ell=np.trace(k).real
    norm=1+ell/2+ell**2/8+np.trace(k@k).real/4
    exact_n=np.linalg.solve(eye-k,k)
    quartic_n=((1+ell/2)*k+k@k)/norm
    exact_m=np.linalg.solve(eye-k,S)
    quartic_m=((1+ell/2)*S+k@S)/norm
    return exact_n,quartic_n,exact_m,quartic_m


def square(x,power):
    e,u=eigh((x+x.conj().T)/2)
    assert e.min()>0
    return (u*e**power)@u.conj().T


def corrected(S,response,retained):
    r=len(S);L=(response[:r]+1j*response[r:])/2
    C=L@L.conj().T
    n,q,m,b=moments(S)
    target=C+n-q
    sc=square(C,.5);inv=square(C,-.5)
    transport=inv@square(sc@target@sc,.5)@inv
    result=transport@L
    error=float(abs(q+result@result.conj().T-(n+C)).max())
    assert error<1e-11
    delta=result-L
    # Bound the overlap loss from the relative Weyl displacement. Both zero-mean pure states
    # have even parity, hence the sine matrix element vanishes. Use I-cos H <= H^2/2.
    def variance(normal,anomalous):
        total=delta.conj().T@(np.eye(r)+2*normal)@delta
        pairing=delta.conj().T@anomalous@delta.conj()
        return float(np.trace(total-2*pairing).real)
    cost=.5*np.sqrt(variance(n,m)*variance(q,b))
    root=max(0.,np.sqrt(retained)-cost)
    return 2*np.vstack((result.real,result.imag)),dict(max_normal_matrix_residual=error,
               mean_added_core_deficit=float(np.trace(n-q).real),mean_squared_shift=float(abs(delta).ravel()@abs(delta).ravel()),
               overlap_decrement=float(cost),root_fidelity_lower=float(root),
               anomalous_frobenius_residual=float(np.linalg.norm(b+result@result.T-m-L@L.T)))
