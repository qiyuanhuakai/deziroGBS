"""Full spatial panel bound from photon number in the discarded output modes."""
import argparse
import json
import numpy as np
from threadpoolctl import threadpool_limits

from iteration117_quartic_laws import physical_models
from iteration118_auxiliary import OUT,hardware,prepare


def sector_table(q,S,ports,top):
    # Joint law of input pair number n and photons a reaching the panel,
    # before the common classical displacement. For fixed n it is a
    # polynomial of degree 2n in the panel counting variable.
    eye=np.eye(len(S));gram=q[ports].conj().T@q[ports]
    p0=np.sqrt(np.linalg.det(eye-S@S.conj().T).real)
    size=2*top+1
    curves=np.zeros((top+1,size),complex)
    for j,z in enumerate(np.exp(2j*np.pi*np.arange(size)/size)):
        D=eye+(z-1)*gram
        eigen=np.linalg.eigvals(S.conj().T@D@S@D.T)
        logs=np.array([0]+[np.sum(eigen**k)/2 for k in range(1,top+1)])
        curves[0,j]=p0
        for n in range(1,top+1):curves[n,j]=logs[1:n+1]@curves[n-1::-1,j]/n
    raw=np.fft.fft(curves,axis=1)/size
    assert abs(raw.imag).max()<2e-14
    a=raw.real
    assert np.minimum(a,0).sum()>-4e-13
    h=np.zeros_like(a)
    for n,row in enumerate(a):
        h[n,:2*n+1]=np.maximum(0,row[:2*n+1][::-1])
    return h


def run(pair):
    optical=hardware();ports=(8*np.array(pair)[:,None]+np.arange(8)).ravel()
    ps=[];rs=[];hs=[];R=1.;top=32
    for s,k,(q,S,*_) in zip(optical[1],(7,5),physical_models(optical)):
        h=sector_table(q,S,ports,top)
        retain=prepare(optical[0],s,k)[-1]
        assert abs(retain-h[:k+1].sum())<3e-13
        ps.append(h.sum(0));rs.append(h[:k+1].sum(0));hs.append(h[k+1:].sum(0));R*=retain
    p=np.outer(*ps);r=np.outer(*rs)
    # Subtract the uncomputed global sectors at the pure-state level.
    # A PGF Markov bound avoids loss of precision from subtracting from one.
    tail=0.
    for _,S,*_ in physical_models(optical):
        eigen=np.linalg.eigvalsh(S@S.conj().T)
        radii=np.linspace(1.01,.999/eigen.max(),150)
        tail+=min(np.exp(.5*np.sum(np.log1p(-eigen)-np.log1p(-x*eigen))-
                        (top+1)*np.log(x)) for x in radii)
    # Per discarded-count sector the two partially projected vectors have
    # norms squared p and r/R, with inner product r/sqrt(R).
    # Positive form avoids cancelling nearly equal numbers in the radicand.
    high=np.outer(hs[0],ps[1])+np.outer(rs[0],hs[1])
    delta=.5*np.sqrt((p-r/R)**2+4*(r/R)*high).sum()
    tail_allowance=np.sqrt(tail)+tail/2
    info=dict(banks=pair,pair_caps=[7,5],explicit_pair_order=top,
              tail_mass_upper=tail,trace_tail_allowance=tail_allowance,
              panel_trace_upper=delta+tail_allowance,retained=R,
              panel_pnr_tv_upper=delta+tail_allowance,panel_click_tv_upper=delta+tail_allowance)
    print(info,flush=True)
    name="panel_distance"+("" if pair==[34,35] else "_"+"_".join(map(str,pair)))
    (OUT/(name+".json")).write_text(json.dumps(info,indent=2)+"\n",encoding="utf8")
    np.savez_compressed(OUT/(name+".npz"),unconditional=p,retained=r)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--banks",type=int,nargs=2,default=[34,35])
    with threadpool_limits(limits=1):run(p.parse_args().banks)
