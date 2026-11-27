"""Validated optical tail and dual inequalities from the archived binary input.

Requires python-flint. Every floating input is converted via its integer ratio.
No numerical eigensystem or numerical optimization is trusted by this verifier.
"""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from flint import arb,acb,arb_mat,acb_mat,ctx

ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'results/iteration83_loss_pilot_v1/input.npz'
OUT=ROOT/'results/iteration120_contractions'


def exact(x):
    a,b=float(x).as_integer_ratio()
    return arb(a)/b


def diagonal(vals):
    n=len(vals)
    return acb_mat(n,n,[vals[i] if i==j else 0 for i in range(n) for j in range(n)])


def pair_matrix(e,s):
    n=len(s);I=diagonal([1]*n);D=diagonal([-exact(x).tanh() for x in s])
    # Contract the zero-momentum environment bra with the pure joint squeezed ket.
    return (I-D*e.transpose()*e).solve(D)


def pair_masses(g,f):
    I=diagonal([1]*g.nrows())
    h=f.conjugate().transpose()*g*f*g.conjugate()
    det=(I-h).det();assert det.imag.contains(0) and det.real>0
    prob=[det.real.sqrt()]
    powers=h;traces=[]
    for order in range(1,11):
        tr=powers.trace();assert tr.imag.contains(0)
        traces.append(tr.real)
        prob.append(sum(traces[k]*prob[order-1-k] for k in range(order))/(2*order))
        if order<10:powers=powers*h
    return [sum(prob[:k+1]) for k in range(11)]


def optical_data(input_path=INPUT):
    with np.load(input_path) as src:
        raw=src['T'];squeezing=src['squeezing'].T.repeat(2,axis=1);order=np.argsort(src['order'])
    # A row permutation cancels from T^dagger T.
    m,n=raw.shape
    a=acb_mat(m,n,[acb(exact(x.real),exact(x.imag)) for x in raw.ravel()])
    scale=1/arb(2).sqrt();u=acb_mat(n,n)
    for i in range(0,n,2):
        u[i,i]=u[i+1,i]=acb(0,scale)
        u[i,i+1]=scale;u[i+1,i+1]=-scale
    g=u.conjugate().transpose()*(a.conjugate().transpose()*a)*u
    I=diagonal([1]*n);A=I-g
    trace=(g**16).trace()
    norm=trace.real.root(16);assert trace.imag.contains(0) and norm<1
    # Newton--Schulz on A=I-G. For each true eigenvalue the residual obeys
    # r_next=r^2(3+r)/4. The difference of Y and sqrt(A) has norm at most r.
    y=A;z=I;err=norm;steps=0
    while not err<arb(2)**-110:
        step=(3*I-z*y)*acb(.5);y=y*step;z=step*z
        err=err*err*(3+err)/4;steps+=1
    radius=arb(0,err.upper())
    enclosure=acb(radius,radius)
    e=acb_mat(n,n,[v+enclosure for v in y.entries()])
    matrices=[pair_matrix(e,s) for s in squeezing]
    a,b=[pair_masses(g,f) for f in matrices]
    return [a[7]*b[5],a[9]*b[6],a[10]*b[6]],steps,norm,g,matrices,raw[order],u


def principal_positive(mat):
    # Sylvester's criterion; the true matrix is real symmetric.
    lowers=[]
    for k in range(1,5):
        sub=arb_mat(k,k,[mat[i,j] for i in range(k) for j in range(k)])
        det=sub.det()
        assert det>0,det
        lowers.append(float(det.lower()))
    return min(lowers)


def verify_dual(R,cert):
    shells=[R[0],R[1]-R[0],R[2]-R[1],1-R[2]]
    assert all(x>0 for x in shells)
    states=[]
    for i,norm in enumerate([*R,arb(1)]):
        col=arb_mat(4,1,[(shells[j]/norm).sqrt() if j<=i else 0 for j in range(4)])
        states.append(col*col.transpose())
    b,m,p,o=states
    # Include both the stated decimal constants and their binary64 encodings.
    lo,hi,l,u=[arb(str(x))+arb(0,abs(exact(x)-arb(str(x))).upper()) for x in cert['clamps']]
    all_bounds=[];smallest=[]
    for dual in cert['dual']:
        y=arb_mat([[exact(x) for x in row] for row in dual['Y']])
        assert y==y.transpose()
        least=principal_positive(y);j=0
        for w,inner in [(l*m,[l*m-p]),(p,[p-l*m,u*m-p]),(u*m,[p-u*m])]:
            for g,outer in [(lo*b,[lo*b-w]),(w,[w-lo*b,hi*b-w]),(hi*b,[w-hi*b])]:
                cons=inner+outer;weight=dual['multipliers'][j];j+=1
                slack=y-dual['sign']*(o-g)
                for value,cut in zip(weight,cons,strict=True):
                    assert value>=0;slack=slack-exact(value)*cut
                least=min(least,principal_positive(slack))
        all_bounds.append(y.trace());smallest.append(least)
    assert len(all_bounds)==2 and {x['sign'] for x in cert['dual']}=={-1,1}
    return all_bounds,smallest


def run(write=True):
    ctx.prec=240;ctx.threads=2
    R,steps,norm,*_=optical_data()
    cert=json.loads((OUT/'operator_certificate.json').read_bytes())
    upper,slacks=verify_dual(R,cert)
    assert all(x<arb('0.004815') for x in upper)
    def H(M,F):return 2*M*(1-F)/(((M-1)**2+4*M*(1-F)).sqrt()+M-1)
    base,mid=R[0]/R[2],R[1]/R[2]
    raw=(1-R[2]).sqrt()
    loss=H(arb(4),base)+H(arb('1.25'),mid)
    extra=arb('.5')*H(arb(2),base)+arb('.8')*H(arb('1.25'),mid)
    assert loss>extra and raw+loss<arb('0.004824')
    info=dict(precision_bits=ctx.prec,optical_input_sha256=hashlib.sha256(INPUT.read_bytes()).hexdigest(),
              contraction_norm_upper=str(norm),root_iterations=steps,
              retention_balls=[str(x) for x in R],triangle_upper_ball=str(raw+loss),
              positive_deficit_surplus_dual_traces=[str(x) for x in upper],
              min_positive_minor_lower_approx=slacks,certified_full_law_upper='0.004815',
              scope='Exact-arithmetic sampler for the dyadic archived transfer and squeezing inputs; no claim of certified runtime floating-point sampling.')
    if write:
        (OUT/'interval_optical.json').write_bytes((json.dumps(info,indent=2)+'\n').encode())
    else:
        deposited=json.loads((OUT/'interval_optical.json').read_bytes())
        assert deposited['optical_input_sha256']==info['optical_input_sha256']
        assert deposited['certified_full_law_upper']==info['certified_full_law_upper']
    print(info,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--write',action='store_true')
    run(parser.parse_args().write)
