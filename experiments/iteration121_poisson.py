"""A finite, bounded PNR witness for every mixture of independent Poisson outputs."""
import argparse
import json
import numpy as np
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits

from iteration114_rank_v1 import realify
from iteration115_visible_v1 import DATA,hardware

OUT = DATA/'iteration121_positive_p'
ETA_DEN = 48


def design():
    t,ss = hardware()
    n = len(t)
    u = realify(t)
    gram = u.T@u
    c = np.zeros((n,n))
    mu = np.zeros(n)
    for s in ss:
        d = np.expm1(np.r_[-2*s,2*s])/(2*ETA_DEN)
        h = np.linalg.solve(np.eye(len(d))+d[:,None]*gram,np.diag(d))
        k = u@h@u.T
        c += sum(k[i*n:(i+1)*n,j*n:(j+1)*n]**2 for i in (0,1) for j in (0,1))/2
        mu += (k.diagonal()[:n]+k.diagonal()[n:])/2
    _,vec = eigh(c,subset_by_index=[0,0])
    v = np.rint(vec[:,0]/np.max(abs(vec))*2**18).astype(np.int64)
    b_numer = round(float(v@mu)*2**32)
    data = dict(eta_den=ETA_DEN,v=v.tolist(),b_numer=b_numer,b_denom=2**32)
    (OUT/'poisson_witness.json').write_bytes((json.dumps(data,indent=2)+'\n').encode())


def verify():
    from flint import arb,arb_mat,acb,acb_mat,ctx
    from iteration120_interval import INPUT,exact
    ctx.prec = 160
    ctx.threads = 2
    data = json.loads((OUT/'poisson_witness.json').read_bytes())
    v = data['v']
    b = arb(data['b_numer'])/data['b_denom']
    eta = 1/arb(data['eta_den'])
    with np.load(INPUT) as f:
        raw = f['T'][np.argsort(f['order'])]
        ss = f['squeezing'].T.repeat(2,axis=1)
    n,m = raw.shape
    a = acb_mat(n,m,[acb(exact(z.real),exact(z.imag)) for z in raw.ravel()])
    transform = acb_mat(m,m)
    scale = 1/arb(2).sqrt()
    for j in range(0,m,2):
        transform[j,j] = transform[j+1,j] = acb(0,scale)
        transform[j,j+1] = scale
        transform[j+1,j+1] = -scale
    t = a*transform
    weighted = acb_mat(n,m,[z*w for row,w in zip(t.tolist(),v) for z in row])
    def real_gram(c):
        values = c.tolist()
        top = [[z.real for z in row]+[-z.imag for z in row] for row in values]
        bot = [[z.imag for z in row]+[z.real for z in row] for row in values]
        return arb_mat(top+bot)
    g,gv = [real_gram(t.conjugate().transpose()*x) for x in (t,weighted)]
    ident = arb_mat(np.eye(2*m,dtype=int).tolist())
    grad,hess,p0 = arb(0),arb(0),arb(1)
    for s in ss:
        d = [eta*((sign*2*exact(x)).exp()-1)/2 for sign in (-1,1) for x in s]
        D = arb_mat(2*m,2*m,[d[i] if i==j else 0 for i in range(2*m) for j in range(2*m)])
        A = ident+D*g
        det = A.det()
        assert det>0
        p0 /= det.sqrt()
        h = A.solve(D)*gv
        grad += sum(h[i,i] for i in range(2*m))/2
        h2 = h*h
        hess += sum(h2[i,i] for i in range(2*m))/2
    optical = p0*((grad-b)**2+hess)
    # Absolute values of all coefficients are bounded by max(b^2,2 max|v|^2,
    # 2|b|max|v|); use twice this limit for a safe range.
    vmax = arb(max(map(abs,v)))
    span = 2*(b*b+2*abs(b)*vmax+2*vmax*vmax)
    lower = -optical/span
    assert lower>arb('4.26e-10')
    result = dict(precision=ctx.prec,thinned_vacuum_ball=str(p0),optical_score_ball=str(optical),
                  oscillation_upper_ball=str(span),full_pnr_tv_lower_ball=str(lower),
                  certified_lower='4.26e-10',scope='Every mixture of conditionally independent nonnegative Poisson rates.')
    (OUT/'poisson_certificate.json').write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(result,flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('stage',choices=['design','verify'])
    with threadpool_limits(limits=4):
        (design if ap.parse_args().stage=='design' else verify)()
