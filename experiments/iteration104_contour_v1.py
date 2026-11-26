"""Re-evaluate the frozen hardware/optical separation witness on a different complex contour."""
import argparse
import ctypes as ct
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,ORDER,Native,realify,bind,I,D,save,manifest
from iteration103_sample_v1 import BASE
from iteration104_optical_v1 import OPT
from iteration104_sample_v1 import FIT


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    runtime=Native(BASE/'spatial.dll')
    calc=bind(ct.CDLL(str(OPT/'moments.dll')),'moments104',[I,D,D,D],None)
    spec=np.load(ORDER)
    m=len(spec['v'])
    v=realify(spec['v'][np.argsort(spec['order'])])[np.c_[np.arange(m),np.arange(m)+m].ravel()]
    ev=[np.linalg.eigh(x) for x in spec['x']]
    u=[np.ascontiguousarray(v@y) for _,y in ev]
    length=640
    radius=1.008
    # Shift the Fourier nodes by half a cell, so no original unit-circle node recurs.
    theta=-2*np.pi*(np.arange(length//2)+.5)/length
    f=np.zeros((3,length),complex)
    for j,angle in enumerate(theta):
        z=radius*np.exp(1j*angle)
        gain=np.exp(sum(-.5*np.log1p((1-z)*e).sum() for e,_ in ev))
        f[0,j]=gain
        r=[z*e/(1+(1-z)*e) for e,_ in ev]
        eta=np.prod([min(1.,float((1+x).real.min())) for x in r])
        if abs(gain)*m*(m-1)*(1+2/eta+1/eta**2)>1e-12:
            q=[]
            for a,y in zip(r,u):
                mat=(y*a.real)@y.T+1j*((y*a.imag)@y.T)
                mat.flat[::2*m+1]+=1.
                q.append(mat)
            ans=np.empty(4)
            calc(m,q[0].view(float),q[1].view(float),ans)
            f[1:,j]=gain*ans.view(complex)
        f[:,-1-j]=f[:,j].conj()
    n=np.arange(length)
    mass=(np.fft.ifft(f,axis=1)*np.exp(1j*np.pi*n/length)/radius**n).real
    z=np.load(OPT/'moments.npz')
    w=np.load(FIT/'witness.npz')
    h=w['h'];indices=w['selected'];co=w['coeff']
    upper=1.-mass[0,indices].sum()
    for k,(a,b,c) in zip(indices,co):
        mu=z['click_mass'][k]/z['prob'][k]
        sd=np.sqrt((z['factorial_mass'][k]+z['click_mass'][k])/z['prob'][k]-mu**2)
        b2=c/sd**2;b1=b/sd-2*mu*b2;b0=a-b*mu/sd+b2*mu**2
        support=np.arange(k+1)
        correction=max(0.,float(np.max(h[:k+1]-(b0+b1*support+b2*support**2))))
        upper+=(b0+correction)*mass[0,k]+b1*mass[1,k]+b2*(mass[2,k]+mass[1,k])
    diff=float(abs(upper-w['upper']))
    absdiff=np.max(abs(mass[:,indices]-np.array([z[k][indices] for k in ('prob','click_mass','factorial_mass')])),axis=1)
    R=1.15
    alias=float((radius/R)**length*np.exp(sum(-.5*np.log1p((1-R)*e).sum() for e,_ in ev))*m*m)
    assert diff<2e-10 and absdiff[2]<1e-10 and alias<1e-16
    save(out,'summary.json',dict(nodes=length,radius=radius,phase_shift=np.pi/length,absolute_mass_discrepancies=absdiff.tolist(),factorial_alias_upper=alias,alternative_optical_upper=float(upper),witness_absolute_difference=diff))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=6,user_api='blas'):main(p.parse_args().output.resolve())
