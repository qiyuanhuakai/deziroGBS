"""Exercise dense-output widths across the original 16-bit port boundary and test increasing source count."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import realify,Native,save,manifest
from iteration106_field_v1 import Field,factors,brightness,optical,DATA,BASE
from iteration106_packed_v1 import field_panel
from iteration106_reference_v1 import truth_click
from iteration106_sample_v1 import benchmark
from iteration100_spatial_v1 import I,D,bind
import ctypes as ct


def case(m,k):
    rng=np.random.default_rng(106130000+m+k)
    z=rng.standard_normal((m,2*k))+1j*rng.standard_normal((m,2*k))
    z[:min(4,m),:min(4,2*k)]+=np.eye(min(4,m),min(4,2*k))*(1+.3j)*np.sqrt(m)
    u=np.linalg.qr(z,mode='reduced')[0]*np.linspace(.68,.91,2*k)
    r=(.55+.52*np.sin(.4+np.arange(k))**2)[None,:]
    return u,r


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    native=Native(BASE/'spatial.dll')
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    rows=[]
    for m,k in [(4,2),(16,2),(144,2),(1152,2),(4097,2),(16384,2),(65537,2),(131072,2),(64,8),(256,16),(1024,32),(4096,64)]:
        t,r=case(m,k);f=factors(t,r)
        eig=np.linalg.eigvalsh(2*f[0]@f[0].T)
        ideal=optical(t,r)
        profile=[p for p in brightness(ideal,eig) if p['feasible']][-1]
        model=Field(t,r,profile)
        ids=np.unique(np.r_[np.arange(min(4,m)),np.linspace(min(4,m-1),m-1,min(4,m),dtype=int)])
        truth=truth_click(np.stack([t[:,2*j:2*j+2] for j in range(k)]),r[0],ids)
        local=field_panel(f,profile,ids,oracle)
        n=min(2048,max(16,4194304//m))
        times=[]
        for pnr in (False,True):
            times.append(benchmark(lambda n,s,o:model.draw(n,pnr,s,o),n,106132000+m+k+int(pnr),5)['us'])
        x=model.draw(16,True,106133000+m+k,3)
        assert np.array_equal(x[4:]>0,model.draw(12,False,106133000+m+k,7))
        rows.append(dict(m=m,sources=k,mean_photons=float(ideal.sum()/2),brightness_atoms=profile['d'],batch=n,click_us=times[0],pnr_us=times[1],panel=ids.tolist(),tv=float(abs(truth-local).sum()/2)))
        del model
    t=np.zeros((65537,2),complex);t[[0,-1],0]=1/np.sqrt(2)
    g=Field(t,np.array([[1.2]]),dict(values=[1.],weights=[1.]))
    discordant=0
    for off in range(0,2048,128):
        x=g.draw(128,False,106140000,off)
        assert not x[:,1:-1].any()
        discordant+=int((x[:,0]!=x[:,-1]).sum())
    assert discordant>300
    np.savez_compressed(out/'example.npz',transfer=case(16,2)[0],squeezing=case(16,2)[1])
    save(out,'summary.json',dict(rows=rows,high_address_records=2048,high_address_discordant=discordant,domain_ports=[0,65536]))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    torch.backends.cuda.matmul.allow_tf32=False
    with threadpool_limits(limits=1,user_api='blas'):run(p.parse_args().output.resolve())
