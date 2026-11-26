"""Check full-record error control at fixed occupied rank on the same high-width optical inputs as the field benchmark."""
import argparse
import ctypes as ct
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_baselines_v1 import frugal
from iteration100_spatial_v1 import save,manifest
from iteration106_groups_v1 import Groups,forest,DATA
from iteration106_reference_v1 import Marked,sectors,truth_click,OUT as REF
from iteration106_width_v1 import case
from iteration106_weak_v1 import covariance
from iteration106_sample_v1 import benchmark,read


def main(out):
    out.mkdir(exist_ok=False,parents=True)
    rows=[];peer=[]
    baseline=read(DATA/'iteration106_width_v1/summary.json')['rows']
    for j,old in enumerate(row for row in baseline if row['sources']==2):
        m=old['m'];t,r0=case(m,2);r=r0[0];cap=16
        sources=np.stack((t[:,:2],t[:,2:]))
        nodes=forest(2,[[0,1]],np.ones((2,2)),{2:cap})
        g=Groups(sources,r,nodes)
        ids=np.array(old['panel'])
        law=Marked(REF/'laws.dll',sources,r,nodes).clicks(ids)
        truth=truth_click(sources,r,ids)
        tv=float(abs(law-truth).sum()/2);tail=float(1-sectors(r,cap).sum())
        assert tv<tail+2e-10
        n=min(2048,max(16,4194304//m))
        times=[benchmark(lambda n,s,o:g.draw(n,p,s,o,16),n,106180000+j*5+int(p),5)['us'] for p in (False,True)]
        assert np.array_equal(g.draw(11,True,106180100+j,5,4)>0,g.draw(11,False,106180100+j,5,1))
        rows.append(dict(m=m,batch=n,cap=cap,click_us=times[0],pnr_us=times[1],panel_tv=tv,
                         field_same_panel_tv=old['tv'],ideal_arithmetic_full_bound=tail+g.base.tail+g.base.rounding,ports=ids.tolist()))
        if m in (16,131072):
            cc=covariance(sources,r,np.arange(16))[None,:,:];true16=truth_click(sources,r,np.arange(16))
            local=Marked(REF/'laws.dll',sources,r,nodes).clicks(np.arange(16))
            lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
            if m==16:
                for k in (2,4):
                    p,ts=frugal(lib,cc,k,106185000+100*k)
                    peer.append(dict(m=m,K=k,frugal_projected_us=float(np.median(ts)),full_16bit_tv=float(abs(p-true16).sum()/2),ours_tv=float(abs(local-true16).sum()/2)))
            else:
                librow=dict(m=m,scope='First 16 detector positions of the 131072-output circuit',ours_tv=float(abs(local-true16).sum()/2))
                p,ts=frugal(lib,cc,2,106187000)
                librow.update(K=2,frugal_projected_us=float(np.median(ts)),full_16bit_tv=float(abs(p-true16).sum()/2))
                peer.append(librow)
        print(rows[-1],flush=True)
    save(out,'summary.json',dict(rows=rows,frugal=peer,scope='Output width varies, fixed two occupied twin-beam sources in one spectrum. The ideal-arithmetic full TV bound includes fallback mass and input-number tables. Finite floating arithmetic checked independently via numerical marginals and seeded replay.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
