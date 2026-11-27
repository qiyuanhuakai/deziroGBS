"""Matched enumeration and identical alias lookup at the two 13-port inputs."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration112_coarse_v1 import Native,alias,DATA
from iteration114_rank_v1 import optical,cov,oracle,gaussian_click,ONE,P,I


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    core=Native(out);fn=core.lib.categorical112
    fn.argtypes=[I,I,I,ct.c_uint64,ct.c_uint32,P,P,P]
    cpu=ct.CDLL(str(DATA/'iteration112_frugal_build_v1/cpu.dll'))
    tree=cpu.fc112_full13;tree.argtypes=[I]+[P]*6+[I]
    quad=oracle(True);msg=ct.create_string_buffer(4096);rows=[]
    for r in (2,4):
        t,s=optical(13,r);x=cov(t,s);covariance=np.ascontiguousarray(2*x+np.eye(26))
        z=np.load(DATA/f'iteration114_positive_v1/m13_r{r}.npz')
        for k in (0,2,4):
            setup=[]
            for rep in range(4):
                start=perf_counter()
                if k==0:law=gaussian_click(quad,x[None],ONE)
                else:
                    th=np.empty(8191);prev=th.copy();proposal=th.copy();stat=np.zeros(3)
                    assert tree(k,*[a.ctypes.data for a in (covariance,th,prev,proposal,stat)],msg,len(msg))==0
                    law=np.ones(1)
                    for depth in range(13):
                        u=np.nan_to_num(th[len(law)-1:2*len(law)-1],nan=1).clip(0,1)
                        law=np.r_[law*u,law*(1-u)]
                prob,alt=alias(law)
                if rep:setup.append(perf_counter()-start)
            us=[];buf=np.empty((2048,13),np.uint8)
            for rep in range(18):
                start=perf_counter()
                rc=fn(len(buf),13,1,114960000+1000*r+100*k+rep,0,alt.ctypes.data,prob.ctypes.data,buf.ctypes.data)
                if rep>2:us.append((perf_counter()-start)*1e6/len(buf))
                assert rc==0
            rows.append(dict(r=r,method='optical' if k==0 else f'frugal{k}',setup_s=float(np.median(setup)),
                us=float(np.median(us)),times=us,click_model_tv=float(abs(law-z['optical_click']).sum()/2)))
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    print([(v['r'],v['method'],v['setup_s'],v['us']) for v in rows],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(p.parse_args().output.resolve())
