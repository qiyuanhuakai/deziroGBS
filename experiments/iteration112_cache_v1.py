"""Apply identical 13-bit lookup to exact, Frugal and compressed distributions."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration112_coarse_v1 import Native,alias,DATA
from iteration112_cases_v1 import optics,oracle,gaussian_click,ONE


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    core=Native(out);fn=core.lib.categorical112
    I=ct.c_int;P=ct.c_void_p
    fn.argtypes=[I,I,I,ct.c_uint64,ct.c_uint32,P,P,P]
    cp=ct.CDLL(str(DATA/'iteration112_frugal_build_v1/cpu.dll'))
    tree=cp.fc112_full13;tree.argtypes=[I]+[P]*6+[I]
    quad=oracle(True);msg=ct.create_string_buffer(4096);rows=[]
    for buses in (2,4):
        _,_,_,_,x,_=optics(buses)
        fit=np.load(DATA/'iteration112_fit_v1/fits.npz')[f'b{buses}n129_law']
        cov=np.ascontiguousarray(2*x+np.eye(26))
        for idx,label in enumerate(('exact','frugal4','projected')):
            prep=[]
            for rep in range(4):
                start=perf_counter()
                if label=='exact':p=gaussian_click(quad,x[None],ONE)
                elif label=='frugal4':
                    t=np.empty(8191);a=t.copy();b=t.copy();stats=np.zeros(3)
                    assert tree(4,*[v.ctypes.data for v in (cov,t,a,b,stats)],msg,len(msg))==0
                    p=np.ones(1)
                    for depth in range(13):
                        u=np.nan_to_num(t[len(p)-1:2*len(p)-1],nan=1).clip(0,1)
                        p=np.r_[p*u,p*(1-u)]
                else:p=fit
                prob,alt=alias(p)
                if rep:prep.append(perf_counter()-start)
            truth=np.load(DATA/'iteration112_cases_v1/laws.npz')[f'truth{buses}']
            us=[];a=np.empty((2048,13),np.uint8)
            for rep in range(14):
                start=perf_counter()
                rc=fn(len(a),13,1,112940000+1000*buses+100*idx+rep,0,alt.ctypes.data,prob.ctypes.data,a.ctypes.data)
                if rep>=3:us.append((perf_counter()-start)*1e6/len(a))
                assert rc==0
            rows.append(dict(buses=buses,law=label,enumeration_and_alias_s=float(np.median(prep)),
                supplied_precomputed_law=label=='projected',us=float(np.median(us)),times=us,model_tv=float(abs(p-truth).sum()/2)))
    (out/'cache.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
    print([(r['law'],round(r['enumeration_and_alias_s'],5),round(r['us'],4)) for r in rows])


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output.resolve())
