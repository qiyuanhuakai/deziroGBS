"""Benchmark positive compressed branches against complete threshold and two-port PNR references."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration112_cases_v1 import optics, local_tables, complete, click_table, pair_counts
from iteration112_positive_v1 import positive


def run(out):
    out.mkdir(exist_ok=True,parents=True)
    refs=np.load(out.parent/'iteration112_cases_v1/laws.npz')
    rows=[];arrays={}
    for buses in (2,3,4):
        local,xs,s,axes,x,_=optics(buses)
        previous=None
        for budget in (17,33,65,129):
            start=perf_counter()
            pos,c,g,info,previous=positive(s,budget,112100+buses,steps=1600,initial=previous)
            fit=perf_counter()-start
            click=[];pn=[]
            for j,(xx,d) in enumerate(local_tables(local,xs,pos,axes)):
                click.append(click_table(xx,d))
                if j<2:
                    sel=[0,len(xx)//2];pn.append(pair_counts(xx[np.ix_(sel,sel)],d[...,sel],64))
            weight=np.outer(c,c)*g
            law=complete(click,weight)
            pair=((pn[1].reshape(11,-1)*weight.ravel())@pn[0].reshape(11,-1).T).real.ravel()
            tv=float(abs(law-refs[f'truth{buses}']).sum()/2)
            assert tv<info['state_bound']+1e-10 and abs(law.sum()-1)<1e-11 and law.min()>-1e-13
            key=f'b{buses}n{budget}'
            for name,value in [('pos',pos),('c',c),('law',law),('pair',pair)]:arrays[f'{key}_{name}']=value
            rows.append(dict(buses=buses,**info,click_tv=tv,pair_tv=float(abs(pair-refs[f'pair{buses}']).sum()/2),fit_s=fit))
            print(buses,budget,'TV',tv,'bound',info['state_bound'],'acceptance',info['acceptance'],flush=True)
            np.savez_compressed(out/'fits.npz',**arrays)
            (out/'fits.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output)
