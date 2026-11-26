"""Test frozen signed optical events on native dense outputs independently of moment fitting and timing."""
import argparse
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest
from iteration106_sample_v1 import panel_hist,read
from iteration106_words_v1 import prepare,DATA


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    model=prepare('optical');packed=DATA/'iteration106_packed_v1'
    panels=[np.load(packed/f'panel_{j}.npz') for j in range(9)]
    ref=np.load(DATA/'iteration106_sample_v1/pair.npz')['target']
    n=1048576
    # Union bound over two routes and nine fixed events; two-sided Hoeffding.
    radius=float(np.sqrt(np.log(2*18/.01)/(2*n)))
    rows=[]
    for pnr in (False,True):
        h=[np.zeros(len(v['target']),np.int64) for v in panels]
        for off in range(0,n,2048):
            x=model.draw(2048,pnr,106220000+pnr,off)
            for y,v in zip(h,panels):y+=panel_hist(x,v['ports'])
        for j,(y,v) in enumerate(zip(h,panels)):
            law=v['1'];optical=v['target'];event=law>optical
            empirical=float(y[event].sum()/n-optical[event].sum())
            tv=float(abs(law-optical).sum()/2)
            assert abs(empirical-tv)<.0045
            rows.append(dict(pnr=pnr,panel=j,model_tv=tv,observed_event_gap=empirical,simulator_full_tv_lower99=max(0.,empirical-radius)))
        np.savez_compressed(out/f'witness_{int(pnr)}.npz',**{str(j):y for j,y in enumerate(h)})
    x=model.draw(32768,True,106221000)
    a=np.bincount(x[:,[289,861]].astype(np.int64).clip(0,12)@np.array([1,13]),minlength=169)
    peer=read(DATA/'iteration103_baselines_v1/summary.json')['pnr'][0]
    save(out,'summary.json',dict(rows=rows,n=n,family_radius=radius,equal_pnr_records=len(x),equal_pnr_empirical_tv=float(abs(a/len(x)-ref).sum()/2),pe_empirical_tv=peer['pe_empirical_tv'],pe_us=peer['pe_us']))
    manifest(out)


if __name__=='__main__':
    torch.backends.cuda.matmul.allow_tf32=False
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
