"""Compare complete projected Frugal laws on the frozen optical stress and random panels."""
import argparse
import ctypes as ct
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import Native,save,manifest
from iteration100_projection_v1 import physical
from iteration100_baselines_v1 import frugal
from iteration105_jiuzhang_v1 import inputs
from iteration106_packed_v1 import OUT,DATA,BASE


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    runtime=Native(BASE/'spatial.dll')
    lib=ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    z=inputs()[0];rows=[]
    for j in (0,1,4,5,6,7,8):
        v=np.load(OUT/f'panel_{j}.npz');ids=v['ports'];truth=v['target']
        q=physical(z,ids);assert len(ids)==16
        row=dict(panel=j,ports=ids.tolist(),packed_full_output_projection_tv=float(abs(v['1']-truth).sum()/2))
        for k in (2,4):
            p,times=frugal(lib,q,k,106190000+100*j+20*k)
            row[f'K{k}']=dict(tv=float(abs(p-truth).sum()/2),us=float(np.median(times)))
        rows.append(row);print(j,row['K2'],row['K4'],flush=True)
    save(out,'summary.json',dict(rows=rows,scope='Frugal CPU FP64 projected input, 2048 records, one thread, two independent spectra. The first 16-bit K2 prefix of the actual full-width run is independently verified in round 105.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
