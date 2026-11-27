"""Paired timing and byte-level equality of sequential and fused spatial corrections."""
import argparse
import ctypes as ct
import json
from pathlib import Path
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration103_gpu_v1 import build as build_gpu,I,U,CU,P
from iteration112_hardware_v1 import engines,attach,DATA


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    build_gpu(out,'spatial_fused112_v1.cu')
    lib=ct.CDLL(str(out/'gpu.dll'))
    fn=lib.fused_emit112;fn.argtypes=[I,I,U,CU]+[P]*8
    c=np.load(DATA/'iteration112_hardware_v1/calibration.npz')
    first=np.load(DATA/'iteration111_transport_v1/holdout.npz')
    assert c['groups'].shape==first['groups'].shape==(18,8)
    assert np.array_equal(np.sort(c['groups'].ravel()),np.sort(first['groups'].ravel()))
    torch.backends.cuda.matmul.allow_tf32=False;make=engines()
    rows=[]
    for route in (0,1):
        base,seq,fused=[make(route) for _ in range(3)]
        attach(seq,c['groups'],c[f'gate{route}'],c[f'deficit{route}'])
        args=[fused.gpu_groups,fused.gpu_cdf,fused.move]+[torch.as_tensor(np.ascontiguousarray(v),device='cuda')
                    for v in (c['groups'],c[f'deficit{route}'],c[f'gate{route}'])]
        def emit(g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf,out,error,stream):
            rc=fused.without_windows(g,f,s,n,pn,wide,seed,off,field,eig,lut,cdf,out,error,stream)
            if rc:return rc
            return fn(n,wide,seed,off,*[v.data_ptr() for v in args],out,stream)
        fused.emit=emit
        for off in range(0,65536,2048):
            a=seq.draw(2048,route,112751000+route,off);b=fused.draw(2048,route,112751000+route,off)
            assert np.array_equal(a,b)
        rng=np.random.default_rng(112752000+route)
        us={}
        for batch in (2048,16384):
            times=[[] for _ in range(3)]
            for rep in range(15):
                for k in rng.permutation(3):
                    start=perf_counter()
                    (base,seq,fused)[k].draw(batch,route,112753000+route,batch*rep)
                    if rep>=3:times[k].append((perf_counter()-start)*1e6/batch)
            us[str(batch)]=times
        rows.append(dict(pnr=bool(route),identical=65536,times=us))
        print(route,[np.median(t) for t in us['16384']],flush=True)
    (out/'fused.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2):run(ap.parse_args().output.resolve())
