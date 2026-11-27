"""Compare CUDA field construction and the fitted FP64 maps to CPU arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

import iteration121_goodman as g
from iteration121_witness import ROOT


def run(cache):
    t,ss = g.hardware()
    results = []
    for route in ('pnr','click'):
        with np.load(cache/(route+'.npz')) as f:
            layers = [(f[f'{i}_0'],f[f'{i}_1']) for i in range(10)]
        ref = json.loads((ROOT/(route+'_fp64.json')).read_bytes())
        assert ref['layer_sha256']==[hashlib.sha256(a.tobytes()+b.tobytes()).hexdigest() for a,b in layers]
        for wide in (False,True):
            fields = g.Fields(t,ss,wide=wide)
            rng = g.generator(route,13,int(wide))
            shape = (len(ss),g.BATCH,t.shape[1])
            noises = [torch.randn(shape,device=g.DEVICE,dtype=fields.dtype,generator=rng)[:,:64].cpu().numpy() for _ in (0,1)]
            intensities = np.zeros((64,len(t)),complex)
            for j,s in enumerate(ss):
                n = np.sinh(s)**2
                m = np.sinh(s)*np.cosh(s)
                x = noises[0][j]*np.sqrt((n+m)/2)
                y = noises[1][j]*np.sqrt((m-n)/2)
                intensities += ((x-y)@t.T)*((x+y)@t.conj().T)
            expected = (intensities if route=='pnr' else np.exp(-intensities)).real
            raw = fields.raw(g.generator(route,13,int(wide)),route)[:64]
            err = float(np.max(abs(raw.cpu().numpy()-expected)))
            assert err<(1e-11 if wide else 2e-6)
            row = dict(route=route,wide=wide,field_max_absolute_error=err)
            if wide:
                hi = None if route=='pnr' else 1.
                cpu = np.clip(expected,0,hi)
                gpu = g.clip(raw,route)
                for mat,offset in layers:
                    cpu = np.clip(cpu@mat+offset,0,hi)
                    gpu = g.clip(gpu@torch.tensor(mat,device=g.DEVICE)+torch.tensor(offset,device=g.DEVICE),route)
                row['ten_layer_max_absolute_error'] = float(abs(cpu-gpu.cpu().numpy()).max())
                assert row['ten_layer_max_absolute_error']<2e-8
            results.append(row)
    (ROOT/'field_checks.json').write_bytes((json.dumps(results,indent=2)+'\n').encode())
    print(results,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cache',type=Path,required=True)
    with threadpool_limits(limits=4),torch.inference_mode():
        run(parser.parse_args().cache)
