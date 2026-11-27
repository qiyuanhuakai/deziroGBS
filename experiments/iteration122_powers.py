"""Sample both readouts or certify their full laws on the published power inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/iteration122_families'
POWERS = ('low','median','high')
BOUNDS = dict(low='0.002276',median='0.003628',high='0.004815')


def json_write(path,obj):
    path.write_bytes((json.dumps(obj,indent=2)+'\n').encode())


def pack():
    from openpyxl import load_workbook
    from iteration117_power_inputs import CACHE
    inventory = json.loads((ROOT/'results/iteration117_quartic/power_sources.json').read_bytes())
    for label in POWERS:
        folder = OUT/label
        folder.mkdir(parents=True,exist_ok=True)
        files = [row for row in inventory if f'/{label} power.' in row['name']]
        assert len(files)==2
        for row in files:
            p = CACHE/row['name']
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256']
            if p.suffix=='.mtx':
                a = np.loadtxt(p,skiprows=3)
                raw = (a[:,0]+1j*a[:,1]).reshape(1152,50,order='F')
            else:
                book = load_workbook(p,read_only=True,data_only=True)
                squeezing = np.array(list(book.active.values),float)
                book.close()
        np.savez_compressed(folder/'input.npz',T=raw,squeezing=squeezing,order=np.arange(len(raw)))
        json_write(folder/'sources.json',files)


def optics(label):
    with np.load(OUT/label/'input.npz') as f:
        raw = f['T'];ss = f['squeezing'].T.repeat(2,axis=1)
    t = np.empty_like(raw)
    t[:,::2] = 1j*(raw[:,::2]+raw[:,1::2])/np.sqrt(2)
    t[:,1::2] = (raw[:,::2]-raw[:,1::2])/np.sqrt(2)
    return t,ss


def certify(label,solve=False):
    from flint import ctx,arb
    from iteration120_interval import optical_data,verify_dual
    p = OUT/label
    ctx.prec=240;ctx.threads=2
    R,steps,norm,*_ = optical_data(p/'input.npz')
    if solve:
        from iteration120_operator import run
        cert=run([float(v.mid()) for v in R],[.5,4,.8,1.25])
        json_write(p/'dual.json',cert)
    upper,minors = verify_dual(R,json.loads((p/'dual.json').read_bytes()))
    assert all(v<arb(BOUNDS[label]) for v in upper)
    res = dict(power=label,input_sha256=hashlib.sha256((p/'input.npz').read_bytes()).hexdigest(),
               full_tv_upper=BOUNDS[label],retained=[str(v) for v in R],
               signed_upper=[str(v) for v in upper],positive_minor_minima=minors,
               gram_norm_upper=str(norm),square_root_steps=steps)
    if solve:
        json_write(p/'certificate.json',res)
    else:
        old=json.loads((p/'certificate.json').read_bytes())
        assert old['input_sha256']==res['input_sha256'] and old['full_tv_upper']==res['full_tv_upper']
    print(label,res['signed_upper'],flush=True)


def sample(label,route,stage,n):
    from threadpoolctl import threadpool_limits
    import iteration120_generate as gen
    gen.OUT = OUT/label
    k = POWERS.index(label)
    if not 1<=n<=16384:
        raise ValueError('batch outside reserved proposal index range')
    first = (4+2*k+route)*2**22
    with threadpool_limits(1):
        if stage=='base':
            gen.generate(route,n,10,optics(label),first)
            with np.load(gen.OUT/f'base_{route}_{n}.npz') as f:
                assert all(f[f'log{route}_{j}'][-1,0] < first+2**22 for j in range(2))
        else:
            gen.score(route,n,2,optics(label),1220+k)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['pack','solve','verify','base','score'])
    ap.add_argument('power',nargs='?',choices=POWERS)
    ap.add_argument('route',nargs='?',type=int,choices=[0,1])
    ap.add_argument('--candidates',type=int,default=128)
    args=ap.parse_args()
    if args.stage=='pack':
        pack()
    elif args.stage in ('solve','verify'):
        for p in ([args.power] if args.power else POWERS):
            certify(p,args.stage=='solve')
    else:
        if args.power is None or args.route is None:
            ap.error('sampling requires power and route')
        sample(args.power,args.route,args.stage,args.candidates)
