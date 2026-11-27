"""Full-width two-spectrum adaptation of Goodman et al., arXiv:2604.12330v1.

Training and design/test trajectories use separate streams. Optical targets are
computed independently of those trajectories. The training maps stay fixed.
"""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from scipy.linalg import eigh
from scipy.stats import beta
from threadpoolctl import threadpool_limits

from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration114_rank_v1 import physical_total
from iteration115_goodman_review_v3 import targets
from iteration115_visible_v1 import DATA, hardware
from iteration120_calibration import BANKS, PAIRS, vacancy

OUT = DATA / 'iteration121_positive_p'
BATCH = 2048
TRAIN = 131072
DESIGN = 131072
TEST = 524288
DEVICE = 'cuda'


def moments(raw, gains, route):
    x = optical_x(raw, gains)
    m = len(raw)
    port = np.arange(m)
    if route == 'pnr':
        n = (x[:, :m, :m] + x[:, m:, m:] + 1j*(x[:, m:, :m]-x[:, :m, m:]))/2
        a = (x[:, :m, :m] - x[:, m:, m:] + 1j*(x[:, m:, :m]+x[:, :m, m:]))/2
        return n.diagonal(axis1=1,axis2=2).real.sum(0), (abs(n)**2+abs(a)**2).sum(0)
    axes = np.stack([port, port+m],axis=1)
    mean = vacancy(x,axes)
    second = np.empty((m,m))
    u,v = np.triu_indices(m,1)
    for off in range(0,len(u),8192):
        i,j = u[off:off+8192],v[off:off+8192]
        joint = vacancy(x,np.column_stack([i,j,i+m,j+m]))
        second[i,j] = second[j,i] = joint
    second[port,port] = vacancy(2*x,axes)
    return mean,second - np.outer(mean,mean)


def power(mat,exponent):
    e,u = eigh((mat+mat.T)/2)
    keep = e > 1e-13*e[-1]
    return (u[:,keep]*e[keep]**exponent) @ u[:,keep].T,e


def generator(route,stage,role):
    seed = np.random.SeedSequence([121, int(route=='pnr'), stage,role]).generate_state(1,dtype=np.uint64)
    return torch.Generator(device=DEVICE).manual_seed(int(seed[0]))


class Fields:
    def __init__(self,t,ss):
        self.t = torch.tensor(t.T.copy(),device=DEVICE,dtype=torch.complex64)
        self.width = len(t)
        n = np.sinh(ss)**2
        a = np.cosh(ss)*np.sinh(ss)
        self.plus = torch.tensor(np.sqrt((n+a)/2),device=DEVICE,dtype=torch.float32)
        self.minus = torch.tensor(np.sqrt((a-n)/2),device=DEVICE,dtype=torch.float32)

    def raw(self,rng,route):
        shape = (len(self.plus), BATCH, self.t.shape[0])
        x = torch.randn(shape,device=DEVICE,generator=rng)*self.plus[:,None]
        y = torch.randn(shape,device=DEVICE,generator=rng)*self.minus[:,None]
        a = (x-y).to(torch.complex64) @ self.t
        b = (x+y).to(torch.complex64) @ self.t
        v = (a*b.conj()).sum(0)
        return v.real if route=='pnr' else torch.exp(-v).real


def clip(x,route):
    return x.clamp(min=0,max=None if route=='pnr' else 1)


def train(fields,route,mean,cov,n):
    rng = generator(route,1,0)
    y = torch.empty((n,fields.width),device=DEVICE)
    for start in range(0,n,BATCH):
        y[start:start+BATCH] = clip(fields.raw(rng,route),route)
    color,e = power(cov,.5)
    layers,diagnostics = [],[]
    for _ in range(10):
        center = y.mean(0,dtype=torch.float64).cpu().numpy()
        center_gpu = torch.as_tensor(center,device=DEVICE)
        gram = torch.zeros((fields.width,fields.width),device=DEVICE,dtype=torch.float64)
        for block in y.split(8192):
            z = block.double()-center_gpu
            gram.addmm_(z.T,z)
        white,ev = power(gram.cpu().numpy()/n,-.5)
        mat = white@color
        offset = mean-center@mat
        layer = [torch.as_tensor(a.astype(np.float32),device=DEVICE) for a in (mat,offset)]
        for block in y.split(8192):
            block.copy_(clip(block@layer[0]+layer[1],route))
        layers.append(layer)
        diagnostics.append(dict(min_training_eigenvalue=float(ev[0]),max_training_eigenvalue=float(ev[-1]),
                                max_mean_error=float(np.max(abs(y.mean(0,dtype=torch.float64).cpu().numpy()-mean)))))
    return layers,dict(min_target_eigenvalue=float(e[0]),max_target_eigenvalue=float(e[-1]),
                       negative_target_eigenvalues=int(sum(e < -1e-12)),iterations=diagnostics)


def projections(t,ss,route):
    with np.load(DATA/'iteration120_contractions/calibration_predictions.npz') as f:
        laws = {'occupancy':f['published_witness']}
        for p,law in enumerate(f['published_pairs']):
            laws['pair'+str(p)] = law
    groups = {}
    if route=='pnr':
        groups['whole'] = np.arange(len(t))
        groups['even_banks'] = np.where((np.arange(len(t))//8)%2==0)[0]
        for name,idx in groups.items():
            spectrum = [np.fft.rfft(physical_total(t[idx],s,512)) for s in ss]
            laws[name] = np.fft.irfft(spectrum[0]*spectrum[1],n=512)
    return laws,groups


def histograms(rows,groups):
    clicks = (rows>0).reshape(-1,144,8).sum(2,dtype=np.int32)
    code = (clicks[:,BANKS]!=0) @ (1<<np.arange(8))
    out = {'occupancy':np.bincount(code,minlength=256)}
    for p,(a,b) in enumerate(PAIRS):
        out['pair'+str(p)] = np.bincount(9*clicks[:,a]+clicks[:,b],minlength=81)
    for name,idx in groups.items():
        out[name] = np.bincount(rows[:,idx].sum(1,dtype=np.int64)%512,minlength=512)
    return out


def evaluate(fields,layers,route,stage,n,laws,groups):
    rng,draw = [generator(route,stage,i) for i in (0,1)]
    hist = {k:np.zeros(len(v),np.int64) for k,v in laws.items()}
    times,examples = [],None
    for i in range(n//BATCH):
        torch.cuda.synchronize()
        start = perf_counter()
        y = clip(fields.raw(rng,route),route)
        for mat,offset in layers:
            y = clip(y@mat+offset,route)
        if route=='pnr':
            rows = torch.poisson(y,generator=draw).int()
        else:
            rows = torch.rand(y.shape,device=DEVICE,generator=draw)>y
        torch.cuda.synchronize()
        times.append(perf_counter()-start)
        a = rows.cpu().numpy()
        if examples is None:
            examples = a[:64].copy()
        for k,h in histograms(a,groups).items():
            hist[k] += h
    return hist,times,examples


def run(route,path,train_rows):
    path.mkdir(parents=True,exist_ok=True)
    dst = path/(route+'.json')
    if dst.exists():
        raise FileExistsError(dst)
    torch.set_float32_matmul_precision('highest')
    raw,gains = actual_input()
    t,ss = hardware()
    start = perf_counter()
    mean,cov = moments(raw,gains,route)
    idx = np.arange(272,288)
    m0,c0 = targets(raw[idx],gains)[int(route=='pnr')]
    assert np.max(abs(mean[idx]-m0))<1e-12 and np.max(abs(cov[np.ix_(idx,idx)]-c0))<1e-12
    optics = perf_counter()-start
    laws,groups = projections(t,ss,route)
    start = perf_counter()
    fields = Fields(t,ss)
    layers,diag = train(fields,route,mean,cov,train_rows)
    torch.cuda.synchronize()
    fit_seconds = perf_counter()-start
    arrays,stats = {},{}
    for stage,kind,count in [(2,'design',DESIGN),(3,'test',TEST)]:
        hs,times,example = evaluate(fields,layers,route,stage,count,laws,groups)
        arrays.update({kind+'_'+k:v for k,v in hs.items()})
        arrays[kind+'_seconds'] = times
        arrays[kind+'_examples'] = example
        if stage==2:
            # A union of bins is learned on the design fold. Select exactly one
            # projection on this fold, before the test stream is generated.
            masks = {k:hs[k]/count>opt for k,opt in laws.items()}
            chosen = max(laws,key=lambda k:sum((hs[k]/count-laws[k])[masks[k]]))
        else:
            mask = masks[chosen]
            hits = int(hs[chosen][mask].sum())
            target = float(laws[chosen][mask].sum())
            lower = float(beta.ppf(.005,hits,count+1-hits)) if hits else 0.
            stats = dict(chosen=chosen,hits=hits,optical_event=target,empirical_event=hits/count,
                         full_tv_lower_simultaneous99=max(0.,lower-target-2e-9),
                         empirical_projection_tv={k:float(abs(hs[k]/count-v).sum()/2) for k,v in laws.items()},
                         gpu_us_per_row=sum(times)*1e6/count,batch_median_us=float(np.median(times)*1e6/BATCH))
        print(route,kind,chosen,flush=True)
    for k,opt in laws.items():
        arrays['optical_'+k] = opt
        arrays['event_'+k] = masks[k]
    # Store model digests so the benchmark describes a single fitted generator.
    layer_hash = [hashlib.sha256(b''.join(z.cpu().numpy().tobytes() for z in layer)).hexdigest() for layer in layers]
    np.savez_compressed(path/(route+'.npz'),**arrays)
    result = dict(route=route,width=len(t),train=train_rows,design=DESIGN,test=TEST,batch=BATCH,
                  torch=torch.__version__,gpu=torch.cuda.get_device_name(),moments_s=optics,fit_s=fit_seconds,
                  layer_sha256=layer_hash,diagnostics=diag,**stats)
    dst.write_bytes((json.dumps(result,indent=2)+'\n').encode())
    print(route,stats,flush=True)


if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('route',choices=['click','pnr'])
    ap.add_argument('--out',type=Path,default=OUT)
    ap.add_argument('--train',type=int,default=TRAIN)
    args = ap.parse_args()
    if args.train%BATCH or args.train<BATCH:
        ap.error('training rows must be a positive multiple of the batch size')
    with threadpool_limits(limits=6),torch.inference_mode():
        run(args.route,args.out,args.train)
