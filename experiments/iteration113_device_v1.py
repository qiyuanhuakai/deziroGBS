"""All 144 spatial ports: joint true number and number of fired pseudo-PNR bins."""
import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration113_support_v1 import ROOT, library, sectors, tv, benchmark


def transform(x, inverse=False):
    y=x.copy()
    for bit in range(8):
        stride=1<<bit
        view=y.reshape(-1,2,stride,y.shape[-1])
        if inverse:
            view[:,1]-=view[:,0]
        else:
            view[:,1]+=view[:,0]
    return y


def run(out, lib):
    out.mkdir(parents=True,exist_ok=True)
    t,r=actual_input()
    path=ROOT.parent/'external_sources/jiuzhang3_iteration112_independent'
    sources=json.loads((ROOT/'results/iteration112_hardware_v1/sources.json').read_text())
    captures=[]
    for source in sources:
        file=path/source['member']
        with file.open('rb') as f:
            assert hashlib.file_digest(f,'sha256').hexdigest()==source['sha256']
        a=np.memmap(file,np.uint8,'r').reshape(-1,144)
        assert len(a)==source['rows']
        reverse=np.unpackbits(np.arange(256,dtype=np.uint8)[:,None],axis=1)@(1<<np.arange(8))
        captures.append(np.array([np.bincount(a[:,j],minlength=256)[reverse] for j in range(144)]))
    cap=80
    grouped=[]
    words=[]
    tail=[]
    checks=[]
    pop=np.array([int(w).bit_count() for w in range(256)])
    moment_check=[]
    start=perf_counter()
    for port in range(144):
        xs=optical_x(t[8*port:8*port+8],r)
        a,b=[transform(sectors(lib,x,cap)[2]) for x in xs]
        allclick=transform((a[:,-1]*b[:,-1])[:,None],True).ravel()
        nfft=256
        g=np.fft.irfft(np.fft.rfft(a[:,:cap+1],nfft)*np.fft.rfft(b[:,:cap+1],nfft),nfft)[:,:cap+1]
        joint=transform(g,True)
        mask=pop[:,None]>np.arange(cap+1)
        checks.append([float(abs(joint[mask]).sum()),float(-np.minimum(joint,0).sum()),
                       float(abs(joint.sum(1)-allclick).sum())])
        joint[mask]=0
        joint=joint.clip(0)
        joint/=joint.sum()
        words.append(joint.sum(1))
        grouped.append(np.array([joint[pop==h].sum(0) for h in range(9)]))
        ev=np.concatenate([np.linalg.eigvalsh(x) for x in xs])
        scale=np.linspace(1+1e-6,1+.99/ev.max(),256)
        tail.append(float(np.exp(np.min(-.5*np.log1p((1-scale[:,None])*ev).sum(1)-(cap+1)*np.log(scale)))))
        moment_check.append(float(abs(joint.sum(0)@np.arange(cap+1)-ev.sum()/2)))
    setup=perf_counter()-start
    law=np.array(grouped);words=np.array(words);h,n=np.indices((9,cap+1))
    missing=np.einsum('ihn,hn->i',law,n-h)
    collided=law[:,n>h].sum(1)
    mean_n=float(np.einsum('ihn,hn->',law,n))
    hist=np.array(captures);prob=hist/hist.sum(2,keepdims=True)
    empirical=np.stack([prob[:,:,pop==i].sum(2) for i in range(9)],axis=-1)
    distance=abs(empirical-law.sum(2)).sum(2)/2
    word_tv=abs(prob-words).sum(2)/2
    ids=((np.arange(256)[:,None]>>np.arange(8))&1).astype(np.uint8)
    direct,_=benchmark(lib,words[34],ids,2)
    values=np.column_stack((h.ravel(),n.ravel())).astype(np.uint8)
    marked,_=benchmark(lib,law[34].ravel(),values,3)
    assert max(moment_check)<1e-8 and max(tail)<1e-9 and np.max(checks)<1e-8
    summary=dict(ports=144,bins_per_port=8,cap=cap,files=[s['file'] for s in sources],
        shots=[s['rows'] for s in sources],setup_seconds=setup,
        mean_true_total=mean_n,mean_fired_bins=mean_n-float(missing.sum()),
        mean_missed_total=float(missing.sum()),missed_fraction=float(missing.sum()/mean_n),
        per_port_collision_probability_mean=float(collided.mean()),
        per_port_collision_probability_range=[float(collided.min()),float(collided.max())],
        average_spatial_number_tv=distance.mean(1).tolist(),max_spatial_number_tv=distance.max(1).tolist(),
        average_eight_bin_word_tv=word_tv.mean(1).tolist(),max_eight_bin_word_tv=word_tv.max(1).tolist(),
        between_capture_avg_word_tv=float(abs(prob[0]-prob[1]).sum(1).mean()/2),
        between_capture_avg_number_tv=float(abs(empirical[0]-empirical[1]).sum(1).mean()/2),
        max_tail=max(tail),max_joint_checks=np.max(checks,axis=0).tolist(),
        max_mean_number_error=max(moment_check),one_bank_direct=direct,one_bank_joint=marked)
    np.savez_compressed(out/'banks.npz',number_bins=law,words=words,hardware=hist,tail=tail)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    parser.add_argument('--library',type=Path,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=2):
        run(args.output,library(args.library))
