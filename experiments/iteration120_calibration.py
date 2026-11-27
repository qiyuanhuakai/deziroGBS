"""Fit only one-bank vacuum probabilities, then freeze higher-order predictions."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
from threadpoolctl import threadpool_limits
from iteration114_rank_v1 import cov
from iteration115_visible_v1 import hardware,ROOT,DATA
from iteration119_hardware import digest,patterns
from iteration119_powers import retained

OUT=DATA/'iteration120_contractions'
CACHE=ROOT.parent/'external_sources/jiuzhang3_iteration120_holdout'
BANKS=np.array([2,54,55,88,89,98,125,140])
PAIRS=[(34,35),(71,72),(102,143)]
BITS=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
PAIR_INDEX=np.triu_indices(144,1)


def observations(raw):
    n=len(raw)
    mult=np.zeros((144,9),np.int64)
    both=np.zeros((144,144))
    pair=np.zeros((len(PAIRS),81),np.int64)
    for off in range(0,n,8192):
        b=raw[off:off+8192];h=BITS[b]
        mult+=np.bincount((h+9*np.arange(144)).ravel(),minlength=144*9).reshape(144,9)
        fire=(b!=0).astype(float);both+=fire.T@fire
        for k,(i,j) in enumerate(PAIRS):pair[k]+=np.bincount(9*h[:,i]+h[:,j],minlength=81)
    return mult/n,both/n,pair/n,patterns(raw,BANKS)/n


def axes(sites):
    return np.concatenate([sites,sites+1152],axis=-1)


def vacancy(X,idx):
    size=idx.shape[-1];v=1.
    for A in X:
        sign,log=np.linalg.slogdet(np.eye(size)+A[idx[...,None],np.expand_dims(idx,-2)])
        if np.any(sign<=0):raise ArithmeticError('nonpositive vacuum determinant')
        v=v*np.exp(-.5*log)
    return v


def words(X,sites):
    # Each entry in sites is one detector or a group of eight physical outputs.
    d=len(sites);masks=np.arange(1<<d);bits=(masks[:,None]>>np.arange(d))&1
    empty=np.ones(len(masks))
    for k in range(1,d+1):
        selected=masks[bits.sum(1)==k]
        for offset in range(0,len(selected),128):
            chunk=selected[offset:offset+128]
            cols=np.where(bits[chunk])[1].reshape(len(chunk),k)
            idx=axes(sites[cols].reshape(len(chunk),-1))
            empty[chunk]=vacancy(X,idx)
    law=empty[::-1].astype(np.longdouble)
    for j in range(d):
        u=law.reshape(-1,2,1<<j);u[:,1]-=u[:,0]
    assert law.min()>-2e-10 and abs(law.sum()-1)<1e-12
    return np.array(law,float)


def predict(t,ss):
    X=[cov(t,s) for s in ss];all_sites=np.arange(1152).reshape(144,8)
    mult=np.stack([np.bincount(BITS,weights=words(X,sites[:,None]),minlength=9) for sites in all_sites])
    first,second=PAIR_INDEX;both=np.zeros(len(first))
    for i in range(0,len(first),128):
        s=slice(i,i+128);idx=axes(np.column_stack([all_sites[first[s]],all_sites[second[s]]]))
        both[s]=1-mult[first[s],0]-mult[second[s],0]+vacancy(X,idx)
    pair=[]
    code=9*BITS[np.arange(65536)%256]+BITS[np.arange(65536)//256]
    for a,b in PAIRS:
        law=words(X,all_sites[[a,b]].reshape(16,1))
        pair.append(np.bincount(code,weights=law,minlength=81))
    return mult,both,np.array(pair),words(X,all_sites[BANKS])


def fit(t,ss,observed):
    gains=[]
    for bank,pzero in enumerate(observed):
        rows=t[8*bank:8*bank+8]
        values=np.concatenate([np.linalg.eigvalsh(cov(rows,s)) for s in ss])
        def error(x):return -.5*np.log1p(x*values).sum()-np.log(pzero)
        hi=1.
        while error(hi)>0:
            hi*=1.1
            if min(1+hi*values)<=0:raise ArithmeticError('vacuum probability not bracketed')
        gains.append(brentq(error,0,hi,xtol=1e-13))
    return np.array(gains)


def freeze():
    OUT.mkdir(exist_ok=True)
    f=json.loads((DATA/'iteration112_hardware_v1/sources.json').read_bytes())[0]
    path=ROOT.parent/'external_sources/jiuzhang3_iteration112_independent'/f['member']
    assert f['file']==10 and digest(path)==f['sha256']
    raw=np.memmap(path,dtype=np.uint8,mode='r').reshape(-1,144)
    observed=observations(raw)[0][:,0]
    t,ss=hardware();gains=fit(t,ss,observed)
    adjusted=t*np.sqrt(np.repeat(gains,8))[:,None]
    max_singular=float(np.linalg.svd(adjusted,compute_uv=False)[0])
    assert max_singular<1
    arrays={'gains':gains}
    for name,mat in [('published',t),('adjusted',adjusted)]:
        for key,arr in zip(('multiplicity','both','pairs','witness'),predict(mat,ss)):arrays[name+'_'+key]=arr
    err=float(abs(arrays['adjusted_multiplicity'][:,0]-observed).max());assert err<1e-11
    target=OUT/'calibration_predictions.npz'
    np.savez_compressed(target,**arrays)
    protocol=dict(training_file=10,training_sha256=f['sha256'],holdouts=[14,15],banks=BANKS.tolist(),bank_pairs=PAIRS,
        fit='144 positive row-intensity multipliers fixed by the probabilities of zero clicks in each bank on capture 10.',
        sha256_predictions=digest(target),adjusted_transfer_max_singular=max_singular,
        max_training_vacuum_residual=err,range_intensity_gain=[float(gains.min()),float(gains.max())],
        metrics=['median and maximum TV of within-bank multiplicities conditional on a nonempty bank',
          'RMS error of connected occupancies over all 10296 unordered bank pairs',
          'complete joint multiplicity TV on three predeclared bank pairs',
          'complete TV on the archived eight-bank occupancy words and absolute error on the round-119 event'],
        scope='Descriptive distances to each complete held-out file. No hypothesis of independent device records.')
    (OUT/'calibration_protocol.json').write_bytes((json.dumps(protocol,indent=2)+'\n').encode())
    print(protocol,flush=True)


def evaluate():
    from iteration112_hardware_files_v1 import run
    proto=json.loads((OUT/'calibration_protocol.json').read_bytes())
    path=OUT/'calibration_predictions.npz';assert digest(path)==proto['sha256_predictions']
    run(CACHE,numbers=proto['holdouts'])
    with np.load(path) as archive:pred={k:archive[k] for k in archive}
    event=json.loads((DATA/'iteration119_weights/hardware_protocol.json').read_bytes())['event_words']
    results=[]
    for f in json.loads((CACHE/'files.json').read_bytes()):
        path=CACHE/f['member'];assert digest(path)==f['sha256']
        raw=np.memmap(path,dtype=np.uint8,mode='r').reshape(-1,144)
        mult,both,pairs,witness=observations(raw);a,b=PAIR_INDEX
        info=dict(file=f['file'],rows=len(raw),sha256=f['sha256'])
        for name in ('published','adjusted'):
            m,q,p,w=[pred[name+'_'+k] for k in ('multiplicity','both','pairs','witness')]
            conditional=.5*abs(m[:,1:]/(1-m[:,:1])-mult[:,1:]/(1-mult[:,:1])).sum(1)
            empirical=both[a,b]-(1-mult[a,0])*(1-mult[b,0])
            model=q-(1-m[a,0])*(1-m[b,0])
            info[name]=dict(conditional_multiplicity_tv_median=float(np.median(conditional)),
              conditional_multiplicity_tv_max=float(conditional.max()),connected_rms=float(np.sqrt(np.mean((model-empirical)**2))),
              joint_pair_tv=(.5*abs(p-pairs).sum(1)).tolist(),eight_bank_tv=float(.5*abs(w-witness).sum()),
              fixed_event_error=float(abs(w[event].sum()-witness[event].sum())))
        results.append(info);print(info,flush=True)
    (OUT/'calibration_holdout.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','evaluate'))
    with threadpool_limits(limits=2):
        (freeze if parser.parse_args().stage=='freeze' else evaluate)()
