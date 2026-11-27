"""Two readout response models with identical complete bank-occupancy predictions."""
import argparse
import json
import math
import numpy as np
from threadpoolctl import threadpool_limits
from iteration120_calibration import OUT,CACHE,BANKS,PAIRS,PAIR_INDEX,observations
from iteration119_hardware import DATA,digest


def kernels(z,target,mult):
    # Either response leaves an empty input bank empty with probability a.
    a=target/z
    if not np.all((a>=0)&(a<=1)):raise ValueError('empty-bank filling requires nonnegative excess occupancy')
    fill=np.broadcast_to(np.eye(9),(144,9,9)).copy()
    fill[:,0,0]=a
    fill[:,0,1:]=(1-a[:,None])*mult[:,1:]/(1-z[:,None])
    noise=np.zeros_like(fill);survive=a**(1/8)
    for k in range(9):
        for j in range(k,9):
            noise[:,k,j]=math.comb(8-k,j-k)*(1-survive)**(j-k)*survive**(8-j)
    assert max(abs(x.sum(2)-1).max() for x in (fill,noise))<1e-13
    return a,noise,fill


def freeze():
    with np.load(OUT/'calibration_predictions.npz') as f:prior={k:f[k] for k in f}
    p=prior['published_multiplicity'];z=p[:,0];target=prior['adjusted_multiplicity'][:,0]
    a,noise,fill=kernels(z,target,p)
    left,right=PAIR_INDEX
    # All complete occupancy-word probabilities follow the same local channel
    # under both response mechanisms, including correlations among any banks.
    w=prior['published_witness'].copy()
    for j,bank in enumerate(BANKS):
        view=w.reshape(-1,2,1<<j)
        view[:,1]+=(1-a[bank])*view[:,0];view[:,0]*=a[bank]
    old_cov=prior['published_both']-(1-z[left])*(1-z[right])
    both=a[left]*a[right]*old_cov+(1-target[left])*(1-target[right])
    out=dict(empty_survival=a,poisson_mean_per_port=-np.log(a)/8,**prior)
    for name,mat in [('poisson',noise),('refill',fill)]:
        out[name+'_multiplicity']=np.einsum('ik,ikj->ij',p,mat)
        out[name+'_both']=both;out[name+'_witness']=w
        out[name+'_pairs']=np.stack([(mat[i].T@q.reshape(9,9)@mat[j]).ravel()
            for q,(i,j) in zip(prior['published_pairs'],PAIRS)])
    path=OUT/'response_predictions.npz';np.savez_compressed(path,**out)
    protocol=dict(training_file=10,development_files=[14,15],fresh_file=16,prediction_sha256=digest(path),
        models=['published','adjusted','poisson','refill'],
        parameters='Each additional model uses the same 144 training-file bank vacuum frequencies. Poisson adds independent equal-rate background counts in the eight ports. Refill replaces some empty banks with a draw from the nominal nonempty bank distribution.',
        primary='Median across the 144 banks of empirical conditional-multiplicity TV on capture 16.',
        secondary='All previously frozen metrics: maximum conditional multiplicity TV, connected RMS, three joint multiplicity TVs, 256-word TV and the round-119 event.',
        scope='Independent test of previously developed models on one new acquisition; finite-file descriptive metrics.')
    (OUT/'response_protocol.json').write_bytes((json.dumps(protocol,indent=2)+'\n').encode());print(protocol)


def evaluate():
    from iteration112_hardware_files_v1 import run
    proto=json.loads((OUT/'response_protocol.json').read_bytes())
    path=OUT/'response_predictions.npz';assert digest(path)==proto['prediction_sha256']
    with np.load(path) as f:pred={k:f[k] for k in f}
    where=CACHE.parent/'jiuzhang3_iteration120_response'
    run(where,numbers=[proto['fresh_file']])
    src=json.loads((where/'files.json').read_bytes())[0];path=where/src['member'];assert digest(path)==src['sha256']
    raw=np.memmap(path,np.uint8,'r').reshape(-1,144)
    mult,both,pairs,witness=observations(raw);a,b=PAIR_INDEX
    event=json.loads((DATA/'iteration119_weights/hardware_protocol.json').read_bytes())['event_words']
    table={}
    for name in proto['models']:
        m,q,p,w=[pred[name+'_'+k] for k in ('multiplicity','both','pairs','witness')]
        conditional=.5*abs(m[:,1:]/(1-m[:,:1])-mult[:,1:]/(1-mult[:,:1])).sum(1)
        actual=both[a,b]-(1-mult[a,0])*(1-mult[b,0]);model=q-(1-m[a,0])*(1-m[b,0])
        table[name]=dict(conditional_multiplicity_tv_median=float(np.median(conditional)),conditional_multiplicity_tv_max=float(conditional.max()),
          connected_rms=float(np.sqrt(np.mean((model-actual)**2))),joint_pair_tv=(.5*abs(p-pairs).sum(1)).tolist(),
          eight_bank_tv=float(.5*abs(w-witness).sum()),fixed_event_error=float(abs(w[event].sum()-witness[event].sum())))
    result=dict(file=src['file'],rows=len(raw),sha256=src['sha256'],metrics=table)
    (OUT/'response_holdout.json').write_text(json.dumps(result,indent=2)+'\n');print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','evaluate'))
    with threadpool_limits(limits=2):
        (freeze if p.parse_args().stage=='freeze' else evaluate)()
