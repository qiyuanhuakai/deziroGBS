"""Simultaneous, time-uniform panel TV limits for the held-out acquisition."""
import json
import math
import numpy as np
from iteration120_calibration import OUT,ROOT,DATA,CACHE,PAIRS
from iteration119_hardware import digest


def hist(path):
    # Direct popcount, independent of the byte-lookup counting implementation.
    raw=np.memmap(path,np.uint8,'r').reshape(-1,144)
    counts=np.zeros((3,81),np.int64);banks=np.asarray(PAIRS).ravel()
    for off in range(0,len(raw),8192):
        block=np.ascontiguousarray(raw[off:off+8192,banks])
        pop=np.unpackbits(block[:,:,None],axis=2).sum(2,dtype=np.int64)
        for i in range(3):counts[i]+=np.bincount(9*pop[:,2*i]+pop[:,2*i+1],minlength=81)
    assert np.all(counts.sum(1)==len(raw))
    return counts,len(raw)


def radius(log_budget,design_horizon,n):
    # Ville's inequality for conditional Hoeffding with a fixed positive bet.
    lam=math.sqrt(8*log_budget/design_horizon)
    return lam/8+log_budget/(n*lam)


def run():
    check=json.loads((OUT/'panel_law_checks.json').read_bytes())
    eps=2*float(check['tv_rounding_bound'])
    caps=[row['full_panel_emitted_tv_upper'] for row in json.loads((OUT/'panel_certificate.json').read_bytes())]
    f=json.loads((DATA/'iteration112_hardware_v1/sources.json').read_bytes())[0]
    path=ROOT.parent/'external_sources/jiuzhang3_iteration112_independent'/f['member'];assert digest(path)==f['sha256']
    train,design=hist(path)
    where=CACHE.parent/'jiuzhang3_iteration120_response';f=json.loads((where/'files.json').read_bytes())[0]
    path=where/f['member'];assert digest(path)==f['sha256']
    observed,n=hist(path);hat=observed/n
    with np.load(OUT/'response_predictions.npz') as f:
        outputs={}
        for name in ('published','adjusted','poisson','refill'):
            records=[]
            for j,q in enumerate(f[name+'_pairs']):
                # One half of the error budget covers every subset of the k
                # largest model cells, for all 81 values of k, all models and panels.
                order=np.argsort(-q,kind='stable');options=[]
                for k in range(1,82):
                    keep=order[:k];tail=q[order[k:]].sum()
                    d=np.maximum(0,q[keep]-hat[j,keep]).sum()
                    rad=radius(math.log(2*12*81/.01)+k*math.log(2),design,n)
                    options.append((float(tail+d+rad+eps),k,rad,float(tail)))
                ub,k,rad,tail=min(options)
                # Other half covers one training-defined directional event per model and panel.
                event=q>train[j]/design
                lr=radius(math.log(2*12/.01),design,n)
                lb=max(0,float(q[event].sum()-hat[j,event].sum()-lr-eps))
                tv=float(np.abs(q-hat[j]).sum()/2)
                row=dict(banks=PAIRS[j],empirical_tv=tv,average_conditional_optical_tv_lower99=lb,
                         average_conditional_optical_tv_upper99=ub,kept_cells=k,
                         upper_radius=rad,model_tail=tail,lower_event_cells=int(event.sum()),lower_radius=lr)
                if name in ('published','refill','poisson'):
                    row.update(emitted_to_finite_file_upper=tv+eps+caps[j],
                               emitted_to_average_conditional_upper99=ub+caps[j],
                               emitted_to_average_conditional_lower99=max(0,lb-caps[j]))
                records.append(row)
            outputs[name]=records
    report=dict(holdout=16,rows=n,design_horizon_from_training=design,confidence=.99,panels=outputs,
                scope='Post-protocol simultaneous sensitivity analysis for all four frozen models and all three specified 81-cell projections. The hardware estimand is the file average of conditional laws; a supermartingale permits stopping and serial dependence.')
    (OUT/'hardware_panel_bounds.json').write_text(json.dumps(report,indent=2)+'\n')
    for name,rows in outputs.items():print(name,[(r['average_conditional_optical_tv_lower99'],r['average_conditional_optical_tv_upper99'],r.get('emitted_to_average_conditional_upper99')) for r in rows])


if __name__=='__main__':run()
