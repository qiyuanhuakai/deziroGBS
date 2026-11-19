"""Fixed-family quality analysis; failures are target-zero outcomes, never refilled.
Nominal bounds condition on iid and fixed numerical references. Address domain
separation repairs stream reuse; it does not prove mathematical iid.
"""
import math
from scipy.stats import beta as beta_distribution
from iteration93_common_v5 import *
from iteration92_quality_v1 import confidence

def failure_upper(k,n,delta):return 1. if k==n else float(beta_distribution.ppf(1-delta,k+1,n-k))

def analyze(job,raw,plan,out):
    j=job['case'];z=np.load(ROOT/f'results/iteration93_reference_v2/law_{j}.npz');meta=read(ROOT/f'results/iteration93_reference_v2/row_{j}.json');counts=raw.get('counts',raw.get('clicks'));errors=raw['failed'].astype(bool);N=len(counts);assert N==job['records'];pilot=plan['pilot_records'];delta=plan['confidence']['failure_budget']/plan['confidence']['events'];m=counts.shape[1];cl=sum((counts[:,i]>0).astype(int)<<i for i in range(m));cl[errors]=1<<m;click,ch=confidence(cl,np.r_[z['click'],0.],N,pilot,delta);hist={'click_'+k:v for k,v in ch.items()};result=dict(job=job,failures=int(errors.sum()),click=click)
    if 'counts' in raw:
        if j in (0,1):
            keys=z['keys'];packed=sum(counts[:,i].astype(np.uint64)<<(8*i) for i in range(m));pos=np.searchsorted(keys,packed);inside=pos<len(keys);inside[inside]&=keys[pos[inside]]==packed[inside];labels=np.full(N,len(keys),int);labels[inside]=pos[inside];target=np.r_[z['probabilities'],float(meta['total_tail']),0.];tail=float(meta['tail_chernoff'])
        elif j==3:
            inside=(counts<=64).all(1);labels=np.full(N,65**2,int);labels[inside]=counts[inside,0]*65+counts[inside,1];target=np.r_[z['pnr'].ravel(),float(meta['pnr_tail']),0.];tail=float(meta['tail_chernoff'])
        if j!=2:
            labels[errors]=len(target)-1;pnr,ph=confidence(labels,target,N,pilot,delta,tail);result['pnr']=pnr;hist.update({'pnr_'+k:v for k,v in ph.items()})
        else:
            values=counts.sum(1);values[errors]=2**31;ecdf=np.array([(values<=k).mean() for k in range(65)]);reference=np.cumsum(z['total']);distance=float(np.max(abs(ecdf-reference)));radius=math.sqrt(math.log(2/delta)/(2*N));result['total_cdf']=dict(empirical_distance=distance,lower=max(0.,distance-radius),upper=min(1.,distance+radius+float(meta['tail_chernoff'])),radius=radius);hist.update(total_empirical_cdf=ecdf,total_reference_cdf=reference)
    if job['family']=='mps':
        failure=failure_upper(int(errors.sum()),N,delta);result.update(failure_probability_upper=failure,ideal_arithmetic_compression_tv_upper=min(1.,job['budget']+failure),compression_bound_scope='Analytic truncation model plus observed-failure confidence event; not an interval-certified global numerical implementation bound.');result['maximum_charged']=float(raw['statistics'][:,5].max());result['maximum_bond']=int(raw['statistics'][:,8].max())
    np.savez_compressed(out/'histogram.npz',**hist);save(out,'quality.json',result);return result
