"""Independent covariance, operator bound, and saved-capture checks."""
import json
import numpy as np
import mpmath as mp
from iteration118_auxiliary import prepare
from threadpoolctl import threadpool_limits
from iteration114_rank_v1 import cov
from iteration119_hardware import OUT,CACHE,hardware,digest

def run():
    protocol=json.loads((OUT/'hardware_protocol.json').read_bytes())
    hold=json.loads((OUT/'hardware_holdout.json').read_bytes())
    t,ss=hardware()
    banks=np.asarray(protocol['banks'])
    empty=np.ones(256)
    for mask in range(1,256):
        bank=banks[(mask & (1<<np.arange(8)))!=0]
        sites=(bank[:,None]*8+np.arange(8)).ravel()
        for s in ss:
            mat=np.eye(2*len(sites))+cov(t[sites],s)
            empty[mask]/=np.prod(np.diag(np.linalg.cholesky(mat)))
    # Inclusion-exclusion for each individual word, independent of the fast transform.
    law=np.zeros(256)
    for word in range(256):
        sub=word
        while True:
            law[word]+=(-1)**sub.bit_count()*empty[(255^word)|sub]
            if sub==0:break
            sub=(sub-1)&word
    probability=float(law[protocol['event_words']].sum())
    assert min(law)>0 and abs(law.sum()-1)<1e-12
    assert abs(probability-protocol['optical_probability'])<1e-12
    by_file={f['file']:f for f in json.loads((CACHE/'files.json').read_bytes())}
    decision=np.zeros(256,bool);decision[protocol['event_words']]=True
    anytime=[]
    for rec in hold['holdouts']:
        path=CACHE/by_file[rec['file']]['member']
        assert digest(path)==rec['sha256']
        raw=np.memmap(path,np.uint8,'r').reshape(-1,144)
        n=0
        for offset in range(0,len(raw),16384):
            batch=raw[offset:offset+16384]
            word=np.zeros(len(batch),np.uint8)
            for j,bank in enumerate(banks):word |= (batch[:,bank]>0).astype(np.uint8)<<j
            n+=int(decision[word].sum())
        assert n==rec['event_count'] and len(raw)==rec['record_count']
        length=len(raw)
        # The failure budgets alpha/(2*n*(n+1)) sum to alpha over both files and all lengths.
        radius=np.sqrt(np.log(2*length*(length+1)/protocol['alpha_family'])/(2*length))
        anytime.append(n/length-probability-radius)
    clamp=json.loads((OUT/'clamped_pilot_8.json').read_bytes())
    R=np.array(clamp['retention'])
    # Alternate retention path from the singular values of the actual pair
    # matrices. Multiply independent one-mode pair laws at 65 decimal digits.
    mp.mp.dps=65
    independent=np.ones(3)
    for s,caps in zip(ss,[(7,9,10),(5,6,6)]):
        matrix=prepare(t,s,2)[1]
        prob=[mp.mpf(1)]+[mp.mpf(0)]*10
        for value in np.linalg.svd(matrix,compute_uv=False):
            x=mp.mpf(float(value))**2
            local=[mp.sqrt(1-x)]
            for j in range(1,11):local.append(local[-1]*x*(2*j-1)/(2*j))
            prob=[sum(prob[j]*local[n-j] for j in range(n+1)) for n in range(11)]
        independent*=[float(sum(prob[:k+1])) for k in caps]
    assert max(abs(independent-R))<2e-14
    def positive(m,f):
        x=np.array([np.sqrt(f),np.sqrt(1-f)])
        return np.linalg.eigvalsh(np.outer(x,x)-np.diag([m,0.]))[-1]
    fb,fm=R[:2]/R[-1]
    bound=np.sqrt(1-R[-1])+max(positive(4,fb)+positive(1.25,fm),
                                .5*positive(2,fb)+.8*positive(1.25,fm))
    assert abs(bound-clamp['full_record_tv_upper'])<1e-14
    rng=np.random.default_rng(119986)
    for _ in range(1000):
        p,m,b=rng.dirichlet(np.full(8,.2),3)
        g=np.clip(np.clip(p,.8*m,1.25*m),.5*b,4*b)
        L=np.maximum(0,p-g).sum();E=np.maximum(0,g-p).sum()
        assert .5*np.abs(p-g/g.sum()).sum()<=max(L,E)+1e-14
        assert L<=np.maximum(0,p-4*b).sum()+np.maximum(0,p-1.25*m).sum()+1e-14
        assert E<=np.maximum(0,.5*b-p).sum()+np.maximum(0,.8*m-p).sum()+1e-14
    result=dict(independent_optical_event_probability=probability,independent_operator_bound=bound,
                independent_singular_value_retention=independent.tolist(),
                all_acquisition_lengths_tv_lower99=anytime,
                capture_counts_verified=[x['event_count'] for x in hold['holdouts']],clamp_examples=1000)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(result)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
