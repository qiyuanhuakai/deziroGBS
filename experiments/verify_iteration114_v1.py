"""Check both PNR kernels on complete finite sectors and audit the direct-click optical law."""
import argparse
import ctypes as ct
import hashlib
import json
import math
import os
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration114_rank_v1 import ROOT,P,I,J,U,BLOCK,optical,model,libopen,draw
from iteration114_pairs_v1 import prepare,sector
from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x

DATA=ROOT/'results'


def exact_sector(t,lam,n,m):
    # In creation operators the pure fixed sector is the kth power of a quadratic.
    b=(t*lam)@t.T
    d=len(t);poly={(0,)*d:1.}
    for step in range(n//2):
        new={}
        for key,v in poly.items():
            for i in range(d):
                for j in range(i+1):
                    word=list(key);word[i]+=1;word[j]+=1;word=tuple(word)
                    new[word]=new.get(word,0)+v*b[i,j]*(1 if i==j else 2)
        poly=new
    out=np.zeros((n+1)**m)
    for key,a in poly.items():
        code=sum(key[j]*(n+1)**j for j in range(m))
        out[code]+=abs(a)**2*math.prod(math.factorial(y) for y in key)
    return out/out.sum()


def native_sectors(lib,other):
    t,s=optical(3,2);obj=model(t,s,6,9)
    sp,_,_=prepare(t,s[None],6);args=sp[0]
    checks=[];shots=131072
    for n in (4,6):
        cdf=np.zeros(7);cdf[n:]=1.
        obj['pnr'][0]=cdf;args[0]=cdf
        exact=exact_sector(args[1],args[2],n,3)
        for method in ('fock','pairs'):
            buf=np.empty((shots,3),np.uint8)
            if method=='fock':draw(lib,obj,True,shots,8,114910000+n,0,buf)
            else:
                counters=np.zeros(2,np.uint64)
                assert other.pairs114(shots,3,5,2,6,BLOCK,8,114920000+n,0,0,
                    *[a.ctypes.data for a in args],buf.ctypes.data,counters.ctypes.data)==0
            hist=np.bincount(buf@((n+1)**np.arange(3)),minlength=len(exact))
            assert np.all(hist[exact==0]==0)
            tv=float(abs(hist/shots-exact).sum()/2)
            sampling_budget=float(np.sqrt(exact*(1-exact)/shots).sum()/2)
            assert tv<sampling_budget+4/np.sqrt(shots)
            checks.append(dict(n=n,method=method,shots=shots,full_sector_empirical_tv=tv,
                               expected_tv_upper=sampling_budget))
    return checks


def independent_direct(r):
    t,s=optical(13,r);m=model(t,s,24 if r==2 else 16,25 if r==2 else 49)
    _,p,proposal,_,_,_=m['click'];info=m['info']
    # Positive branch amplitudes are recovered from the normalized proposal CDF.
    c=np.diff(proposal,prepend=0.)/np.sqrt(info['acceptance'])
    gram=np.exp(-.5*((p[:,None]-p[None])**2).sum(2))*np.outer(c,c)
    law=np.load(DATA/f'iteration114_positive_v1/m13_r{r}.npz')['projected_click']
    masks=np.arange(8192);worst=0.
    rng=np.random.default_rng(114902+r)
    for mask in [0,8191,*rng.integers(1,8191,48)]:
        ids=np.flatnonzero((mask>>np.arange(13))&1)
        z=t[ids]@p.T
        vacuum=np.sum(gram*np.exp(-(z.conj().T@z))).real
        worst=max(worst,abs(float(vacuum-law[(masks&mask)==0].sum())))
    assert worst<1e-10
    return worst


def pair_diagnostic(row,path):
    s=np.load(path)['s'];av=0;var=0
    for gain in s:
        mass,p=sector(gain,row['cap_per_spectrum']);s2=sum(np.tanh(gain)**2)
        a=[math.factorial(2*k)*p[k]/(p[0]*math.prod(range(1,2*k,2))**2*s2**k) for k in range(len(p))]
        one=sum(p[1:]/mass/a[1:]);av+=one
        var+=sum(p[1:]/mass*(2-np.array(a[1:]))/np.array(a[1:])**2)-one**2
    z=(row['attempts_per_shot']-av)/np.sqrt(var/row['shots'])
    assert abs(z)<5
    return float(z)


def hardware_check():
    mat=np.load(DATA/'iteration114_saturation_v1/covariances.npz')['covariance']
    t,r=actual_input()
    errors=[]
    for a,b in [(34,35),(1,143),(50,70)]:
        delta=0.
        for i in range(8*a,8*a+8):
            for j in range(8*b,8*b+8):
                x=optical_x(t[[i,j]],r)
                def fn(u,v):
                    w=np.tile([1-u,1-v],2)
                    return np.exp(-.5*np.log(np.linalg.det(np.eye(4)+x*w)).sum())
                step=1e-6
                delta+=fn(1+1j*step,1).imag/step*fn(1.,0.).real-fn(1+1j*step,0).imag/step
        errors.append(float(abs(mat[2,a,b]-delta)))
    assert max(errors)<1e-9
    return errors


def run(out,rank,pairs):
    out.mkdir(exist_ok=True,parents=True)
    ranklib=libopen(rank);other=ct.CDLL(str(pairs.resolve()))
    other.pairs114.argtypes=[I]*7+[U,J,I]+[P]*7
    independent=dict(fixed_emission=native_sectors(ranklib,other),
        max_projected_vacuum_difference=[independent_direct(r) for r in (2,4)],
        mixed_bank_covariance_abs_error=hardware_check())
    small=[];wide=[];paired=[]
    for f in sorted((DATA/'iteration114_positive_v1').glob('*.json')):
        row=json.loads(f.read_text());data=np.load(f.with_suffix('.npz'))
        for key in ('sample_0','sample_1'):
            assert data[key].shape[1]==row['m'] and data[key].dtype==np.uint8
        assert data['sample_0'].max()<=1
        entry={k:row[k] for k in ('m','r','cap','shots','batch','persistent_pnr_bytes',
              'persistent_click_bytes','pnr_setup_s','click_setup_s','pnr_full_tv_upper','click_projection')}
        entry['eight_thread']=[{k:v for k,v in a.items() if k!='times'} for a in row['measurements'] if a['threads']==8]
        if row['m']==13:entry['direct_model_tv']=row['direct_model_tv'];small.append(entry)
        else:wide.append(entry)
    for f in sorted((DATA/'iteration114_pairs_v1').glob('*.json')):
        row=json.loads(f.read_text())
        row['attempts_z']=pair_diagnostic(row,f.with_suffix('.npz'));row.pop('times')
        paired.append(row)
    precise=[]
    import mpmath as mp
    for f in sorted((DATA/'iteration114_precise_v1').glob('*.json')):
        row=json.loads(f.read_text())
        r=row['r'];t,gains=optical(13,r)
        config=model(t,gains,24 if r==2 else 16,row['budget'],'hermite' if r==2 else 'positive')
        pos=config['click'][1];info=config['info']
        c=np.diff(config['click'][2],prepend=0.)/np.sqrt(info['acceptance'])
        with mp.workdps(55):
            nodes=[[mp.mpf(a) for a in v] for v in pos]
            weights=list(map(mp.mpf,c))
            norm=mp.fsum(weights[i]*weights[j]*mp.exp(-sum((a-b)**2 for a,b in zip(u,v))/2)
                           for i,u in enumerate(nodes) for j,v in enumerate(nodes))
            overlap=mp.fsum(w*mp.exp(-sum(vv**2*(1-mp.tanh(g)) for vv,g in zip(v,gains))/2)
                               for w,v in zip(weights,nodes))/mp.sqrt(mp.fprod(mp.cosh(g) for g in gains))
            bound=float(mp.sqrt(1-overlap**2/norm))
        assert abs(bound-row['click_projection']['state_bound'])<5e-10
        precise.append(dict(m=row['m'],r=r,branches=info['branches'],click_full_tv_upper=bound,
            direct_model_tv=row.get('direct_model_tv'),click_setup_s=row['click_setup_s'],
            click_us=row['measurements'][1]['us'],pnr_us=row['measurements'][3]['us']))
    cache=json.loads((DATA/'iteration114_cache_v1/summary.json').read_text())
    for row in cache:row.pop('times')
    assert len(cache)==6 and len(precise)==4
    sat=json.loads((DATA/'iteration114_saturation_v1/summary.json').read_text())
    peers=json.loads((DATA/'iteration114_peers_v1/summary.json').read_text())
    for v in peers:v.pop('times')
    summary=dict(small=small,wide=wide,precise=precise,cached_click=cache,pair_sampler=paired,peers=peers,
                high_power=sat,independent_checks=independent,jiuzhang3_full_record_reference_us=1.27)
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    files=sorted(f for f in ROOT.glob('results/iteration114*/*') if f.is_file() and f.name!='manifest.json')
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf8')
    print(independent,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    p.add_argument('--rank',type=Path,required=True);p.add_argument('--pairs',type=Path,required=True)
    a=p.parse_args()
    with threadpool_limits(limits=2):run(a.output,a.rank,a.pairs)
