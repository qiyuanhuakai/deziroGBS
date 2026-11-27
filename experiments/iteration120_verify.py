"""Verify the scientific release, stored record decisions and manuscript sources."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/iteration120_contractions'
PAPER=ROOT/'papers/number-tilted-gbs'
MANIFEST=OUT/'manifest.json'

def read(path):return json.loads(path.read_bytes())

def digest(path):
    data=path.read_bytes()
    if path.suffix in {'.py','.json','.tex','.md'}:data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()

def sources():
    with ZipFile(PAPER/'latex-source.zip') as z:
        assert z.testzip() is None
        paperfiles=[PAPER/n for n in z.namelist()]
        for p in paperfiles:
            a=p.read_bytes();b=z.read(p.name)
            if p.suffix=='.tex':a=a.replace(b'\r\n',b'\n');b=b.replace(b'\r\n',b'\n')
            assert a==b,p
    paths=[*sorted((ROOT/'experiments').glob('iteration120_*.py')),ROOT/'experiments/iteration119_clamps.py',
           *sorted(OUT.glob('*.json')),*sorted(OUT.glob('*.npz')),ROOT/'results/iteration83_loss_pilot_v1/input.npz',
           PAPER/'manuscript.pdf',PAPER/'README.md',PAPER/'latex-source.zip',*paperfiles]
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths if p!=MANIFEST}

def verify():
    proof=read(OUT/'interval_optical.json')
    assert proof['optical_input_sha256']==digest(ROOT/'results/iteration83_loss_pilot_v1/input.npz')
    assert proof['certified_full_law_upper']=='0.004815'
    old=read(ROOT/'results/iteration119_weights/clamped_bands_64.json')['decisions']
    comparisons=read(OUT/'replay_64_2.json')
    near=read(OUT/'decision_checks.json');assert len(near)==16
    with np.load(OUT/'response_samples.npz') as post:
        for r,name in enumerate(('pnr','click')):
            compared=comparisons[r];oldrows=[a for a in old if a['direct']==bool(r)]
            assert len(compared['rows'])==len(oldrows)==64
            for a,b in zip(oldrows,compared['rows'],strict=True):
                assert (a['index'],a['stages'],a['accepted'])==(b['index'],b['stages'],b['accepted'])
                assert np.allclose(a['weights'],b['weights'],rtol=1e-7,atol=1e-10)
            with np.load(OUT/f'base_{r}_256.npz') as base,np.load(OUT/f'full_{r}_256.npz') as full:
                a,b=[base[f'row{r}_{j}'].astype(np.int64) for j in range(2)]
                assert a.shape==b.shape==(256,1152) and a.min()>=0 and b.min()>=0
                assert np.array_equal(base[name],(a|b) if r else (a+b))
                for j in range(2):
                    log=base[f'log{r}_{j}']
                    assert np.array_equal(log[:,0],2**20+np.arange(len(log)))
                    assert np.all((log[:,2]>=0)&(log[:,2]<=1+3e-12))
                    assert np.array_equal(log[:,3],[np.random.default_rng([119,82,r,j,int(i)]).random() for i in log[:,0]])
                    assert np.sum(log[:,2]>log[:,3])>=256
                data=read(OUT/f'full_{r}_256.json');keep=[]
                for i,row in enumerate(data['decisions']):
                    assert row['direct']==bool(r) and row['index']==i
                    t=np.random.default_rng([120,92,0,r,i]).random()*4
                    assert row['threshold']==t
                    weights=row['weights'];assert all(np.isfinite(v) and v>=0 for v in weights)
                    if t<.5:assert len(weights)==0;decision=True
                    elif t<.8*weights[0]:assert len(weights)==1;decision=True
                    elif t>=1.25*weights[0]:assert len(weights)==1;decision=False
                    else:assert len(weights)==2;decision=t<weights[1]
                    assert row['accepted']==decision and row['stages']==len(weights)
                    if decision:keep.append(i)
                assert len(data['decisions'])==256 and len(keep)==data['accepted']==(69 if r else 70)
                assert np.array_equal(full['accepted_indices'],keep) and np.array_equal(full[name],base[name][keep])
                for row in near:
                    if row['direct']==bool(r):
                        a=data['decisions'][row['index']];assert a['weights']==row['reference']
                        assert a['accepted']==row['accepted'] and len(row['compensated'])==a['stages']
                for model in ('refill','poisson'):
                    arr=post[model+'_'+name];assert arr.shape==full[name].shape and np.all(arr>=full[name])
                    if r:assert arr.max()==1
    for name,key in [('calibration','sha256_predictions'),('response','prediction_sha256')]:
        assert read(OUT/(name+'_protocol.json'))[key]==digest(OUT/(name+'_predictions.npz'))
    with np.load(OUT/'response_predictions.npz') as f:
        assert np.array_equal(f['poisson_witness'],f['refill_witness'])
        assert np.allclose(f['refill_both'],f['poisson_both'],rtol=0,atol=1e-15)
    coverage=read(OUT/'hardware_panel_bounds.json')
    for row,cap in zip(coverage['panels']['poisson'],read(OUT/'panel_certificate.json'),strict=True):
        assert abs(row['average_conditional_optical_tv_upper99']+cap['full_panel_emitted_tv_upper']-row['emitted_to_average_conditional_upper99'])<1e-14
    assert [x['file'] for x in read(OUT/'calibration_holdout.json')]==[14,15]
    assert read(OUT/'response_holdout.json')['rows']==coverage['rows']==1476267
    assert max(r['compensated_absolute_error'] for r in read(OUT/'contraction_checks.json'))<2e-11

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');args=ap.parse_args()
    verify();hashes=sources()
    if args.write:MANIFEST.write_bytes((json.dumps({'sha256':hashes},indent=2)+'\n').encode())
    assert read(MANIFEST)['sha256']==hashes
    print('Verified',len(hashes),'artifacts, 128 previous decisions, 512 fresh candidates and 139 delivered optical records.')
