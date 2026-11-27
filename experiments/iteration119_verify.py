"""Verify archived files, decisions, optical bounds and the paper's source package."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/iteration119_weights'
PAPER=ROOT/'papers/number-tilted-gbs'
MANIFEST=OUT/'manifest.json'

def read(name):
    return json.loads((OUT/name).read_bytes())

def digest(path):
    data=path.read_bytes()
    # Git canonicalizes CRLF on text checkout. Binary arrays and PDFs are literal.
    if path.suffix in {'.md','.cpp','.py','.json','.tex'}:data=data.replace(b'\r\n',b'\n')
    return hashlib.sha256(data).hexdigest()

def files():
    paths=[*sorted((ROOT/'experiments').glob('iteration119_*.py')),
           ROOT/'experiments/iteration118_auxiliary.py',ROOT/'experiments/iteration112_hardware_files_v1.py',
           ROOT/'native/record_proposal119.cpp',*sorted(OUT.glob('*.npz')),*sorted(OUT.glob('*.json')),
           *sorted(PAPER.glob('*.tex')),*sorted(PAPER.glob('*.pdf')),PAPER/'README.md',PAPER/'latex-source.zip']
    return {p.relative_to(ROOT).as_posix():digest(p) for p in paths if p!=MANIFEST}

def verify():
    stats=read('clamped_bands_64.json')
    assert stats['full_record_tv_upper']<.004824
    assert abs(stats['intrinsic_tv_upper']+max(stats['positive_deficit_upper'],stats['positive_surplus_upper'])-stats['full_record_tv_upper'])<1e-14
    audit=read('audit.json')
    assert np.max(np.abs(np.array(audit['independent_singular_value_retention'])-stats['retention']))<2e-14
    exact=read('exact_bands_64.json')
    decisions=stats['decisions']
    assert len(decisions)==128
    with np.load(OUT/'exact_bands_64.npz') as base,np.load(OUT/'clamped_bands_64.npz') as accepted:
        for r,name in enumerate(('pnr','click')):
            records=base[name]
            a,b=[base[f'row{r}_{j}'].astype(np.int64) for j in (0,1)]
            assert records.shape==(64,1152) and np.array_equal(records,(a|b) if r else (a+b))
            assert records.min()>=0 and (r==0 or records.max()<=1)
            rows=[x for x in decisions if x['direct']==bool(r)]
            assert [x['index'] for x in rows]==list(range(64))
            indices=[]
            for x in rows:
                i=x['index'];u=4*np.random.default_rng([119,92,0,r,i]).random()
                assert u==x['threshold']
                w=x['weights'];assert len(w)==x['stages'] and all(np.isfinite(v) and v>=0 for v in w)
                if not w:assert u<.5;yes=True
                elif len(w)==2:
                    assert .8*w[0]<=u<1.25*w[0];yes=u<np.clip(w[1],.8*w[0],1.25*w[0])
                else:
                    assert u<.8*w[0] or u>=1.25*w[0];yes=u<.8*w[0]
                assert x['accepted']==yes
                if yes:indices.append(i)
            assert np.array_equal(accepted[name],records[indices])
            assert len(indices)==stats['accepted_by_readout'][name]
            for j in (0,1):
                logs=base[f'log{r}_{j}'];assert len(logs)==exact[r]['spectra'][str(j)]['proposals']
                for idx,band,ratio,u,*_ in logs:
                    assert u==np.random.default_rng([119,82,r,j,int(idx)]).random() and 0<=ratio<=1+3e-12
                assert np.sum(logs[:,3]<logs[:,2])>=64
            replay=read('timing_replay.json')[r]
            with np.load(OUT/'exact_bands_8.npz') as prefix:
                assert np.array_equal(prefix[name],records[:8])
            for x,y in zip(rows[:8],replay['replayed'],strict=True):
                assert (x['index'],x['accepted'],x['stages'])==(y['index'],y['accepted'],y['stages'])
                for u,v in zip(x['weights'],y['weights'],strict=True):
                    assert abs(u-v)<1e-7*max(1.,abs(u)),(r,x['index'],u,v)
    with np.load(OUT/'score_4096_4096_7.npz') as score:
        for r in range(2):
            assert score[f'err{r}'].shape==(4096,) and max(score[f'err{r}'])<8e-14
    assert read('checks.json')['continuous_click_completeness']<6e-15
    frugal=read('frugal_full.json')
    assert frugal['full_click_tv_lower']>.107710 and max(frugal['covariance_max_errors'])<1e-14
    hold=read('hardware_holdout.json');protocol=read('hardware_protocol.json')
    assert hold['protocol_sha256']==digest(OUT/'hardware_protocol.json')
    assert [h['file'] for h in hold['holdouts']]==protocol['heldout_files']==[12,13]
    for h in hold['holdouts']:
        p=h['event_count']/h['record_count']
        radius=np.sqrt(np.log(2/protocol['alpha_family'])/(2*h['record_count']))
        assert abs(p-hold['optical_probability']-radius-h['average_conditional_law_tv_lower99'])<1e-14
    with ZipFile(PAPER/'latex-source.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():
            disk=(PAPER/name).read_bytes();zipped=z.read(name)
            if name.endswith('.tex'):disk=disk.replace(b'\r\n',b'\n');zipped=zipped.replace(b'\r\n',b'\n')
            assert disk==zipped,name

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--write',action='store_true');opts=ap.parse_args()
    verify();hashes=files()
    if opts.write:MANIFEST.write_bytes((json.dumps({'sha256':hashes},indent=2)+'\n').encode())
    assert read('manifest.json')['sha256']==hashes
    print('Verified',len(hashes),'artifacts and 128 final decisions.')
