"""Check frozen event files, archived record counts and simultaneous TV statements."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/iteration121_positive_p'
PAPER = ROOT/'papers/number-tilted-gbs'


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    b = path.read_bytes()
    if path.suffix in ('.py','.md','.json','.tex'):
        b = b.replace(b'\r\n',b'\n')
    return hashlib.sha256(b).hexdigest()


def check():
    rows = []
    for route in ('pnr','click'):
        for method,version,nkey,hkey in [('witness','','test','hits'),('records','','records','test_hits'),
                                           ('fp64','','rows','hits'),('large','_large','rows','hits')]:
            meta = read(OUT/f'{route}_{method}.json')
            protocol = OUT/f'{route}{version}_protocol.npz'
            assert meta['protocol_sha256']==sha(protocol)
            with np.load(protocol) as p:
                banks,event,opt = [p[k] for k in ('banks','event','optical')]
                assert len(set(banks))==len(banks) and opt.min()>0 and abs(opt.sum()-1)<1e-11
                assert event.dtype==bool and len(event)==1<<len(banks)
                assert abs(meta['optical']-opt[event].sum())<1e-12
            enclosure = read(OUT/f'{route}{version}_interval.json')
            assert enclosure['cells']==len(event)
            if method=='witness':
                h = np.load(OUT/f'{route}_counts.npy')
            else:
                with np.load(OUT/f'{route}_{method}.npz') as records:
                    h = records['counts']
                    sample = records['first4096']
                    assert sample.shape==(4096,1152)
                    assert sample.dtype==np.uint8 and (route=='pnr' or sample.max()==1)
                    w = (sample.reshape(-1,144,8)[:,banks,:]!=0).any(2) @ (1<<np.arange(len(banks)))
                    assert np.all(np.bincount(w,minlength=len(h))<=h)
                    times = records['batch_seconds']
                    assert len(times)*2048==h.sum() and min(times)>0
                    timing = meta.get('host_us_per_record',meta.get('delivered_us_per_record'))
                    assert abs(times.sum()*1e6/h.sum()-timing)<1e-9
            n,hits = int(h.sum()),int(h[event].sum())
            assert n==meta[nkey] and hits==meta[hkey] and abs(meta['empirical']-hits/n)<1e-13
            opt = meta['optical']
            lower = max(0.,float(beta.ppf(.01/8,hits,n-hits+1))-opt-1e-8)
            rows.append(dict(route=route,experiment=method,records=n,hits=hits,
                             optical_probability=opt,full_tv_lower99=lower))
    return rows


def files():
    with ZipFile(PAPER/'latex-source.zip') as z:
        assert z.testzip() is None
        members = [PAPER/name for name in z.namelist()]
        for p in members:
            data = p.read_bytes();archived = z.read(p.name)
            if p.suffix=='.tex':
                data = data.replace(b'\r\n',b'\n')
                archived = archived.replace(b'\r\n',b'\n')
            assert data==archived
    paths = [*sorted((ROOT/'experiments').glob('iteration121_*.py')),
             *sorted(p for p in OUT.iterdir() if p.name!='manifest.json'),
             *members,PAPER/'README.md',PAPER/'manuscript.pdf',PAPER/'latex-source.zip',
             ROOT/'results/iteration83_loss_pilot_v1/input.npz']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in paths}


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--write',action='store_true')
    args = p.parse_args()
    rows = check()
    if args.write:
        (OUT/'comparison.json').write_bytes((json.dumps(rows,indent=2)+'\n').encode())
        (OUT/'manifest.json').write_bytes((json.dumps(files(),indent=2)+'\n').encode())
    assert rows==read(OUT/'comparison.json')
    assert files()==read(OUT/'manifest.json')
    print('Verified eight test streams, four optical word tables and the manuscript source package.')
