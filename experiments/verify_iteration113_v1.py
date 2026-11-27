"""Independent repeated-hafnian spot checks, optical bounds, hardware histograms and release manifest."""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from iteration113_support_v1 import ROOT, optics, tv


def reference(x,occupations):
    m=len(occupations)
    identity=np.eye(m)
    transform=np.block([[identity,identity],[1j*identity,-1j*identity]])/np.sqrt(2)
    q=np.eye(2*m)+x
    b=transform.T@(np.eye(2*m)-np.linalg.inv(q))@transform
    @lru_cache(None)
    def haf(v):
        if not v:
            return 1.
        first=v[0]
        return sum(b[first,v[i]]*haf(v[1:i]+v[i+1:]) for i in range(1,len(v)))
    indices=tuple(np.repeat(np.arange(2*m),np.tile(occupations,2)))
    p=haf(indices)*np.exp(-np.linalg.slogdet(q)[1]/2)/math.prod(math.factorial(int(i)) for i in occupations)
    assert abs(p.imag)<1e-12
    return p.real


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    results=ROOT/'results'
    get=lambda name:json.loads((results/name).read_text(encoding='utf8'))
    small=[]
    x=optics(4)[4]
    truth=np.load(results/'iteration112_four_peers_v1/peers.npz')
    rng=np.random.default_rng(113711)
    for name in ('support','deep'):
        folder=f'iteration113_{name}_v1'
        data=np.load(results/folder/'laws.npz')
        stats=get(folder+'/summary.json')
        keys=data['keys']
        values=data['values']
        table=data['sector']
        spectrum=np.arange(table.shape[1]-1)
        size=np.array([i.bit_count() for i in range(len(table))])[:,None]
        unresolved=(spectrum>stats['common'])&((size>stats['support_width'])|(spectrum>stats['extended']))
        unresolved &= (size>1)&(spectrum>size)
        assert abs(table[:,:-1].clip(0)[unresolved].sum()-stats['unresolved_mass'])<1e-13
        exact=table[:,:-1].sum(0)
        assert tv(exact,truth['total'][:len(exact)])<1e-10
        offsets=data['offsets']
        checked=[]
        for n in [0,1,2,4,6,8,stats['common'],12,14]:
            for idx in rng.integers(offsets[n],offsets[n+1],size=3):
                occ=np.array([(int(keys[idx])>>(4*j))&15 for j in range(13)])
                p=reference(x,occ)
                checked.append(abs(p-values[idx])/max(p,1e-24))
        assert max(checked)<2e-8
        ds=data['direct_samples'];ps=data['pnr_samples']
        assert ds.shape==ps.shape==(8192,13) and ds.max()==1 and ps.max()>1
        small.append(dict(cut=stats['common'],full_pnr_truncation_tv_upper=stats['pnr_truncation_tv_bound'],
            exact_count_coefficients=len(keys),hafnian_checks=len(checked),max_relative_hafnian_error=max(checked),
            direct_click_tv=stats['numerical_checks']['click_tv'],two_port_model_tv=stats['pnr_pair_model_tv'],
            two_port_empirical_tv=stats['pnr_pair_empirical_tv'],max_total_cdf_gap=stats['max_total_cdf_gap'],
            direct_setup_s=stats['direct_setup_s'],pnr_setup_s=stats['pnr_setup_s'],
            click_us=stats['direct']['us'],pnr_us=stats['pnr']['us']))
    hd=np.load(results/'iteration113_device_v1/banks.npz')
    h=get('iteration113_device_v1/summary.json')
    pop=np.array([i.bit_count() for i in range(256)])
    observed=hd['hardware'];actual=observed/observed.sum(2,keepdims=True)
    assert np.all(observed.sum(2)==np.array(h['shots'])[:,None])
    np.testing.assert_allclose((abs(actual-hd['words']).sum(2)/2).mean(1),h['average_eight_bin_word_tv'])
    assert abs(hd['number_bins'].sum((1,2))-1).max()<1e-12
    peers=get('iteration112_four_peers_v1/peers.json')
    old=get('iteration112_release_v1/validation.json')
    hardware={k:v for k,v in h.items() if not k.startswith('one_bank_')}
    hardware['hardware_mean_clicks']=(actual@pop).sum(1).tolist()
    summary=dict(small=small,hardware=hardware,jiuzhang3_paper_full_record_us=1.27,
        four_phase_prior=old['small'][-1],same_input_cached_click=[v for v in old['matched_cache'] if v['buses']==4],
        frugal=[{k:v for k,v in row.items() if k!='times'} for row in peers['frugal']],
        photonic_engines=[{k:v for k,v in row.items() if k!='times'} for row in peers['pe']])
    (out/'validation.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8')
    files=sorted(f for f in ROOT.glob('results/iteration113*/*') if f.is_file() and f.name!='manifest.json')
    hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (out/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n',encoding='utf8')
    print([(r['cut'],r['max_relative_hafnian_error']) for r in small],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    run(parser.parse_args().output)
