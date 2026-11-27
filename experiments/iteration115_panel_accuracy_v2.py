"""Deterministic PNR bounds and sampling noise on the matched 16-detector panel."""
import argparse
import ctypes as ct
import os
import shutil
from time import perf_counter
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits

from iteration106_field_v1 import actual_input
from iteration107_cliques_v1 import optical_x
from iteration100_baselines_v1 import pair_reference, bind, I, D, B
from iteration115_visible_v1 import DATA, energy_law

BANKS = np.array([34, 35])
GRID = 128
SHOTS = 4096


def count_pgf(x, first, grid):
    """Evaluate E[z^N_A w^N_B] by a block Schur complement on the unit torus."""
    eye = np.eye(len(x))
    k = eye - np.linalg.inv(eye + x)
    vacuum = np.exp(-.5 * np.linalg.slogdet(eye + x)[1])
    a, b = np.array(first), np.setdiff1d(np.arange(len(x)), first)
    aa, ab, ba, bb = (k[np.ix_(u,v)] for u,v in ((a,a),(a,b),(b,a),(b,b)))
    unit = np.exp(2j*np.pi*np.arange(grid)/grid)
    spec = np.linalg.eigvalsh(aa)
    values = np.empty((grid,grid),complex)
    for i,z in enumerate(unit):
        effective = bb + z*ba@np.linalg.solve(np.eye(len(a))-z*aa, ab)
        eigen = np.linalg.eigvals(effective)
        assert abs(eigen).max() < 1
        log_det = np.log1p(-z*spec).sum() + np.log1p(-unit[:,None]*eigen).sum(1)
        values[i] = vacuum*np.exp(-.5*log_det)
    return values


def bank_counts(mats, first):
    grid = np.prod([count_pgf(x,first,GRID) for x in mats], axis=0)
    coefficients = np.fft.fft2(grid)/GRID**2
    assert abs(coefficients.imag).max()<1e-12 and coefficients.real.min()>-1e-13
    return coefficients.real


def dark_projection(grouped, gains):
    u = np.linalg.svd(grouped,full_matrices=False)[0]
    rotated = (u.conj().transpose(0,2,1)@grouped).reshape(16,-1)
    bright = [0,16,8,24]
    dark = np.setdiff1d(np.arange(32),bright)
    log_vacuum=0.
    compressed=[]
    for x in optical_x(rotated,gains):
        ab = x[np.ix_(bright,dark)]
        dd = np.eye(len(dark))+x[np.ix_(dark,dark)]
        log_vacuum -= .5*np.linalg.slogdet(dd)[1]
        compressed.append(x[np.ix_(bright,bright)] - ab@np.linalg.solve(dd,ab.T))
    return compressed, -np.expm1(log_vacuum), abs(u[:,:,0])**2


def tail(mats):
    _, upper = energy_law(np.concatenate([np.linalg.eigvalsh(x) for x in mats]),GRID-1)
    return upper


def tv(a,b):
    return float(np.abs(a-b).sum()/2)


def prepare(out):
    raw,gains = actual_input()
    transfer = raw.reshape(144,8,-1)[BANKS]
    optical = optical_x(transfer.reshape(16,-1),gains)
    exact = bank_counts(optical, [*range(8),*range(16,24)])
    compact, leak, split = dark_projection(transfer,gains)
    projected = bank_counts(compact,[0,1])
    with np.load(DATA/'iteration115_tilt_v1/model.npz') as model:
        ids=(2*BANKS[:,None]+np.arange(2)).ravel()
        mixture=np.zeros_like(exact)
        ref_2=np.zeros(81)
        model_tail=0.
        for weight,component in zip(model['weights'],model['marginal']):
            mats = component[:,ids][:,:,ids]
            mixture += weight*bank_counts(mats,[0,1])
            model_tail += weight*tail(mats)
            # Each detector observes its split fraction; local quadrature rotations do
            # not affect photon numbers. Convert interleaved (q,p) to q,q,p,p.
            order=[0,2,1,3]
            fraction=np.sqrt(split[:,0].tolist()*2)
            sub=mats[:,order][:,:,order]*fraction[:,None]*fraction[None,:]
            ref_2 += weight*pair_reference(np.eye(4)+sub,8)
        np.testing.assert_allclose(split,np.diff(model['split'],axis=1,prepend=0)[BANKS],atol=1e-12)
    # Independent univariate PGF; sum the two bank totals modulo the grid.
    total_law, total_tail = energy_law(np.concatenate([np.linalg.eigvalsh(x) for x in optical]),GRID-1)
    two_sum = np.indices((GRID,GRID)).sum(0)%GRID
    np.testing.assert_allclose(np.bincount(two_sum.ravel(),weights=exact.ravel()),total_law,
                               rtol=0,atol=total_tail+1e-13)
    for p in (exact,projected,mixture,ref_2):
        assert abs(p.sum()-1)<1e-12
    two_optical = np.eye(4)+optical[:,[0,8,16,24]][:,:,[0,8,16,24]]
    truth_2 = pair_reference(two_optical,8)
    mod_tv=tv(mixture,exact)
    projected_tv=tv(mixture,projected)
    proj_tail=tail(compact)
    # Modulo counts give a lower bound. Removing wrapping costs at most the two tails.
    upper=projected_tv+proj_tail+model_tail+np.sqrt(leak)
    assert mod_tv<=upper
    stats=dict(banks=BANKS.tolist(),grid=GRID,full_pnr_tv_lower=mod_tv,
               full_pnr_tv_upper=float(upper),dark_nonvacuum=float(leak),
               compressed_modulo_tv=projected_tv,projected_tail=proj_tail,model_tail=model_tail,
               optical_tail=tail(optical),two_channel_model_tv=tv(truth_2,ref_2))
    # Recompute at a second resolution (modulo 64 bins) and verify nested projections.
    check=np.prod([count_pgf(x,[*range(8),*range(16,24)],64) for x in optical],axis=0)
    np.testing.assert_allclose(np.fft.fft2(check).real/64**2,
                               exact.reshape(2,64,2,64).sum((0,2)),rtol=0,atol=1e-14)
    np.savez_compressed(out/'pnr_laws.npz',optical_bank=exact,projected_bank=projected,
                        model_bank=mixture,optical_pair=truth_2,model_pair=ref_2)
    (out/'pnr_bounds.json').write_text(json.dumps(stats,indent=2)+'\n')
    print(stats,flush=True)


def frugal_4096(out):
    """Time both spectra and their logical OR with the same 4,096-row batch."""
    raw, gains = actual_input()
    transfer = raw.reshape(144,8,-1)[BANKS].reshape(16,-1)
    cov = 2*optical_x(transfer,gains)+np.eye(32)
    if os.name == 'nt':
        runtime = os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib = ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll'))
    sample = bind(lib,'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I])
    msg = ct.create_string_buffer(2048)
    exact = np.load(DATA/'iteration115_panel_v1/panel.npz')['optical']
    prior = json.loads((DATA/'iteration115_pair_banks_v1/summary.json').read_bytes())
    reference = prior['independent_65536_word_checks'][0]
    stats = []
    histograms = {}
    for cutoff in (2,4):
        times=[]
        for rep in range(6):
            start=perf_counter()
            samples=[]
            for j,v in enumerate(cov):
                output=np.empty((SHOTS,16),np.uint8)
                status=np.zeros(7)
                seed=115960000+100*cutoff+2*rep+j
                rc=sample(cutoff,SHOTS,seed,v,output,status,msg,len(msg))
                assert rc==0 and status[5]==64 and output.max()<=1
                samples.append(output)
            words=samples[0]|samples[1]
            if rep:
                times.append((perf_counter()-start)*1e6/SHOTS)
        h=np.bincount(words@(1<<np.arange(16)),minlength=65536)
        histograms[f'K{cutoff}']=h
        stats.append(dict(cutoff=cutoff, shots=SHOTS,median_us=float(np.median(times)),
                          mean_us=float(np.mean(times)),times=times,empirical_tv=tv(h/SHOTS,exact),
                          exact_model_tv=reference[f'frugal{cutoff}']['complete_word_tv']))
    np.savez_compressed(out/'frugal_histograms.npz',**histograms)
    return stats


def panel_device():
    import torch
    from iteration115_tilt_v1 import Device,setup
    from iteration115_visible_v1 import hardware,U,J,P
    torch.set_float32_matmul_precision('highest')
    args,_,_=setup(*hardware())
    mat,split,trans,scales,tab,gates,weights=args
    local=[mat.reshape(len(mat),144,4)[:,BANKS].reshape(len(mat),-1),split[BANKS],trans,
           scales[:,BANKS],tab[:,BANKS],gates[:,BANKS],weights]
    lib=ct.CDLL(str(DATA/'iteration115_tilt_v1/gpu.dll'))
    lib.normal103_launch.argtypes=[I,I,U,J,I,P,P]
    lib.tilt115_latent.argtypes=[I]*4+[U,J]+[P]*6
    lib.tilt115_banks.argtypes=[I]*5+[U,J]+[P]*9
    return Device(lib,local,False)


def audit_click_batches(out):
    """Post hoc replication of the high plug-in TV in the scored click batch."""
    panel=np.load(DATA/'iteration115_panel_v1/panel.npz')
    report=json.loads((out/'validation.json').read_bytes())
    sampler=panel_device()
    values=[]
    for j in range(64):
        rows=sampler.draw(SHOTS,0,115980000,j*SHOTS)
        h=np.bincount(rows@(1<<np.arange(16)),minlength=65536)
        values.append(tv(h/SHOTS,panel['optical']))
    assert sampler.error.item()==0
    first=report['measurements'][0]['click_empirical_tv']
    return dict(shots_per_batch=SHOTS,batches=len(values),seed=115980000,
                median=float(np.median(values)),range=[min(values),max(values)],
                at_least_original=int(np.sum(np.array(values)>=first)),observed_tv=values)


def replay_readouts(out,panel,laws):
    """Recover the scored surrogate batches and cross-check their published histograms."""
    sampler=panel_device()
    raw={}
    for pn,name in enumerate(['gpu_direct','gpu_pnr']):
        rows=sampler.draw(SHOTS,pn,115680000+pn,9*SHOTS)
        code=(rows>0)@(1<<np.arange(16))
        np.testing.assert_array_equal(np.bincount(code,minlength=65536),panel[name+'_click'])
        raw[name]=rows.copy()
    assert sampler.error.item()==0 and raw['gpu_direct'].max()==1
    rows=raw['gpu_pnr'].astype(int)
    ids=np.minimum(rows[:,0],8)+9*np.minimum(rows[:,8],8)
    np.testing.assert_array_equal(np.bincount(ids,minlength=81),panel['gpu_pnr_pair'])
    counts=rows.reshape(-1,2,8).sum(2)
    assert counts.max()<GRID
    hist=np.bincount(counts[:,0]*GRID+counts[:,1],minlength=GRID**2).reshape(GRID,GRID)
    raw['gpu_pnr_bank_hist']=hist
    np.savez_compressed(out/'scored_readouts.npz',**raw)
    return tv(hist/SHOTS,laws['optical_bank'])


def complete(out):
    bounds=json.loads((out/'pnr_bounds.json').read_bytes())
    run=DATA/'iteration115_panel_v1'
    raw=json.loads((run/'summary.json').read_bytes())
    panel=np.load(run/'panel.npz')
    laws=np.load(out/'pnr_laws.npz')
    np.testing.assert_allclose(panel['pairref'],laws['optical_pair'],atol=1e-13)
    np.testing.assert_allclose(tv(panel['model'],panel['optical']),raw['full_model_tv'],atol=1e-14)
    masks=np.arange(65536)
    two_click=(masks&1)+2*((masks>>8)&1)
    labels=np.minimum(np.arange(9),1)
    pair_click=labels[None,:]+2*labels[:,None]
    for key,word in [('optical_pair','optical'),('model_pair','model')]:
        p=np.bincount(pair_click.ravel(),weights=laws[key],minlength=4)
        h=np.bincount(two_click,weights=panel[word],minlength=4)
        np.testing.assert_allclose(p,h,rtol=0,atol=2e-11)
    np.testing.assert_allclose(laws['optical_bank'][0,0],panel['optical'][0],atol=2e-12)
    # Against an independent inclusion--exclusion solution on every one of 65,536 words,
    # check that multinomial splitting of the PNR law reproduces all 81 click multiplicities.
    from iteration115_pair_banks_v1 import occupancies
    weights=np.diff(np.load(DATA/'iteration115_tilt_v1/model.npz')['split'],axis=1,prepend=0)
    occ=occupancies(weights,GRID-1)[BANKS]
    from_number=occ[0]@laws['model_bank']@occ[1].T
    pop=np.array([i.bit_count() for i in range(256)])
    click_codes=9*pop[masks&255]+pop[masks>>8]
    from_words=np.bincount(click_codes,weights=panel['model'],minlength=81).reshape(9,9)
    np.testing.assert_allclose(from_number,from_words,rtol=0,atol=2e-11)
    stats=[]
    for row in raw['rows']:
        name=row['name']
        assert row['shots']==SHOTS
        hist=panel[name+'_click']
        assert hist.sum()==SHOTS
        np.testing.assert_allclose(tv(hist/SHOTS,panel['optical']),row['click_empirical_tv'])
        item={k:v for k,v in row.items() if k!='times'}
        item['timed_blocks']=len(row['times'])
        item['mean_us']=float(np.mean(row['times']))
        item['max_us']=float(np.max(row['times']))
        if name!='gpu_direct':
            pair=panel[name+'_pair']
            assert pair.sum()==SHOTS
            np.testing.assert_allclose(tv(pair/SHOTS,laws['optical_pair']),row['pair_empirical_tv'])
        stats.append(item)
    bank_empirical_tv=replay_readouts(out,panel,laws)
    noise={}
    for j,(name,prob,ref) in enumerate([('optical_click',panel['optical'],panel['optical']),
                         ('model_click',panel['model'],panel['optical']),
                         ('optical_pair',laws['optical_pair'],laws['optical_pair']),
                         ('model_pair',laws['model_pair'],laws['optical_pair']),
                         ('optical_bank',laws['optical_bank'].ravel(),laws['optical_bank'].ravel())]):
        rng=np.random.default_rng(np.random.SeedSequence([115,904,j]))
        prob=prob.clip(0);prob/=prob.sum()
        values=[tv(rng.multinomial(SHOTS,prob)/SHOTS,ref) for _ in range(1000)]
        noise[name]=np.quantile(values,[.025,.5,.975]).tolist()
    report=dict(two_bank_pnr_empirical_tv=bank_empirical_tv,frugal=frugal_4096(out),shots_per_sampler=SHOTS,ports=16,pnr=bounds,full_click_model_tv=raw['full_model_tv'],
                measurements=stats,cache_us=raw['optical_cached']['us'],cache_enumeration_s=raw['exact_enumeration_s'],
                null_empirical_tv_quantiles=noise,replicates=1000)
    (out/'validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    p.add_argument('--complete',action='store_true')
    p.add_argument('--audit-clicks',action='store_true')
    args=p.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    with threadpool_limits(limits=1):
        if args.audit_clicks:
            out=args.output/'click_replication.json'
            out.write_text(json.dumps(audit_click_batches(args.output),indent=2)+'\n')
        elif args.complete:complete(args.output)
        else:prepare(args.output)
