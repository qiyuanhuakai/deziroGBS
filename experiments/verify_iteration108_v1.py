"""Check the complete-law comparisons, sparse replay, RNG domains and every published stage."""
import argparse
import math
import mpmath as mp
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,H,save,manifest
from iteration106_sample_v1 import DATA,read
from iteration107_fast_v1 import oracle
from iteration108_matching_v1 import certificate
from iteration108_compact_v1 import Compact
from iteration108_compact_bench_v1 import expand
from iteration108_replicas_v1 import Replicas
from iteration108_device_v1 import Global

STAGES=('device','bench','sectors','low_bench','matching','quantum_bench','compact','compact_bench','exact',
        'replicas','bright','jiuzhang','actual_quantum','whole_pnr','residual','figure')


def high_precision_bound(r,cap,tries,copies):
    with mp.workdps(90):
        good=mp.mpf(1)
        for row in np.tanh(np.atleast_2d(r)):
            x=[mp.mpf(float(l))**2 for l in row];weights=[mp.mpf(1)]+[mp.mpf(0)]*cap
            for value in x:
                for n in range(1,cap+1):weights[n]+=value*weights[n-1]
            vacuum=mp.fprod(1-w for w in x);s=mp.fsum(x);part=vacuum
            for n in range(1,cap+1):
                acc=mp.factorial(n)*weights[n]/(math.prod(range(1,2*n,2))*s**n)
                part+=vacuum*weights[n]*(1-(1-acc)**tries)
            good*=part
        risk=1-good**copies
        return float(np.nextafter(float(risk),float('inf')))


def main(out):
    hashes={};st={}
    for name in STAGES:
        d=DATA/f'iteration108_{name}_v1';p=d/'manifest.json';items=read(p)
        assert set(items)=={v.name for v in d.iterdir() if v.name!='manifest.json'}
        for f,h in items.items():assert H(d/f)==h,(name,f)
        hashes[name]=H(p)
        if (d/'summary.json').exists():st[name]=read(d/'summary.json')
        if (d/'build.json').exists():
            meta=read(d/'build.json')['source_sha256']
            if isinstance(meta,dict):
                for src,h in meta.items():assert H(ROOT/src)==h
            elif name=='device':assert H(ROOT/'native/global_readout108_v1.cu')==meta
    # Reconstruct the selected complete optical laws from stored word-by-word data.
    for i,row in enumerate(st['exact']['rows']):
        with np.load(DATA/f'iteration108_exact_v1/laws_{i}.npz') as z:
            tv=abs(z['laws'][1:]-z['target']).sum(1)/2
            assert np.max(abs(tv-row['tv_by_cap']))<1e-14
            event=z['laws'][5]>z['target']
            assert abs(abs(z['observed'][event].sum()/row['native_n']-z['laws'][5,event].sum())-row['cap4_validation_event_gap'])<1e-12
    for i,row in enumerate(st['bright']['rows']):
        for pnr in (0,1):
            with np.load(DATA/f'iteration108_bright_v1/case{i}_{pnr}.npz') as z:
                assert z['tally'][:,0].sum()==row['shots']*row['replicas'] and z['total'].sum()==row['shots']
                assert abs(z['panel']/row['shots']-z['truth']).sum()/2==row['runs'][pnr]['panel_empirical_tv']
        assert abs(row['bound']+np.expm1(row['replicas']*np.log1p(-row['per_replica'])))<1e-15
    controls=st['compact_bench']['controls']
    oracle() # Include the archived GCC runtime directory.
    u=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2)
    hom=Compact(u,np.array([[.54]]));x=hom.draw(131072,True,108053000)
    selected=x[x.sum(1)==2];assert len(selected)==controls['hom_two_photon_records'] and np.all(selected[:,0]!=1)
    x=hom.draw(512,True,108111000,sparse=True,slots=1)
    assert np.array_equal(expand(x)[37:]>0,hom.draw(475,False,108111000,37))
    wide=Replicas(np.eye(2,dtype=complex),np.array([[.4]]),4096,cache_mb=1)
    x,counts=wide.draw(160,True,108112000,stats=True)
    assert x.dtype==np.uint32 and np.array_equal(x[:,0],x[:,1]) and x.min()>255 and counts[:,0].sum()==160*4096
    assert np.array_equal(x>0,wide.draw(160,False,108112000))
    # Sector-dependent purposes do not share one Philox address. GPU global
    # readout reserves counter word 1 = 108; the latent field uses zero there.
    domains={((s<<16)|purpose) for s in range(65536) for purpose in (320,321,322,323,325)}
    assert len(domains)==5*65536 and 332 not in domains
    for row in st['bench']['rows']:
        with np.load(DATA/f"iteration108_bench_v1/obs{row['m']}.npz") as o,np.load(DATA/f"iteration108_bench_v1/law{row['m']}.npz") as z:
            gap=abs(o['1_photon'].cumsum()/row['shots']-z['optical_total'].cumsum()).max()
            assert abs(gap-row['runs'][1]['total_ks'])<1e-12
    with np.load(DATA/'iteration108_bench_v1/jiuzhang.npz') as z:t,r=z['transfer'],z['squeezing']
    with np.load(DATA/'iteration108_jiuzhang_v1/targeted.npz') as z:device=Global(t,r,dict(z))
    for pn in (False,True):
        a=device.draw(73,pn,108113000+pn,4)
        assert np.array_equal(a[9:],device.draw(64,pn,108113000+pn,13))
    residual=st['residual']['spectra']
    for v in residual:assert v['min_noise_eigenvalue']>0 and abs(v['residual_photons']+v['displacement_photons']-v['output_photons'])<1e-10
    b=st['bench']['rows'];b107=read(DATA/'iteration107_global_v1/summary.json')
    jq=st['actual_quantum'];j=st['jiuzhang']
    selected=dict(device=read(DATA/'iteration108_device_v1/build.json')['device'],hardware_us=1.27,
        actual_jiuzhang=dict(full_device=b[-1],focused_timing_us=j['focused_timing'],focused_coarse_law=j['optical'],
          focus_click_reference_tv=j['focused_click_target_check'],focus_pnr_reference_tv=j['focused_pnr_target_check'],
          local_peers=j['rows'],frugal_projected_gpu=j['frugal_gpu'],pnr_equal32768_tv=j['pnr_equal_tv'],pe_equal32768_tv=j['pe_equal_tv'],pe_us=j['pe_us'],
          quantum_matching=jq,quadrature_decomposition=residual),
        global_131072=dict(this_round=b[-2],previous=b107['rows'][-1]),
        complete_click_comparisons=st['exact']['rows'],whole_pnr_comparison=st['whole_pnr'],
        weak_quantum=st['compact_bench']['rows'],generic_quantum_peer_4096=st['quantum_bench']['rows'][2],
        many_independent_spectra=st['bright']['rows'],low_sector_wide=st['low_bench']['rows'][1],
        validation=dict(controls=controls,independent_source_domains=len(domains),wide_min=int(x.min()),wide_max=int(x.max()),
                        max_complete_oracle_difference=max(z['oracle_check'] for z in st['exact']['rows']),stage_manifests=hashes),
        interpretation='TV and KS are fractions. Full-law bounds are ideal-arithmetic coupling bounds; empirical statistics include finite-shot error. Orthogonal spectral replicas are a distinct input from actual Jiuzhang.')
    certified=[]
    for modes,k,mean,copies,cap,tries,reported in [
        (131072,128,.12,1,4,2048,st['compact_bench']['rows'][-2]['cert']['full_joint_tv_bound']),
        (131072,128,.015,1,4,2048,st['compact_bench']['rows'][-1]['cert']['full_joint_tv_bound']),
        (4096,64,.12,1024,5,8192,st['bright']['rows'][1]['bound']),
        (8192,128,.12,1024,5,8192,st['bright']['rows'][2]['bound'])]:
        from iteration108_low_bench_v1 import weak
        r=weak((.55+.52*np.sin(.4+np.arange(k))**2)[None,:],mean)
        value=high_precision_bound(r,cap,tries,copies)
        assert abs(value-reported)<1e-12
        certified.append(dict(m=modes,sources=k,mean_pairs_per_spectrum=mean,spectra=copies,full_tv_upper=value))
    selected['high_precision_full_bounds']=certified
    out.mkdir(exist_ok=False,parents=True);save(out,'validation.json',selected);manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):main(p.parse_args().output.resolve())
