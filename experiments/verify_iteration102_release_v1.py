"""Check all spatial count pairs supported by optical tables; compare three hardware click marginals."""
import argparse
import ctypes as ct
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,HIST,ORDER,I,D,bind,H,save,manifest
from iteration100_projection_v1 import physical
from iteration102_projection_v1 import Projection,PROJ
from iteration102_sample_v1 import Sampler,BLOCK
from iteration102_forest_v1 import FIT,RUN,DATA
from iteration102_triads_v1 import TRI
from iteration102_four_v1 import FOUR


def read(p):
    return json.loads(p.read_text())


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    dirs=[FIT,TRI,FOUR,RUN,PROJ]+[DATA/f'iteration102_{s}_v1' for s in ('release','baselines','blocks','throughput')]
    hashes={}
    for folder in dirs:
        for f,digest in read(folder/'manifest.json').items():assert H(folder/f)==digest,(folder,f)
        hashes[folder.name]=H(folder/'manifest.json')
    print('verified',len(hashes),'manifests',flush=True)
    projection=Projection()
    rows=read(PROJ/'summary.json')['rows']
    structured=[r for r in rows if r['kind'] in ('edge','triad')]
    a,b=np.array([r['groups'] for r in structured]).T
    clip=6
    size=(clip+1)**2
    expected=[]
    for i,j in zip(a,b):
        pm=projection.pair(i,j,4)
        ix=np.minimum(np.arange(len(pm)),clip)[:,None]*(clip+1)+np.minimum(np.arange(pm.shape[1]),clip)
        expected.append(np.bincount(ix.ravel(),weights=pm.ravel(),minlength=size))
    expected=np.array(expected)
    model=Sampler(RUN/'forest.dll',depth=4)
    oracle=bind(ct.CDLL(str(DATA/'iteration96_quad_v1/oracle.dll')),'oracle96',[I,I,D,D,D,D])
    spec=np.load(ORDER)
    with np.load(HIST) as z:
        ports=z['ports']
        htrain,hhold=z['histograms'].transpose(1,0,2)
    optical=[]
    for ids in ports:
        p=np.empty(256)
        assert oracle(8,2,physical(spec,ids),np.zeros((2,16)),p,np.empty(2))==0
        optical.append(p)
    weights=(1<<np.arange(8)).astype(np.uint16)
    routes=[]
    n=524288
    radius=float(np.sqrt((size*np.log(2)+np.log(2*len(a)/.005))/(2*n)))
    for route in (False,True):
        counts=np.zeros((len(a),size),np.int64)
        offset_codes=np.arange(len(a),dtype=np.int32)*size
        eight=np.zeros((3,256),np.int64)
        pilot=eight.copy()
        for is_pilot,total,seed,hist in ((True,65536,102141000+route,pilot),(False,n,102142000+route,eight)):
            for off in range(0,total,BLOCK):
                x,c=model.sample(BLOCK,route,True,16,seed,off)
                if route:assert np.array_equal(x.reshape(len(c),-1,8).sum(2),c)
                for k,group in enumerate(ports):
                    bits=(x[:,group]>0)@weights
                    hist[k]+=np.bincount(bits,minlength=256)
                if is_pilot:continue
                cc=c.clip(0,clip)
                ids=cc[:,a]*(clip+1)+cc[:,b]+offset_codes
                counts+=np.bincount(ids.ravel(),minlength=counts.size).reshape(counts.shape)
        empirical=abs(counts/n-expected).sum(1)/2
        assert empirical.max()<=radius,(empirical.max(),radius)
        panels=[]
        for i in range(3):
            p=eight[i]/n
            q=hhold[i]/hhold[i].sum()
            event=pilot[i]/pilot[i].sum()>htrain[i]/htrain[i].sum()
            difference=float((p-q)[event].sum())
            e=np.sqrt(np.log(12/.005)/(2*n))+np.sqrt(np.log(12/.005)/(2*hhold[i].sum()))
            panels.append(dict(ports=ports[i].tolist(),hardware_holdout=int(hhold[i].sum()),empirical_hw_tv=float(abs(p-q).sum()/2),
                empirical_optical_tv=float(abs(p-optical[i]).sum()/2),optical_vs_hardware_empirical_tv=float(abs(optical[i]-q).sum()/2),
                event_difference=difference,hardware_tv_lower99=max(0.,difference-e)))
        routes.append(dict(pnr=route,n=n,coarse_pair_empirical_tv_max=float(empirical.max()),
            largest_pair=[int(a[empirical.argmax()]),int(b[empirical.argmax()])],all_93_sampling_radius99=radius,hardware_8bit=panels))
        np.savez_compressed(out/f'counts_{int(route)}.npz',count_histograms=counts,pairs=np.c_[a,b],click_hold=eight,click_pilot=pilot)
    past=read(DATA/'iteration102_release_v1/validation.json')
    base=read(DATA/'iteration102_baselines_v1/summary.json')
    blocks=read(DATA/'iteration102_blocks_v1/summary.json')
    throughput=read(DATA/'iteration102_throughput_v1/summary.json')
    # Round-level summaries retain precise units; full trial timings remain in the archive.
    valid=[r for r in rows if '4' in r['depths']]
    worst={k:max(valid,key=lambda r:r['depths']['4'][k]) for k in ('click_tv','count_tv')}
    save(out,'validation.json',dict(manifests=hashes,hardware_and_count_audit=routes,full_count_pairs=len(structured),
        detector_pairs=past['complete_panels'],residual={k:dict(groups=r['groups'],tv=r['depths']['4'][k]) for k,r in worst.items()},
        hardware=[{k:r[k] for k in ('pnr','n','hardware_ks','hardware_radius99','photon_ks','max_vacuum_gap','all_pair_99_interval')}
                  for r in past['heldout']],fixed_16_threads=[{k:r[k] for k in ('pnr','median_us','click_hardware_ks','photon_optical_ks')}
                  for r in past['archived_benchmark'] if r['depth']==4 and r['field']],
        frugal=[dict(groups=r['groups'],ours_us=r['median_us'],ours_tv=r['click_tv'],first_batch_us=r['cold_us'],
                  controls=[{k:f[k] for k in ('k','median_us','tv')} for f in r['frugal']]) for r in base['rows']],
        pnrs=dict(ports=blocks['ports'],full_numeric_upper=blocks['block_full_pnr_tv_numerical_upper'],pe_us=blocks['pe_median_us'],
                  ours_us=blocks['rows'][1]['median_us'],pe_empirical=blocks['pe_empirical'],ours_empirical=blocks['ours_empirical']),
        block_scaling=[{k:r[k] for k in ('width','pnr','workers','median_us','numerical_full_tv_upper')} for r in blocks['rows']],
        retiming=[dict(pnr=r['pnr'],selected=r['chosen'],checks=[{k:v[k] for k in ('workers','n','median_us')} for v in r['hold']])
                  for r in throughput['rows']]))
    manifest(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):
        main(ap.parse_args().output.resolve())
