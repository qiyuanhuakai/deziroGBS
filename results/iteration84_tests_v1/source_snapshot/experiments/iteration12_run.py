"""Direct local MPS quality and scaling panel; never overwrite an old run."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from time import perf_counter
import numpy as np
import scipy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.local_mps import local_circuit, build_local_core, build_local_core_layerwise, sample_local_threshold, LocalSamplingError
from gbslab.mps_probe import mps_to_dense, fixed_threshold_law
from gbslab.gaussian import covariance_from_interferometer
from gbslab.exact import ThresholdModel
from iteration10_reference import core_parameters, gaussian_averaged_law, fixed_displacement_law
from iteration12_reference import gaussian_box_coefficients


def save(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def run_case(cfg,out,m,seed,profile,depth,d,chi,quality):
    label=f"m{m}_L{depth}_{profile['name']}_s{seed}_d{d}_chi{chi}"
    row={'input':label,'modes':m,'depth':depth,'profile':profile['name'],'seed':seed,
        'd':d,'chi':chi,'requested_samples':cfg['samples'],'builder':cfg.get('builder','gatewise')}
    started=perf_counter()
    try:
        gates,u=local_circuit(m,depth,np.random.default_rng(seed))
        pars=core_parameters(profile['r'],profile['eta'])
        row['circuit_and_parameters_seconds']=perf_counter()-started
        builder=build_local_core_layerwise if cfg.get('builder')=='layerwise' else build_local_core
        tensors,info=builder(m,pars['core_squeezing'],d,chi,gates)
        row.update(info)
        noise_start=perf_counter()
        rng=np.random.default_rng(seed+10000)
        normals=rng.normal(size=(cfg['samples'],m))
        alpha=.5j*math.sqrt(pars['residual_variance'])*(normals@u.T)
        row['displacement_seconds']=perf_counter()-noise_start
        sample_start=perf_counter()
        try:
            samples,sampling_info=sample_local_threshold(tensors,alpha,rng)
            row['sampling_status']='complete'
            row.update(sampling_info)
        except LocalSamplingError as e:
            samples=e.completed_samples
            row.update(sampling_status='failed',sampling_error=str(e),
                completed_samples=len(samples),failed_sample=e.sample_index,failed_mode=e.mode,failed_prefix=e.prefix)
        row['sampling_seconds']=perf_counter()-sample_start
        row['end_to_end_compute_seconds']=perf_counter()-started
        row['status']='complete' if row['sampling_status']=='complete' else 'sampling_failed'
        np.savez_compressed(out/(label+'.npz'),U=u,alpha=alpha,samples=samples,
            **{f'A{i}':a for i,a in enumerate(tensors)})
        if quality:
            reference_start=perf_counter()
            x=mps_to_dense(tensors)
            gaussian=gaussian_box_coefficients(u,pars['core_squeezing'],d)
            overlap=np.vdot(gaussian.ravel(),x.ravel())
            row['state_trace_distance']=math.sqrt(max(0.,1-abs(overlap)**2))
            row['state_bound_check']=row['state_trace_distance']<=row['construction_trace_budget']+1e-10
            state={'occupations':np.indices(x.shape).reshape(m,-1).T,
                'amplitudes':x.ravel(),'cutoff':m*(d-1)}
            fixed=fixed_threshold_law(tensors,alpha[0])
            ref=fixed_displacement_law(state,alpha[0],max_pairs=cfg['max_reference_pairs'])['distribution']
            row['fixed_reference_max_error']=float(np.max(np.abs(fixed-ref)))
            np.savez_compressed(out/(label+'_reference.npz'),gaussian_box=gaussian,
                mps_tensor=x,fixed_law=fixed,reference_fixed_law=ref)
            if m==2 or (m==4 and d==4 and chi==4):
                try:
                    O=np.block([[u.real,-u.imag],[u.imag,u.real]])
                    L=math.sqrt(pars['residual_variance'])*O[:,m:]
                    state['displacement_covariance']=L@L.T
                    average=gaussian_averaged_law(state,max_pairs=cfg['max_reference_pairs'],max_moments=cfg['max_reference_moments'])
                    target=ThresholdModel(covariance_from_interferometer(u,profile['r'],profile['eta'])).full_distribution()
                    row['averaged_tv']=float(.5*np.abs(average['distribution']-target).sum())
                    row['averaged_bound_check']=row['averaged_tv']<=row['state_trace_distance']+1e-10
                    row['averaged_status']='complete'
                    row['averaged_diagnostics']=average['diagnostics']
                    np.savez_compressed(out/(label+'_averaged.npz'),target=target,
                        distribution=average['distribution'],raw=average['raw_distribution'])
                except Exception as e:
                    row.update(averaged_status='failed',averaged_error=repr(e))
            row['reference_seconds']=perf_counter()-reference_start
    except Exception as e:
        row.update(status='failed',error=repr(e),elapsed_to_error=perf_counter()-started)
    save(out/(label+'.json'),row)
    print(label,row['status'],round(row.get('construction_trace_budget',-1),6),row.get('averaged_status',''),flush=True)
    return row


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('panel',choices=['quality','scaling'])
    parser.add_argument('--config',type=Path,default=ROOT/'configs/iteration12_local_mps.json')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--builder',choices=['gatewise','layerwise'],default='gatewise')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    cfg=json.loads(args.config.read_text(encoding='utf-8'))
    cfg['builder']=args.builder
    save(args.output/'plan.json',dict(cfg,panel=args.panel))
    files=['src/gbslab/local_mps.py','src/gbslab/mps_probe.py','src/gbslab/gaussian.py','src/gbslab/exact.py',
        'experiments/iteration12_run.py','experiments/iteration12_reference.py','experiments/iteration10_reference.py',
        'configs/iteration12_local_mps.json','tests/test_local_mps.py']
    hashes={}
    for f in files:
        dst=args.output/'source_snapshot'/f; dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/f,dst); hashes[f]=hashlib.sha256((ROOT/f).read_bytes()).hexdigest()
    save(args.output/'metadata.json',{'date':'2026-11-08','platform':platform.platform(),
        'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,
        'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'source_hashes':hashes,'cost_scope':'Circuit, parameters, product construction, gates, all QR/SVD, correlated displacements, canonical validation, sampling and output array creation. Excludes file IO and independent quality references.'})
    jobs=[]
    modes=cfg['quality_modes'] if args.panel=='quality' else cfg['scaling_modes']
    depths=[cfg['quality_depth']] if args.panel=='quality' else cfg['depths']
    representations=([{'d':d,'chi':c} for d in cfg['quality_dimensions'] for c in cfg['quality_bonds']]
        if args.panel=='quality' else cfg['representations'])
    for m in modes:
        for depth in depths:
            for seed in cfg['seeds']:
                for p in cfg['profiles']:
                    for rep in representations:
                        jobs.append((m,seed,p,depth,rep['d'],rep['chi']))
    save(args.output/'jobs.json',[{'modes':m,'seed':s,'profile':p,'depth':l,'d':d,'chi':c} for m,s,p,l,d,c in jobs])
    rows=[]
    for job in jobs:
        rows.append(run_case(cfg,args.output,*job,args.panel=='quality'))
        save(args.output/'rows.json',rows)
    save(args.output/'completion.json',{'planned':len(jobs),'finished':len(rows),
        'complete':sum(r['status']=='complete' for r in rows),'failures':sum(r['status']!='complete' for r in rows),
        'averaged_complete':sum(r.get('averaged_status')=='complete' for r in rows),
        'averaged_failed':sum(r.get('averaged_status')=='failed' for r in rows)})


if __name__=='__main__': main()
