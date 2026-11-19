"""Audit saved pipeline states, original-state budgets, sampling and sources."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import zipfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary
from gbslab.gaussian_mps_feasibility import equal_core_spectra,marginal_photon_tails
from gbslab.local_mps import local_circuit
from gbslab.mps_overlap import compare_mps
from gbslab.mps_probe import mps_to_dense,geometric_top_weights
from iteration10_reference import core_parameters
from iteration22_external import ARCHIVE_SHA
from iteration23_support import digest,load_mps,mps_fingerprint,save


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--hafnian-archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    files=sorted({Path(__file__).relative_to(ROOT).as_posix(),'experiments/iteration23_support.py','experiments/iteration10_reference.py','experiments/iteration22_external.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in files}
    for f in files:
        target=out/'source_snapshot'/f; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,target)
    save(out/'source_hashes.json',hashes)
    save(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,scipy=scipy.__version__))
    checks=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    states={}; file_hashes={}; spectral_cache={}; pairs=[]; sample_count=0; all_rows=[]
    pars=core_parameters(.9,.65)
    def state(row):
        record=row['state']; name=record['path']; fp=record['array_fingerprint']
        if name not in file_hashes: file_hashes[name]=digest(ROOT/name)
        check(name+' file hash',file_hashes[name]==record['file_sha256'])
        if fp not in states:
            a=load_mps(ROOT/name); check(name+' array fingerprint',mps_fingerprint(a)==fp); states[fp]=a
        return states[fp]
    comparison_cache={}
    for phase in ('quality','performance'):
        panel=ROOT/f'results/iteration23_{phase}'; plan=json.loads((panel/'plan.json').read_text()); rows=json.loads((panel/'rows.json').read_text()); all_rows.extend(rows)
        check(phase+' exactly 48 terminal rows',len(rows)==len(plan['jobs'])==48 and [r['sequence'] for r in rows]==list(range(48)))
        for name,sha in plan['source_hashes'].items(): check(phase+' source '+name,digest(ROOT/name)==sha and digest(panel/'source_snapshot'/name)==sha)
        grouped={}
        for job,row in zip(plan['jobs'],rows):
            i=row['sequence']; tag=f'{phase} {i}'; check(tag+' frozen job',all(row[k]==v for k,v in job.items()))
            complete=row['status'] in ('accepted','budget_missed') and row.get('validation_status')=='passed'
            check(tag+' completed and validated',complete)
            if not complete: continue
            a=state(row); info=row['info']; m=row['modes']; d=row['dimension']; rank=row['rank']
            check(tag+' MPS dimensions',len(a)==m and a[0].shape[0]==a[-1].shape[2]==1 and all(t.shape[1]==d and max(t.shape[0],t.shape[2])<=rank for t in a))
            rng=np.random.default_rng(row['seed']); U=haar_unitary(m,rng) if row['ensemble']=='haar' else local_circuit(m,12,rng)[1]
            B=-math.tanh(pars['core_squeezing'])*(U@U.T)
            with np.load(panel/f'run_{i:02d}.input.npz',allow_pickle=False) as data:
                check(tag+' original input',np.array_equal(U,data['U']) and np.array_equal(B,data['pairing']))
            key=(m,row['seed'],row['ensemble'],d,rank)
            if key not in spectral_cache:
                independent=equal_core_spectra(U,pars['core_squeezing'],rank)
                marginal=[float(marginal_photon_tails(independent['normal_mean'],abs(independent['anomalous'][j,j]),[d])['tail_upper_estimates'][0]) for j in range(m)]
                tails=[max(0.,1-math.fsum(c['top_weights'])) for c in independent['cuts']]
                spectral_cache[key]=(independent,marginal,tails)
            independent,marginal,tails=spectral_cache[key]; discrepancies=[]
            for record,cut in zip(info['cuts'],independent['cuts']):
                observed=geometric_top_weights(record['parameters'],rank); expected=cut['top_weights']; size=max(len(observed),len(expected))
                discrepancies.append(float(np.max(np.abs(np.pad(observed,(0,size-len(observed)))-np.pad(expected,(0,size-len(expected)))))))
            check(tag+' independent original cut spectra',len(info['cuts'])==m-1 and max(discrepancies,default=0.)<1e-10 and abs(max(discrepancies,default=0.)-row['independent_spectrum_max_error'])<1e-15)
            check(tag+' original cut tails',np.allclose(tails,info['cut_tail_estimates'],atol=1e-12,rtol=0))
            check(tag+' original physical tails',np.allclose(marginal,info['marginal_tail_estimates'],atol=1e-13,rtol=0))
            loss=math.fsum(tails)+math.fsum(marginal); bound=math.sqrt(min(1.,loss)); success=info['projection_success']
            check(tag+' budget reconstruction',abs(bound-row['construction_trace_budget'])<1e-10 and abs(loss-info['original_loss_sum'])<1e-11)
            check(tag+' target status',(row['status']=='accepted')==row['target_met']==(row['construction_trace_budget']<=.1))
            check(tag+' success versus original projector loss',0<success<=1+1e-8 and (loss>1 or success>=1-4*loss/(1+loss)**2-1e-9))
            check(tag+' backend and stage timing',info['coefficient_backend']==row['backend'] and all(v>=0 for v in info['phase_seconds'].values()) and abs(sum(info['phase_seconds'].values())-info['total_seconds'])<1e-9 and info['total_seconds']<=row['phase_seconds']['construction']+1e-8)
            check(tag+' complete pipeline timing',all(v>=0 for v in row['phase_seconds'].values()) and abs(sum(row['phase_seconds'].values())-row['pipeline_seconds'])<1e-9 and row['pipeline_seconds']<=row['worker_seconds']<=row['process_wall_seconds']+1e-5)
            with np.load(panel/f'run_{i:02d}.samples.npz',allow_pickle=False) as data: alpha=data['alpha']; samples=data['samples']
            expected_alpha=.5j*math.sqrt(pars['residual_variance'])*(np.random.default_rng(row['seed']+10000).standard_normal((16,m))@U.T)
            check(tag+' shared displacement',np.array_equal(alpha,expected_alpha))
            check(tag+' observed samples',samples.shape==(16,m) and np.all((samples==0)|(samples==1)) and row['completed_samples']==16 and row['sampling_status']=='complete')
            sample_count+=len(samples)
            if phase=='quality':
                with np.load(panel/f'run_{i:02d}.reference.npz',allow_pickle=False) as data: reference=data['gaussian_box']
                actual=math.sqrt(max(0.,1-abs(np.vdot(reference,mps_to_dense(a)))**2))
                check(tag+' original Gaussian actual quality',abs(actual-row['actual_original_distance'])<1e-12 and actual<=bound+1e-10 and row['actual_bound_check'])
            groupkey=key+(row['repetition'],); grouped.setdefault(groupkey,{})[row['backend']]=(row,samples)
        check(phase+' paired cases',len(grouped)==24 and all(set(v)=={'pco','recurrence'} for v in grouped.values()))
        for key,group in sorted(grouped.items()):
            if set(group)!={'pco','recurrence'}: continue
            (left,ls),(right,rs)=group['pco'],group['recurrence']; fp=(left['state']['array_fingerprint'],right['state']['array_fingerprint'])
            if fp not in comparison_cache: comparison_cache[fp]=compare_mps(state(left),state(right))
            comparison=comparison_cache[fp]
            check(phase+' paired state '+str(key),comparison['pure_trace_distance_from_difference']<1e-10 and abs(comparison['log_norm_squared_left'])<1e-10 and abs(comparison['log_norm_squared_right'])<1e-10)
            check(phase+' paired budget '+str(key),left['construction_trace_budget']==right['construction_trace_budget'] and abs(left['info']['projection_success']-right['info']['projection_success'])<1e-11)
            check(phase+' paired sample '+str(key),np.array_equal(ls,rs))
            pairs.append(dict(phase=phase,case=list(key),pco_sequence=left['sequence'],recurrence_sequence=right['sequence'],comparison=comparison,samples_equal=bool(np.array_equal(ls,rs))))
        print('verified',phase,len(rows),flush=True)
    check('1536 generated outputs with repeated benchmark seeds',sample_count==1536)
    precision=ROOT/'results/iteration23_precision'; pp=json.loads((precision/'plan.json').read_text()); pr=json.loads((precision/'rows.json').read_text())
    check('precision archive',digest(args.archive)==ARCHIVE_SHA)
    for name,sha in pp['source_hashes'].items(): check('precision source '+name,digest(ROOT/name)==sha and digest(precision/'sources'/Path(name).name)==sha)
    check('four precision pairs',len(pr)==4 and [r['pair'] for r in pr]==pp['pairs'] and all(r['status']=='complete' for r in pr))
    for r in pr:
        check(r['pair']+' independent norm methods',abs(r['log_norm_qr_residual_left'])<1e-9 and abs(r['log_norm_qr_residual_right'])<1e-9)
        check(r['pair']+' difference/overlap consistency',abs(r['pure_trace_distance_from_difference']**2+math.expm1(min(0.,r['raw_log_fidelity'])))<1e-9)
    external_hashes={}
    with zipfile.ZipFile(args.archive) as z:
        for path in [*sorted(precision.glob('*_inputs.json')),precision/'supplementary_inventory.json']:
            for entry in json.loads(path.read_text()):
                name=entry['name']
                if name not in external_hashes: external_hashes[name]=hashlib.sha256(z.read(name)).hexdigest()
                check('external '+name,external_hashes[name]==entry['sha256'])
    source_inventory=json.loads((ROOT/'research_archive/external_sources/zenodo7992736_inventory.json').read_text())
    check('archived hafnian source checksum',digest(args.hafnian_archive)==source_inventory['archive_sha256'])
    with zipfile.ZipFile(args.hafnian_archive) as z:
        for entry in source_inventory['files']:
            data=z.read(entry['name'])
            check('archived source '+entry['name'],len(data)==entry['bytes'] and hashlib.sha256(data).hexdigest()==entry['sha256'])
    tests=ROOT/'results/iteration23_tests'; tr=json.loads((tests/'validation.json').read_text())
    check('five new tests and inherited failures',tr['new_tests_passed']==5 and tr['unchanged_failure_ids'] and tr['unchanged_sources'])
    for name,sha in json.loads((tests/'source_hashes_at_start.json').read_text()).items(): check('test source '+name,digest(ROOT/name)==sha and digest(tests/'source_snapshot'/name)==sha)
    check('audit sources unchanged',hashes=={f:digest(ROOT/f) for f in files})
    summary=dict(count=len(checks),passed=sum(c['passed'] for c in checks),paired_comparisons=len(pairs),distinct_comparisons=len(comparison_cache),unique_states=len(states),samples=sample_count,external_arrays=len(external_hashes),state_dependencies=file_hashes,checks=checks)
    save(out/'comparisons.json',pairs); save(out/'validation.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('checks','state_dependencies')},indent=2))
    if not all(c['passed'] for c in checks): raise AssertionError('Audit failed; preserve this output directory')


if __name__=='__main__': main()
