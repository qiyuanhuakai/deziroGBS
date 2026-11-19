"""Verify original-cut Gaussian maps, saved states, independent replay, tests."""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import unittest
import numpy as np
import scipy
import psutil
from iteration21_worker import save
from iteration10_reference import core_parameters
from verify_iteration11 import RecordingResult
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.banded_gaussian_mps import pairing_covariance,pairing_log_normalization
from gbslab.gaussian_schmidt_mps import apply_gaussian_kernel,top_occupations
from gbslab.gaussian_mps_feasibility import equal_core_spectra,marginal_photon_tails
from gbslab.gaussian import haar_unitary
from gbslab.local_mps import local_circuit
from gbslab.mps_probe import mps_to_dense


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    source_files=('experiments/verify_iteration21.py','src/gbslab/gaussian_schmidt_mps.py','tests/test_gaussian_schmidt_mps.py')
    source_hashes={}
    for f in source_files:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
        source_hashes[f]=hashlib.sha256(dest.read_bytes()).hexdigest()
    save(out/'source_hashes_at_start.json',source_hashes)
    save(out/'environment.json',dict(platform=platform.platform(),python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
        logical_cpus=psutil.cpu_count(),physical_cpus=psutil.cpu_count(logical=False),physical_memory_bytes=psutil.virtual_memory().total,
        longdouble_mantissa_bits=int(np.finfo(np.longdouble).nmant),threads={k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    check=lambda name,value:checks.append(dict(check=name,passed=bool(value)))
    read=lambda p:json.loads(p.read_text()); pars=core_parameters(.9,.65); count=cuts=quality=samples=failures=0; failed_controls=[]
    for phase,expected_count in (('pilot',1),('panel',58)):
        panel=ROOT/f'results/iteration21_{phase}'; reference=ROOT/f'results/iteration21_{phase}_reference'
        plan=read(panel/'plan.json'); rows=read(panel/'rows.json'); references=read(reference/'rows.json')
        check(phase+' all planned work',len(plan['jobs'])==len(rows)==len(references)==expected_count and read(panel/'completion.json')['finished']==expected_count)
        hashes=read(panel/'metadata.json')['source_sha256']
        for f,digest in hashes.items():
            check(phase+' frozen source '+f,hashlib.sha256((panel/'source_snapshot'/f).read_bytes()).hexdigest()==digest)
            check(phase+' current source '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
        digest=read(reference/'metadata.json')['source_sha256']
        check(phase+' independent source',hashlib.sha256((reference/'iteration21_reference.py').read_bytes()).hexdigest()==digest==hashlib.sha256((ROOT/'experiments/iteration21_reference.py').read_bytes()).hexdigest())
        for i,(job,row,replay) in enumerate(zip(plan['jobs'],rows,references)):
            label=f'{phase}/{i}'; m=job['modes']; d=job['dimension']; rank=job['rank']; stem=panel/f'run_{i:02d}'
            check(label+' identity',row['sequence']==i and replay['sequence']==i and all(row[k]==v for k,v in job.items()))
            check(label+' raw row',dict(read(stem.with_suffix('.json')),sequence=i)==row)
            check(label+' terminal construction record',row['status'] in ('accepted','budget_missed','resource_rejected','time_rejected','failed','worker_failed'))
            if row['status'] not in ('accepted','budget_missed'):
                failures+=1
                check(label+' failure retained without acceptance',bool(row.get('error')) and not row['target_met'] and replay['status']=='not_constructed')
                rng=np.random.default_rng(job['seed']); u=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,job['depth'],rng)[1]
                spectrum=equal_core_spectra(u,pars['core_squeezing'],rank)
                lower=math.sqrt(max(max(0.,1-math.fsum(c['top_weights'])) for c in spectrum['cuts']))
                failed_controls.append(dict(phase=phase,sequence=i,modes=m,seed=job['seed'],ensemble=job['ensemble'],rank=rank,max_cut_rank_distance_lower_estimate=lower,scope='Pure-state rank obstruction only; not threshold-output TV'))
                continue
            count+=1; info=row['info']
            with np.load(stem.with_suffix('.npz')) as data: tensors=[data[f'A{k}'] for k in range(m)]
            with np.load(stem.with_suffix('.maps.npz')) as data:
                u=data['U']; B=data['pairing']; scalar=complex(data['final_scalar']); kernels=[data[f'K{k}'] for k in range(m)]
                bases=[data[f'basis{k}'] for k in range(m+1)]; weights=[data[f'weights{k}'] for k in range(m-1)]
            rng=np.random.default_rng(job['seed']); expected=haar_unitary(m,rng) if job['ensemble']=='haar' else local_circuit(m,job['depth'],rng)[1]
            check(label+' frozen optical input',np.array_equal(u,expected) and np.allclose(B,-math.tanh(pars['core_squeezing'])*(u@u.T),atol=1e-15,rtol=0))
            check(label+' finite tensor shapes',all(a.shape[1]==d and np.all(np.isfinite(a)) for a in tensors) and tensors[0].shape[0]==tensors[-1].shape[2]==1 and all(tensors[k].shape[2]==tensors[k+1].shape[0] for k in range(m-1)))
            check(label+' norm and rank',abs(np.linalg.norm(tensors[0])-1)<1e-11 and max(a.shape[2] for a in tensors)==info['max_bond_actual']<=rank)
            check(label+' storage',sum(a.size for a in tensors)==info['stored_complex_elements'])
            for k,A in enumerate(tensors[1:],1):
                matrix=A.reshape(A.shape[0],-1); check(label+f' right canonical {k}',np.max(np.abs(matrix@matrix.conj().T-np.eye(A.shape[0])))<1e-11)
            independent=equal_core_spectra(u,pars['core_squeezing'],rank); residual=B.copy(); scalar_replay=complex(math.exp(pairing_log_normalization(B)))
            discrepancies=[]; cut_tails=[]
            for k,(record,K,w,original) in enumerate(zip(info['cuts'],kernels,weights,independent['cuts'])):
                l=k+1; cuts+=1; tag=label+f'/cut{l}'; q=np.asarray(record['parameters']); basis,expected_weights=top_occupations(q,rank)
                check(tag+' exact kept basis',np.array_equal(basis,bases[l]) and np.array_equal(w,expected_weights))
                length=max(len(w),len(original['top_weights'])); discrepancy=float(np.max(np.abs(np.pad(w,(0,length-len(w)))-np.pad(original['top_weights'],(0,length-len(original['top_weights']))))))
                discrepancies.append(discrepancy); check(tag+' independent original spectrum',discrepancy<1e-10)
                order=list(range(l,2*l))+list(range(l)); inverse=K[np.ix_(order,order)].conj()
                updated,factor=apply_gaussian_kernel(residual,inverse,l); recovered,back=apply_gaussian_kernel(updated,K,l); c=record['kernel_scale']; scalar_replay*=c*factor
                check(tag+' inverse Gaussian kernel',np.max(np.abs(recovered-residual))<1e-9 and abs(c*c*factor*back-1)<1e-9)
                V=pairing_covariance(updated); idx=list(range(l))+list(range(m,m+l)); nu=(1+q)/(1-q)
                check(tag+' residual virtual thermal basis',np.max(np.abs(V[np.ix_(idx,idx)]-np.diag(np.r_[nu,nu])))<1e-8)
                check(tag+' scalar normalization',abs(math.log(abs(scalar_replay))-pairing_log_normalization(updated))<1e-8)
                cut_tails.append(max(0.,1-math.fsum(w))); residual=updated
            check(label+' final Gaussian and scalar',np.max(np.abs(residual-kernels[-1]))<1e-12 and abs(scalar_replay-scalar)<1e-12)
            check(label+' all cuts accounted',len(info['cuts'])==m-1 and len(info['cut_tail_estimates'])==m-1 and np.allclose(cut_tails,info['cut_tail_estimates'],atol=1e-14,rtol=0))
            check(label+' saved independent error',abs(max(discrepancies)-row['max_independent_schmidt_weight_error'])<1e-14)
            marginal=[float(marginal_photon_tails(independent['normal_mean'],abs(independent['anomalous'][j,j]),[d])['tail_upper_estimates'][0]) for j in range(m)]
            check(label+' original physical marginal tails',np.allclose(marginal,info['marginal_tail_estimates'],atol=1e-13,rtol=0))
            loss=math.fsum(cut_tails)+math.fsum(marginal); bound=math.sqrt(min(1.,loss)); success=info['projection_success']
            check(label+' original-projector upper',abs(loss-info['original_loss_sum'])<1e-11 and abs(bound-row['construction_trace_budget'])<1e-10)
            check(label+' acceptance',row['target_met']==(bound<=job['target'])==(row['status']=='accepted'))
            check(label+' projection success consistency',0<success<=1+1e-8 and (loss>1 or success>=1-4*loss/(1+loss)**2-1e-9))
            with np.load(reference/f'run_{i:02d}.npz') as data:
                patterns=data['patterns']; independent_amp=data['independent_raw_amplitudes']; saved_amp=data['saved_raw_amplitudes']
            check(label+' recurrence replay',replay['status']=='complete' and replay['selected_amplitude_check'] and np.max(np.abs(independent_amp-saved_amp))<1e-10)
            for pattern,amp in zip(patterns,saved_amp):
                value=np.ones(1,complex)
                for A,n in zip(tensors,pattern): value=value@A[:,int(n),:]
                check(label+' saved amplitude binding',abs(value[0]*math.sqrt(success)-amp)<1e-13)
            if job['panel']=='quality':
                quality+=1
                with np.load(stem.with_suffix('.reference.npz')) as data: original=data['gaussian_box']
                state=mps_to_dense(tensors); actual=math.sqrt(max(0.,1-abs(np.vdot(original,state))**2))
                check(label+' actual original Gaussian distance',abs(actual-row['actual_state_distance'])<1e-12 and actual<=bound+1e-10)
                check(label+' full recurrence tensor replay',replay['full_tensor_check'] and replay['full_tensor_max_error']<1e-10 and abs(replay['independent_success']-success)<1e-10)
            with np.load(stem.with_suffix('.samples.npz')) as data: alpha=data['alpha']; observed=data['samples']
            z=np.random.default_rng(job['seed']+10000).standard_normal((job['samples'],m)); expected_alpha=.5j*math.sqrt(pars['residual_variance'])*(z@u.T)
            check(label+' original displacement',np.array_equal(alpha,expected_alpha))
            check(label+' observed sampling',observed.shape==(job['samples'],m) and np.all((observed==0)|(observed==1)) and row['completed_samples']==job['samples'] and row['sampling_status']=='complete')
            samples+=len(observed)
        print('verified',phase,flush=True)
    check('all terminal records and completed samples',count+failures==59 and samples==16*count)
    save(out/'artifact_checks.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),states=count,construction_failures=failures,cuts=cuts,quality_cases=quality,samples=samples,failed_case_rank_lower_estimates=failed_controls,checks=checks))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result=dict(tests_run=r.testsRun,passed=len(r.successes),failures=len(r.failures),errors=len(r.errors),
        skipped_cases=sum('setUpClass' not in t.id() for t,_ in r.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in r.skipped),
        new_tests_passed=sum(s.startswith('test_gaussian_schmidt_mps.') for s in r.successes),
        failure_details=[dict(test=t.id(),traceback=msg) for t,msg in r.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in r.errors],
        skip_details=[dict(test=t.id(),reason=msg) for t,msg in r.skipped],full_gate_passed=r.wasSuccessful() and not r.skipped)
    old=read(ROOT/'results/iteration20_validation/validation.json')
    result['unchanged_failure_ids']={x['test'] for k in ('failure_details','error_details') for x in old[k]}=={x['test'] for k in ('failure_details','error_details') for x in result[k]}
    save(out/'validation.json',result)
    if any(hashlib.sha256((ROOT/f).read_bytes()).hexdigest()!=digest for f,digest in source_hashes.items()):
        raise RuntimeError('Verifier/source changed during audit; preserve output and repeat in a new directory')
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2)); print('Artifact checks',sum(c['passed'] for c in checks),'/',len(checks))
    if not all(c['passed'] for c in checks): raise AssertionError('Artifact checks failed; preserve this output directory')


if __name__=='__main__': main()
