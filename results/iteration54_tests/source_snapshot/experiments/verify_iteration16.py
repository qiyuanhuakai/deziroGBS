"""Check rolling arithmetic, checkpoint acceptance, failed prefixes and saved states."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import unittest
import numpy as np
from iteration14_run import save
from verify_iteration11 import RecordingResult

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.mps_probe import mps_to_dense


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    def check(label,value): checks.append({'check':label,'passed':bool(value)})
    panel=ROOT/'results/iteration16_pilot'; plan=json.loads((panel/'plan.json').read_text()); rows=json.loads((panel/'rows.json').read_text())
    check('all planned runs retained',len(rows)==plan['case_count']==18)
    for f,digest in json.loads((panel/'metadata.json').read_text())['source_sha256'].items():
        check('frozen source '+f,hashlib.sha256((panel/'source_snapshot'/f).read_bytes()).hexdigest()==digest)
    for i,(row,job) in enumerate(zip(rows,plan['jobs'])):
        check(f'{i} planned identity',all(row[k]==v for k,v in job.items()))
        trial_path=panel/f'run_{i:02d}.attempts.json'
        trials=json.loads(trial_path.read_text()) if trial_path.exists() else []
        if row['status']=='accepted':
            check(f'{i} final accepted target',row['target_met'] and row['construction_trace_budget']<=row['target'] and trials[-1]['status']=='accepted')
        elif row['status']=='search_unaccepted':
            check(f'{i} failed attempts retained',len(trials)>0 and all(t['status']!='accepted' for t in trials))
        else: check(f'{i} explicit other failure',row['status'] in ('worker_failed','failed','sampling_failed'))
        for j,t in enumerate(trials):
            label=f'{i} attempt {j}'
            if row['method']=='restart':
                if 'construction_trace_budget' in t:
                    total=t['input_trace_budget']+t['projection_trace_sum']+t['rounding_trace_sum']
                    check(label+' restart complete budget',abs(total-t['unclipped_construction_trace_budget'])<1e-13)
                continue
            if t['status']=='input_budget_rejected':
                check(label+' input rejection',t['input_trace_budget']>row['target']); continue
            projection=0.; rounding=0.
            for k,layer in enumerate(t['layers']):
                projection+=math.sqrt(layer['projection_tail']); accepted=[]
                for n,scan in enumerate(layer['rounding_attempts']):
                    sub=label+f' layer {k} retry {n}'
                    q=scan['conditional_losses']; joint=-math.expm1(sum(math.log1p(-x) for x in q))
                    check(sub+' joint tail',abs(joint-scan['joint_tail'])<1e-14)
                    check(sub+' rank dimensions',len(scan['ranks'])==row['modes']-1 and all(1<=r<=scan['cap'] for r in scan['ranks']))
                    check(sub+' real acceptance',(joint<=scan['joint_allowance'])==scan['accepted'])
                    if 'per_cut_allowances' in scan:
                        spent=0.; good=True
                        for c,(loss,allow) in enumerate(zip(q,scan['per_cut_allowances'])):
                            expected=-math.expm1(min(0.,math.log1p(-scan['joint_allowance'])-spent)/(len(q)-c))
                            good&=abs(expected-allow)<1e-15; spent+=math.log1p(-loss)
                        check(sub+' rolling allocation recurrence',good)
                    if scan['accepted']: accepted.append(scan)
                check(label+f' layer {k} at most one accepted retry',len(accepted)<=1)
                if 'selected_cap' in layer:
                    check(label+f' layer {k} selected last accepted',len(accepted)==1 and accepted[0] is layer['rounding_attempts'][-1] and layer['selected_cap']==accepted[0]['cap'])
                    rounding+=math.sqrt(layer['rounding_tail'])
            total=t['input_trace_budget']+projection+rounding
            check(label+' ledger sums',abs(projection-t['projection_trace_sum'])<1e-13 and abs(rounding-t['rounding_trace_sum'])<1e-13 and abs(total-t['partial_ledger_sum'])<1e-13)
            if t['status']=='projection_budget_exhausted': check(label+' early stop justification',total>row['target'])
            if t['status'] in ('accepted','budget_missed'):
                check(label+' complete budget',abs(total-t['unclipped_construction_trace_budget'])<1e-13)
                expected=sum((row['modes']-offset)//2 for offset in (x%2 for x in range(row['depth'])))
                check(label+' each optical gate once',t['completed_gates']==len(t['gates'])==expected)
            else: check(label+' partial state not certified','construction_trace_budget' not in t)
        state=panel/f'run_{i:02d}.npz'
        if state.exists():
            with np.load(state) as data: tensors=[data[f'A{k}'] for k in range(row['modes'])]
            check(f'{i} final coefficient count',sum(a.size for a in tensors)==row['stored_complex_elements'])
            check(f'{i} normalized center',abs(np.linalg.norm(tensors[0])-1)<1e-10)
            check(f'{i} right canonical',all(np.allclose(a.reshape(a.shape[0],-1)@a.reshape(a.shape[0],-1).conj().T,np.eye(a.shape[0]),atol=1e-10,rtol=0) for a in tensors[1:]))
            if row['panel']=='quality':
                with np.load(panel/f'run_{i:02d}.reference.npz') as ref:
                    actual=mps_to_dense(tensors); overlap=np.vdot(ref['gaussian_box'].ravel(),actual.ravel())
                distance=math.sqrt(max(0,1-abs(overlap)**2))
                check(f'{i} actual small state distance',abs(distance-row['state_trace_distance'])<1e-12)
                check(f'{i} actual state bounded',distance<=row['construction_trace_budget']+1e-10)
        sample_file=panel/f'run_{i:02d}.samples.npz'
        if sample_file.exists():
            with np.load(sample_file) as data: samples=data['samples']
            check(f'{i} saved samples',samples.shape==(row['completed_samples'],row['modes']) and np.all((samples==0)|(samples==1)))
    spectral=ROOT/'results/iteration16_schmidt'
    spectrum_rows=json.loads((spectral/'rows.json').read_text())
    check('all Schmidt rank rows',len(spectrum_rows)==168)
    for file,digest in json.loads((spectral/'source_hashes.json').read_text()).items():
        check('spectral source '+file,hashlib.sha256((spectral/'source_snapshot'/file).read_bytes()).hexdigest()==digest)
    for index,row in enumerate(spectrum_rows):
        with np.load(spectral/f"spectrum_{row['input_index']:02d}_cut{row['cut']}.npz") as data:
            weights=data['top_weights']; parameters=data['parameters']
        tail=max(0.,1-math.fsum(weights[:row['rank']]))
        check(f'spectral {index} parameters',np.all((parameters>=0)&(parameters<1)))
        check(f'spectral {index} sorted weights',np.all(weights>=0) and np.all(np.diff(weights)<=1e-15) and math.fsum(weights)<=1+1e-12)
        check(f'spectral {index} reported obstruction',abs(math.sqrt(tail)-row['pure_state_distance_lower_bound'])<1e-14)
    save(out/'artifact_checks.json',{'count':len(checks),'passed':sum(x['passed'] for x in checks),'checks':checks})
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result={'tests_run':r.testsRun,'passed':len(r.successes),'failures':len(r.failures),'errors':len(r.errors),
        'skipped_cases':sum('setUpClass' not in t.id() for t,_ in r.skipped),'skipped_classes':sum('setUpClass' in t.id() for t,_ in r.skipped),
        'checkpoint_tests_passed':sum(s.startswith('test_checkpoint_mps.') for s in r.successes),
        'failure_details':[{'test':t.id(),'traceback':msg} for t,msg in r.failures],
        'error_details':[{'test':t.id(),'traceback':msg} for t,msg in r.errors],
        'skip_details':[{'test':t.id(),'reason':msg} for t,msg in r.skipped],
        'full_gate_passed':r.wasSuccessful() and not r.skipped}
    old=json.loads((ROOT/'results/iteration15_validation/validation.json').read_text())
    result['unchanged_failure_ids']=set(x['test'] for k in ('failure_details','error_details') for x in old[k])==set(x['test'] for k in ('failure_details','error_details') for x in result[k])
    save(out/'validation.json',result)
    for file in ('experiments/verify_iteration16.py','tests/test_checkpoint_mps.py','src/gbslab/checkpoint_mps.py'):
        dest=out/'source_snapshot'/file; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/file,dest)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2))
    print('Artifact checks:',sum(x['passed'] for x in checks),'/',len(checks))
    return 0 if r.wasSuccessful() and all(x['passed'] for x in checks) else 1


if __name__=='__main__': main()
