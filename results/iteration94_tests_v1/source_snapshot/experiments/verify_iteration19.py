"""Audit all cuts, marginal tails, existence arithmetic and actual small MPS."""
import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import unittest
import mpmath as mp
import numpy as np
from iteration14_run import save
from verify_iteration11 import RecordingResult
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.mps_probe import mps_to_dense,geometric_top_weights


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); checks=[]
    check=lambda name,value:checks.append(dict(check=name,passed=bool(value)))
    read=lambda p:json.loads(p.read_text())
    panel=ROOT/'results/iteration19_panel'; ref=ROOT/'results/iteration19_reference'
    plan=read(panel/'plan.json'); inputs=read(panel/'inputs.json'); rows=read(panel/'rows.json'); built=read(panel/'constructions.json')
    check('all inputs, rows, constructions',len(inputs)==20 and len(rows)==700 and len(built)==72 and all(r['status']=='complete' for r in inputs+rows+built))
    check('all row identities unique',len({(r['index'],r['dimension'],r['rank']) for r in rows})==700)
    check('all construction identities unique',len({(r['index'],r['dimension'],r['rank'],r['method']) for r in built})==72)
    for f,digest in read(panel/'metadata.json')['source_sha256'].items():
        check('frozen source '+f,hashlib.sha256((panel/'source_snapshot'/f).read_bytes()).hexdigest()==digest)
        check('current source '+f,hashlib.sha256((ROOT/f).read_bytes()).hexdigest()==digest)
    digest=read(ref/'source_hashes.json')['experiments/iteration19_reference.py']
    check('reference source',hashlib.sha256((ref/'iteration19_reference.py').read_bytes()).hexdigest()==digest==hashlib.sha256((ROOT/'experiments/iteration19_reference.py').read_bytes()).hexdigest())
    cut_count=0; marginal_count=0
    for i,(job,item) in enumerate(zip(plan['jobs'],inputs)):
        m=job['modes']; check(f'{i} input identity',item['index']==i and all(item[k]==v for k,v in job.items()))
        with np.load(panel/f'input_{i:02d}.npz') as data:
            u=data['U']; M=data['anomalous']; tails=data['marginal_tail_upper_estimates']; probs=data['marginal_probabilities']; remainder=data['remainder_upper_estimates']
        check(f'{i} unitary',np.allclose(u.conj().T@u,np.eye(m),atol=1e-12,rtol=0))
        check(f'{i} symplectic independent check',item['independent_symplectic_max_error']<1e-12)
        spectra=[]
        for cut in range(1,m):
            with np.load(panel/f'input_{i:02d}_cut{cut:03d}.npz') as data: sigma=data['cross_singular_values']; q=data['parameters']; weights=data['top_weights']
            check(f'{i}/{cut} cross singular spectrum',np.allclose(sigma,np.linalg.svd(M[:cut,cut:],compute_uv=False),atol=1e-13,rtol=0))
            check(f'{i}/{cut} stable geometric conversion',np.allclose(q,4*sigma**2/(np.sqrt(1+4*sigma**2)+1)**2,atol=1e-15,rtol=0))
            check(f'{i}/{cut} ordered probability weights',np.all(weights>=0) and np.all(np.diff(weights)<=1e-15) and math.fsum(weights)<=1+1e-12)
            check(f'{i}/{cut} saved heap spectrum',np.array_equal(weights,geometric_top_weights(q,512)))
            spectra.append(weights); cut_count+=1
        with mp.workdps(140):
            reference=read(ref/f'input_{i:02d}.json')
            for mode in range(m):
                marginal_count+=1
                check(f'{i}/{mode} nonnegative PNR and remainder',np.all(probs[mode]>=0) and remainder[mode]>=0)
                check(f'{i}/{mode} PNR normalization',abs(math.fsum(probs[mode])-1)<1e-12)
                for k,d in enumerate(plan['dimensions']):
                    hi=mp.mpf(reference['reference120'][mode][k]); lo=mp.mpf(reference['reference80'][mode][k])
                    check(f'{i}/{mode}/d{d} reference convergence',abs(hi-lo)<mp.mpf('1e-65'))
                    check(f'{i}/{mode}/d{d} relative tail accuracy',abs(mp.mpf(float(tails[mode,k]))-hi)/hi<mp.mpf('1e-11'))
                    check(f'{i}/{mode}/d{d} suffix plus remainder',abs(math.fsum(probs[mode,d:])+remainder[mode]-tails[mode,k])<1e-25)
        for row in (r for r in rows if r['index']==i):
            rank=row['rank']; j=plan['dimensions'].index(row['dimension']); st=[max(0.,1-math.fsum(w[:rank])) for w in spectra]
            box=min(1.,math.fsum(tails[:,j])); total=box+math.fsum(st); label=f'{i}/d{row["dimension"]}/r{rank}'
            check(label+' complete all-cut tail sum',abs(math.fsum(st)-row['schmidt_tail_sum'])<1e-13)
            check(label+' finite box and existence bound',abs(box-row['box_tail_union_estimate'])<1e-15 and abs(math.sqrt(min(1.,total))-row['existence_trace_upper_estimate'])<1e-13)
            check(label+' maximum cut obstruction',abs(math.sqrt(max(st))-row['max_cut_rank_distance_lower_estimate'])<1e-13)
            check(label+' acceptance is existence only',row['target_met']==(row['existence_trace_upper_estimate']<=plan['target']))
    check('all 980 cuts and 1000 marginals',cut_count==980 and marginal_count==1000)
    for row in built:
        i=row['index']; d=row['dimension']; rank=row['rank']; method=row['method']; label=f'{i}/d{d}/r{rank}/{method}'
        with np.load(panel/f'input_{i:02d}_box{d}.npz') as data: box=data['box']
        with np.load(panel/f'input_{i:02d}_d{d}_r{rank}_{method}.npz') as data: tensors=[data[f'A{k}'] for k in range(row['modes'])]
        state=mps_to_dense(tensors); overlap=np.vdot(box.ravel(),state.ravel()); actual=math.sqrt(max(0.,1-abs(overlap)**2))
        check(label+' normalized finite state',abs(np.linalg.norm(state)-1)<1e-12 and state.shape==box.shape)
        check(label+' bond cap and storage',max(a.shape[2] for a in tensors)<=rank and sum(a.size for a in tensors)==row['stored_complex_elements'])
        check(label+' actual full Gaussian overlap',abs(actual-row['state_trace_distance'])<1e-12)
        check(label+' actual state respects applicable bound',actual<=row['bound']+1e-10)
        if method=='direct_box_tt':
            info=row['info']; mass=float(np.vdot(box.ravel(),box.ravel()).real); retained=mass*(1-info['discarded_sum'])
            check(label+' nested TT projection identity',abs(abs(overlap)**2-retained)<1e-12)
            original_tails=[]
            for cut in range(1,row['modes']): original_tails.append(float(np.sum(np.linalg.svd(box.reshape(d**cut,-1),compute_uv=False)[rank:]**2)))
            check(label+' TT error versus original box cuts',mass*info['discarded_sum']<=sum(original_tails)+1e-12)
            check(label+' box union bound',1-mass<=next(r['box_tail_union_estimate'] for r in rows if r['index']==i and r['dimension']==d and r['rank']==rank)+1e-12)
    save(out/'artifact_checks.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
    with (out/'tests.log').open('w',encoding='utf-8') as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
        r=unittest.TextTestRunner(stream=log,verbosity=2,resultclass=RecordingResult).run(suite)
    result=dict(tests_run=r.testsRun,passed=len(r.successes),failures=len(r.failures),errors=len(r.errors),
        skipped_cases=sum('setUpClass' not in t.id() for t,_ in r.skipped),skipped_classes=sum('setUpClass' in t.id() for t,_ in r.skipped),
        new_tests_passed=sum(s.startswith('test_gaussian_mps_feasibility.') for s in r.successes),
        failure_details=[dict(test=t.id(),traceback=msg) for t,msg in r.failures],error_details=[dict(test=t.id(),traceback=msg) for t,msg in r.errors],
        skip_details=[dict(test=t.id(),reason=msg) for t,msg in r.skipped],full_gate_passed=r.wasSuccessful() and not r.skipped)
    old=read(ROOT/'results/iteration18_validation/validation.json')
    result['unchanged_failure_ids']={x['test'] for k in ('failure_details','error_details') for x in old[k]}=={x['test'] for k in ('failure_details','error_details') for x in result[k]}
    save(out/'validation.json',result)
    for f in ('experiments/verify_iteration19.py','src/gbslab/gaussian_mps_feasibility.py','tests/test_gaussian_mps_feasibility.py'):
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)},indent=2)); print('Artifact checks',sum(c['passed'] for c in checks),'/',len(checks))

if __name__=='__main__': main()
