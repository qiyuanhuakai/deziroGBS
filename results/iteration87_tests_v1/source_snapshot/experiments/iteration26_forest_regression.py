"""Paired old/new full surrogate laws, identical seeded samples, and guards."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from gbslab.positive_forest import ForestCompletionSampler
from gbslab.exact import ThresholdModel,NumericalProbabilityError
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from positive_event_fixtures import ConditionalIndependentEvents
from iteration23_support import digest,save


def main():
    out=ROOT/'results/iteration26_forest_regression'; out.mkdir(exist_ok=False)
    sources=sorted({'experiments/iteration26_forest_regression.py','experiments/iteration23_support.py','tests/positive_event_fixtures.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))}); hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dst=out/'source_snapshot'/f; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dst)
    oldpath='results/iteration25_tests/source_snapshot/src/gbslab/positive_forest.py'; oldsha=digest(ROOT/oldpath); manifest=json.loads((ROOT/'results/iteration25_manifest.json').read_text()); assert manifest['artifacts'][oldpath]==oldsha
    jobs=[dict(modes=m,seed=s,k=k,forest=f) for m in (4,6) for s in (260701,260702) for k in (0,2) for f in ('chain','static_mi_tree')]
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies={oldpath:oldsha},squeezing=.6,loss=.7,sample_seed=260999,samples=32,law_tv_tolerance=1e-11))
    shutil.copy2(ROOT/'research/iteration26_regression_plan.md',out/'frozen_plan.md')
    prior=json.loads((ROOT/'results/iteration25_tests/source_hashes_at_start.json').read_text())
    unchanged=['src/gbslab/positive_completion.py','src/gbslab/zero_memory.py','src/gbslab/conditional_memory.py','src/gbslab/exact.py','src/gbslab/gaussian.py']
    assert all(digest(ROOT/f)==prior[f] for f in unchanged)
    spec=importlib.util.spec_from_file_location('gbslab._iteration25_forest',ROOT/oldpath); module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module); Old=module.ForestCompletionSampler
    rows=[]
    for i,j in enumerate(jobs):
        row=dict(j,sequence=i)
        try:
            U=haar_unitary(j['modes'],np.random.default_rng(j['seed'])); V=covariance_from_interferometer(U,.6,.7); model=ThresholdModel(V)
            old=Old(model,j['k'],selection='recent',forest=j['forest']); new=ForestCompletionSampler(model,j['k'],selection='recent',forest=j['forest'])
            p=old.full_distribution(); q=new.full_distribution(); a=old.sample(32,np.random.default_rng(260999)); b=new.sample(32,np.random.default_rng(260999))
            tv=.5*float(np.sum(abs(p-q))); equal=np.array_equal(a,b); row.update(law_tv=tv,samples_equal=equal,status='passed' if tv<1e-11 and equal else 'mismatch')
            np.savez_compressed(out/f'case_{i:02d}.npz',U=U,covariance=V,old_distribution=p,new_distribution=q,old_samples=a,new_samples=b); row['output_sha256']=digest(out/f'case_{i:02d}.npz')
        except Exception: row.update(status='failed',traceback=traceback.format_exc())
        rows.append(row); save(out/'rows.json',rows)
    boundaries=[]
    for cls,name in ((Old,'old'),(ForestCompletionSampler,'new')):
        for forest in ('chain','static_mi_tree'):
            for odds in (-800,800):
                oracle=ConditionalIndependentEvents(odds); sampler=cls(ThresholdModel(1.8*np.eye(12)),0,selection='recent',forest=forest,event_oracle=oracle)
                try: sampler.conditional_click_probability((1,)*5); passed=False
                except NumericalProbabilityError as e: passed='Bernoulli endpoint' in str(e)
                boundaries.append(dict(version=name,forest=forest,odds=odds,kind='positive_endpoint_rejected',passed=passed,all_queried_events_positive=all(e[2]>0 for e in oracle.queried)))
            vacuum=cls(ThresholdModel(np.eye(8)),0,selection='recent',forest=forest).full_distribution(); expected=np.zeros(16); expected[0]=1
            boundaries.append(dict(version=name,forest=forest,kind='analytic_vacuum_exact',passed=np.array_equal(vacuum,expected)))
    save(out/'boundaries.json',boundaries); assert hashes=={f:digest(ROOT/f) for f in sources}
    summary=dict(planned=16,finished=len(rows),passed=sum(r['status']=='passed' for r in rows),maximum_law_tv=max(r.get('law_tv',0.) for r in rows),paired_samples=sum(32 for r in rows if r.get('samples_equal')),boundary_checks=len(boundaries),boundary_passed=sum(r['passed'] for r in boundaries),auxiliary_sources_unchanged=unchanged,unchanged_sources=True,dependencies={oldpath:oldsha})
    save(out/'summary.json',summary); print(json.dumps(summary,indent=2)); assert summary['passed']==16 and summary['boundary_passed']==12


if __name__=='__main__': main()
