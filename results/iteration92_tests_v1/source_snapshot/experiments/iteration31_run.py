"""Full-law comparisons and 144-mode operational runs of the wire forest."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,shutil,sys,traceback
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.wire_forest import WireForestSampler
from gbslab.batched_forest import install_canonical_selector
from gbslab.positive_forest import ForestCompletionSampler
from gbslab.positive_completion import TableEventOracle,PositiveCompletionSamplingError
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.exact import ThresholdModel
from iteration23_support import save,digest

def plain(value):
    from fractions import Fraction
    if isinstance(value,Fraction):return dict(numerator=str(value.numerator),denominator=str(value.denominator))
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    if isinstance(value,np.generic):return value.item()
    return value

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['experiments/iteration31_run.py',*[x.relative_to(ROOT).as_posix() for x in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(ROOT/'research/iteration31_panel_plan.md',out/'frozen_plan.md');target='results/iteration28_panel/case_14.npz';dependency={target:digest(ROOT/target)};assert dependency[target]==json.loads((ROOT/'results/iteration28_manifest.json').read_text())['artifacts'][target]
    old=np.load(ROOT/target);V=pairing_covariance(old['full_pairing'])+old['full_noise_factor']@old['full_noise_factor'].T;symmetric=(V+V.T)/2;delta=float(np.max(abs(symmetric-V)));large_model=ThresholdModel(symmetric);np.savez_compressed(out/'large_input.npz',before_symmetry=V,covariance=large_model.V);assert np.array_equal(large_model.V,symmetric)
    jobs=[dict(panel='small',modes=m,seed=seed,budget=k,forest=forest,selection='recent' if seed==311201 else 'click_influence',samples=128) for m in (4,6) for seed in (311201,311202) for k in (0,2) for forest in ('chain','static_mi_tree')]
    jobs += [dict(panel='large',modes=144,seed=311301,budget=k,forest=forest,selection='recent',samples=8) for k in (0,2) for forest in ('chain','static_mi_tree')]
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependency,library_sha256=digest(a.library),symmetrization_maximum_change=delta))
    rows=[]
    for i,job in enumerate(jobs):
        start=perf_counter();row=dict(sequence=i,**job);sampler=None;print('starting',i,job,flush=True)
        try:
            if job['panel']=='small':
                m=job['modes'];rng=np.random.default_rng(job['seed']);model=ThresholdModel(covariance_from_interferometer(haar_unitary(m,rng),np.linspace(.45,.9,m),np.linspace(.55,.9,m)));target_law=model.full_distribution();order=tuple(rng.permutation(m))
            else:model=large_model;order=tuple(range(144))
            sampler=WireForestSampler(model,job['budget'],forest=job['forest'],selection=job['selection'],order=order,cache_size=8192,library=a.library);row['preparation_seconds']=perf_counter()-start
            if job['panel']=='small':
                reference=ForestCompletionSampler(model,job['budget'],forest=job['forest'],selection=job['selection'],order=order,cache_size=8192,event_oracle=TableEventOracle(target_law));install_canonical_selector(reference);conditionals=[];max_error=0.
                for length in range(m):
                    for mask in range(1<<length):
                        prefix=tuple((mask>>j)&1 for j in range(length));actual=sampler.conditional_details(prefix);truth=reference.conditional_details(prefix);assert actual['retained_clicks']==truth['retained_clicks'] and actual['forest_edges']==truth['forest_edges'];error=abs(actual['click_probability']-truth['click_probability']);max_error=max(max_error,error);conditionals.append([actual['click_probability'],truth['click_probability']])
                law=sampler.enumerate_law();ref_law=reference.enumerate_law();tv=.5*float(np.sum(abs(law['extended_distribution']-ref_law['extended_distribution'])));assert tv<2e-10
                samples=sampler.sample(job['samples'],np.random.default_rng(311400+i));reference_samples=reference.sample(job['samples'],np.random.default_rng(311400+i));assert np.array_equal(samples,reference_samples)
                arrays=dict(covariance=model.V,order=np.array(order),target_law=target_law,native_extended_law=law['extended_distribution'],reference_extended_law=ref_law['extended_distribution'],conditionals=np.array(conditionals),samples=samples,reference_samples=reference_samples)
                row.update(maximum_conditional_error=max_error,extended_law_tv=tv,native_failure_mass=float(law['failure_mass']),reference_failure_mass=float(ref_law['failure_mass']),same_uniform_samples=True,conditional_count=len(conditionals),target_tv=float(.5*np.sum(abs(law['distribution']-target_law))))
            else:
                sampling_start=perf_counter()
                try:samples=sampler.sample(job['samples'],np.random.default_rng(job['seed']));row['sampling_outcome']='completed'
                except PositiveCompletionSamplingError as failure:
                    samples=failure.successful_samples;row.update(sampling_outcome='numerical_failure',failed_sample_index=failure.sample_index,failed_prefix=list(failure.failed_prefix),failure=str(failure))
                row.update(sampling_seconds=perf_counter()-sampling_start,successful_samples=len(samples));arrays=dict(samples=samples,order=np.array(order))
            np.savez_compressed(out/f'case_{i:02d}.npz',**arrays);row['native_diagnostics']=plain(sampler.kernel.cache_info());row['sampler_diagnostics']=plain(sampler.cache_info());row['status']='completed'
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        finally:
            if sampler is not None:sampler.close()
        row['seconds']=perf_counter()-start;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row['status'],row.get('sampling_outcome'),flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=dependency))
if __name__=='__main__':main()
