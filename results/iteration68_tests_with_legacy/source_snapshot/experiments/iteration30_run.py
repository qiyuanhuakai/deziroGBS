"""Full threshold strings from positive finite-input POVM trajectories."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,shutil,sys,traceback
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from gbslab.displaced_sampling import DisplacedThresholdSampler,prior_displacements
from gbslab.mps_probe import dense_to_mps
from test_displaced_sampling import dense_threshold_law,enumerate_refined,mixed_dense_law
from iteration23_support import save,digest,load_mps,mps_fingerprint
from iteration22_global import load_family
from iteration22_external import ARCHIVE_SHA

def statistics(clicks,single,pairs,starts):
    n,m=clicks.shape;empirical_single=1-clicks.mean(axis=0)
    empirical_pairs=np.array([np.bincount(clicks[:,i]+2*clicks[:,i+1],minlength=4)/n for i in starts])
    expected=np.r_[single,pairs.ravel()];observed=np.r_[empirical_single,empirical_pairs.ravel()];se=np.sqrt(expected*(1-expected)/n)
    return dict(empirical_single_vacuum=empirical_single,empirical_pair_threshold=empirical_pairs,click_histogram=np.bincount(clicks.sum(axis=1).astype(int),minlength=m+1)),dict(mean_clicks=float(clicks.sum(axis=1).mean()),expected_mean_clicks=float(np.sum(1-single)),maximum_marginal_absolute_error=float(np.max(abs(observed-expected))),maximum_marginal_standardized_error=float(np.max(abs(observed-expected)/se)))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['experiments/iteration30_run.py','experiments/iteration23_support.py','experiments/iteration22_global.py','experiments/iteration22_external.py','tests/test_displaced_sampling.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]]
    hashes={p:digest(ROOT/p) for p in names}
    for path in names:
        dest=out/'source_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest)
    shutil.copy2(ROOT/'research/iteration30_panel_plan.md',out/'frozen_plan.md')
    dependencies={};old29=json.loads((ROOT/'results/iteration29_manifest.json').read_text())['artifacts'];old25=json.loads((ROOT/'results/iteration25_manifest.json').read_text())['artifacts'];old28=json.loads((ROOT/'results/iteration28_manifest.json').read_text())['artifacts']
    def require(path,manifest):
        sha=digest(ROOT/path);assert sha==manifest[path];dependencies[path]=sha;return ROOT/path
    target='results/iteration28_panel/case_14.npz';F=np.load(require(target,old28))['full_noise_factor'];rng=np.random.default_rng(301301);standard=rng.normal(size=(4096,F.shape[1]));uniforms=rng.random((4096,144));beta=prior_displacements(F,standard)
    np.savez_compressed(out/'large_random.npz',standard=standard,uniforms=uniforms,beta=beta)
    jobs=[dict(panel='small',source=i,samples=65536,seed=301400+i) for i in range(16)]+[dict(panel='large',family=family,samples=4096,seed=301301,source=16+i) for i,family in enumerate(('new_32','new_64','new_128','new_200','author_pco','author_hafnian'))]
    assert digest(args.archive)==ARCHIVE_SHA
    for i in range(16):require(f'results/iteration29_panel/case_{i:02d}.reference.npz',old29)
    for i in range(16,22):require(f'results/iteration29_panel/case_{i:02d}.npz',old29)
    for i in range(16,20):require(f'results/iteration25_panel_retry/run_{i}.state.npz',old25)
    save(out/'plan.json',dict(jobs=jobs,dependencies=dependencies,source_hashes=hashes,archive_sha256=ARCHIVE_SHA,batch=128))
    rows=[]
    for i,job in enumerate(jobs):
        row=dict(sequence=i,**job);start=perf_counter();print('starting',i,job,flush=True)
        try:
            if job['panel']=='small':
                old=np.load(ROOT/f'results/iteration29_panel/case_{i:02d}.reference.npz');dense=old['dense'];noise=old['noise_factor'];m=dense.ndim;tensors,_=dense_to_mps(dense,128)
                rng=np.random.default_rng(job['seed']);z=rng.normal(size=(job['samples'],noise.shape[1]));u=rng.random((job['samples'],m));b=prior_displacements(noise,z)
                fixed=np.linspace(.2,.6,m)*(1+.7j);sampler=DisplacedThresholdSampler(tensors);law,trajectories=enumerate_refined(sampler,fixed);direct=dense_threshold_law(dense,fixed);reference=mixed_dense_law(dense,noise)
                error=float(np.max(abs(law-direct)));assert error<2e-11
                arrays=dict(conditional_beta=fixed,conditional_refined_law=law,conditional_dense_law=direct,mixed_reference=reference)
                row.update(conditional_max_error=error,enumerated_trajectories=trajectories)
            else:
                b,u=beta,uniforms;family=job['family']
                if family.startswith('new_'):tensors=load_mps(ROOT/f'results/iteration25_panel_retry/run_{16+int(np.where(np.array([32,64,128,200])==int(family[4:]))[0][0])}.state.npz')
                else:
                    tensors,entries=load_family(family,ROOT/'results/iteration22_external',args.archive);save(out/(family+'_entries.json'),entries)
                row['raw_mps_fingerprint']=mps_fingerprint(tensors);sampler=DisplacedThresholdSampler(tensors);arrays={}
            del tensors;row['preparation_seconds']=perf_counter()-start;row['input_log_norm_squared']=2*sampler.input_log_norm
            clicks=[];labels=[];residual=0.;sampling_start=perf_counter()
            for first in range(0,len(b),128):
                result=sampler.sample(b[first:first+128],u[first:first+128]);clicks.append(result['clicks']);labels.append(result['refined_labels'].astype(np.uint8));residual=max(residual,result['maximum_normalization_residual'])
                if first==0:
                    arrays.update(audit_beta=b[:16],audit_uniforms=u[:16],audit_selected_refined_probabilities=result['selected_refined_probabilities'][:16],audit_vacuum_given_hidden_history=result['vacuum_given_hidden_history'][:16])
                if job['panel']=='large' and first%1024==0:print('samples',i,first,flush=True)
            clicks=np.concatenate(clicks);labels=np.concatenate(labels);row.update(sampling_seconds=perf_counter()-sampling_start,maximum_normalization_residual=residual);arrays.update(clicks=clicks,refined_labels=labels)
            if job['panel']=='small':
                ids=clicks@(2**np.arange(m-1,-1,-1));histogram=np.bincount(ids,minlength=2**m);empirical=histogram/len(clicks);se=np.sqrt(reference*(1-reference)/len(clicks));arrays.update(histogram=histogram,empirical_law=empirical)
                row.update(empirical_full_tv=float(.5*np.sum(abs(empirical-reference))),maximum_standardized_error=float(np.max(abs(empirical-reference)/se)))
            else:
                old=np.load(ROOT/f'results/iteration29_panel/case_{i:02d}.npz');stats,summary=statistics(clicks,old['single_vacuum'],old['pair_threshold'],old['pair_starts']);arrays.update(stats);row.update(summary)
            np.savez_compressed(out/f'case_{i:02d}.npz',**arrays);row['status']='completed';del sampler
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row['status'],row['seconds'],flush=True)
    unchanged=hashes=={p:digest(ROOT/p) for p in names};assert unchanged
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=dependencies))
if __name__=='__main__':main()
