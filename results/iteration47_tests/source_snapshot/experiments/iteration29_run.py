"""Same-channel displaced MPS vacuum estimates and deterministic local witnesses."""
import argparse
import json
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from gbslab.displaced_vacuum import CoherentMPS,GaussianVacuumProposal,importance_summary
from gbslab.displaced_marginals import local_reduced_states,displacement_vacuum_effect,vacuum_expectation,two_mode_threshold
from gbslab.gaussian import haar_unitary
from gbslab.mps_probe import dense_to_mps
from gbslab.exact import ThresholdModel
from test_displaced_vacuum import gaussian_moment_vacuum,quadrature_vacuum,dense_coherent_log
from test_banded_gaussian_mps import reference_box
from iteration23_support import save,digest,load_mps,mps_fingerprint
from iteration22_global import load_family
from iteration22_external import ARCHIVE_SHA


def marginal_panel(evaluator,proposal,starts):
    singles,pairs=local_reduced_states(evaluator,starts); m=evaluator.modes; L=proposal.noise_factor
    model=ThresholdModel(proposal.covariance+L@L.T); q=np.zeros(m); reference=np.zeros(m); single_effects=[]; pair_effects=[]; distributions=[]; exact=[]
    for i,rho in enumerate(singles):
        effect=displacement_vacuum_effect(L[[i,i+m]],(len(rho),)); single_effects.append(effect)
        q[i]=vacuum_expectation(rho,effect); reference[i]=model.vacuum_probability((i,))
    for i in starts:
        dims=(evaluator.tensors[i].shape[1],evaluator.tensors[i+1].shape[1]); effect=displacement_vacuum_effect(L[[i,i+1,i+m,i+1+m]],dims); pair_effects.append(effect)
        qpair=vacuum_expectation(pairs[i],effect); target=model.vacuum_probability((i,i+1))
        distributions.append(two_mode_threshold(q[i],q[i+1],qpair)); exact.append(two_mode_threshold(reference[i],reference[i+1],target))
    distributions=np.asarray(distributions); exact=np.asarray(exact); tv=.5*np.sum(abs(distributions-exact),axis=1); one=abs(q-reference)
    arrays=dict(single_densities=np.stack(singles),single_effects=np.stack(single_effects),single_vacuum=q,gaussian_single_vacuum=reference,pair_starts=np.array(starts),pair_densities=np.stack([pairs[i] for i in starts]),pair_effects=np.stack(pair_effects),pair_threshold=distributions,gaussian_pair_threshold=exact,single_tv=one,pair_tv=tv)
    row=dict(single_mode_tv_max=float(one.max()),single_mode_witness=int(np.argmax(one)),pair_tv_max=float(tv.max()),pair_witness=[int(starts[int(np.argmax(tv))]),int(starts[int(np.argmax(tv))])+1],full_threshold_tv_lower_diagnostic=float(max(one.max(),tv.max())),maximum_density_trace_residual=float(max(abs(np.trace(r)-1) for r in [*singles,*pairs.values()])),scope='Fixed normalized finite MPS and fixed Gaussian reference through the SAME Gaussian displacement channel. Deterministic local threshold-TV lower diagnostic in floating point; not an experimental identity or interval certificate.')
    return arrays,row


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--archive',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    sources=['experiments/iteration29_run.py','experiments/iteration23_support.py','experiments/iteration22_global.py','experiments/iteration22_external.py','tests/test_displaced_vacuum.py','tests/test_banded_gaussian_mps.py',*[f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py')]]
    hashes={name:digest(ROOT/name) for name in sources}
    for name in sources:
        dest=out/'source_snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    shutil.copy2(ROOT/'research/iteration29_panel_plan.md',out/'frozen_plan.md')
    target='results/iteration28_panel/case_14.npz'; dependency={target:digest(ROOT/target)}
    assert dependency[target]==json.loads((ROOT/'results/iteration28_manifest.json').read_text())['artifacts'][target]
    old25=json.loads((ROOT/'results/iteration25_manifest.json').read_text())['artifacts']; paths={}
    for rank,sequence in zip((32,64,128,200),(16,17,18,19)):
        name=f'results/iteration25_panel_retry/run_{sequence}.state.npz'; dependency[name]=digest(ROOT/name); assert dependency[name]==old25[name]; paths[f'new_{rank}']=name
    assert digest(args.archive)==ARCHIVE_SHA
    large_data=np.load(ROOT/target,allow_pickle=False); large_proposal=GaussianVacuumProposal(large_data['full_pairing'],large_data['full_noise_factor'])
    standard=np.random.default_rng(291301).normal(size=(4096,large_proposal.noise_dimensions)); latent,beta=large_proposal.from_standard(standard)
    np.savez_compressed(out/'large_proposal.npz',standard=standard,latent=latent,beta=beta,reference_log_probability=large_proposal.gaussian_log_probability(beta))
    jobs=[dict(panel='small',modes=m,dimension=d,seed=s,state_kind=kind,samples=8192,batch=256) for m in (2,3) for d in (3,4) for s in (291201,291202) for kind in ('random','gaussian')]
    jobs += [dict(panel='large',modes=144,dimension=4,family=family,samples=4096,batch=128) for family in ('new_32','new_64','new_128','new_200','author_pco','author_hafnian')]
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependency,archive_sha256=ARCHIVE_SHA,large_reference_mixed_vacuum=large_proposal.mixed_vacuum,large_pairs=list(range(0,143,6)),common_random_seed=291301))
    rows=[]
    for i,job in enumerate(jobs):
        start=perf_counter(); row=dict(sequence=i,**job); print('starting',i,job,flush=True)
        try:
            if job['panel']=='small':
                m,d=job['modes'],job['dimension']; rng=np.random.default_rng(job['seed']); U=haar_unitary(m,rng); B=-(U*np.tanh(np.linspace(.05,.45,m)))@U.T
                L=.45*rng.normal(size=(2*m,2 if m==2 else 4)); proposal=GaussianVacuumProposal(B,L)
                dense=rng.normal(size=(d,)*m)+1j*rng.normal(size=(d,)*m) if job['state_kind']=='random' else reference_box(B,d); dense/=np.linalg.norm(dense)
                tensors,_=dense_to_mps(dense,64); local_standard=np.random.default_rng(291400+i).normal(size=(job['samples'],proposal.noise_dimensions)); z,local_beta=proposal.from_standard(local_standard)
                moment=gaussian_moment_vacuum(dense,L); quad=quadrature_vacuum(dense,L); assert abs(moment-quad)<2e-12
                row.update(independent_moment_vacuum=moment,independent_quadrature_vacuum=quad)
                np.savez_compressed(out/f'case_{i:02d}.reference.npz',dense=dense,pairing=B,noise_factor=L,standard=local_standard)
            else:
                proposal=large_proposal; local_beta=beta
                family=job['family']
                if family.startswith('new_'): tensors=load_mps(ROOT/paths[family])
                else:
                    tensors,entries=load_family(family,ROOT/'results/iteration22_external',args.archive); save(out/(family+'_entries.json'),entries)
                row['raw_mps_fingerprint']=mps_fingerprint(tensors)
            evaluator=CoherentMPS(tensors); del tensors; row['input_log_norm_squared']=2*evaluator.input_log_norm
            row['preparation_seconds']=perf_counter()-start; log_probabilities=[]
            for first in range(0,len(local_beta),job['batch']):
                log_probabilities.append(evaluator.log_probability(local_beta[first:first+job['batch']]))
                if job['panel']=='large' and first%1024==0: print('overlaps',i,first,flush=True)
            logp=np.concatenate(log_probabilities); weights,logw=proposal.likelihood_ratios(logp,local_beta)
            row['importance']=importance_summary(weights,proposal.mixed_vacuum); row['reference_mixed_vacuum']=proposal.mixed_vacuum
            if job['panel']=='small':
                direct=dense_coherent_log(dense,local_beta[:128]); row['direct_coherent_log_max_error']=float(np.max(abs(direct-logp[:128]))); assert row['direct_coherent_log_max_error']<2e-10
                se=row['importance']['probability_standard_error']; row['monte_carlo_standardized_error']=(row['importance']['probability_estimate']-moment)/se if se else None
            else: row['fixed_block_summaries']=[importance_summary(weights[j:j+1024],proposal.mixed_vacuum) for j in range(0,len(weights),1024)]
            starts=list(range(job['modes']-1)) if job['panel']=='small' else list(range(0,143,6))
            print('marginals',i,flush=True); marginal_arrays,marginal_row=marginal_panel(evaluator,proposal,starts); row['marginals']=marginal_row
            np.savez_compressed(out/f'case_{i:02d}.npz',log_probabilities=logp,weights=weights,log_weights=logw,**marginal_arrays)
            row['status']='completed'; del evaluator
        except Exception: row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start; rows.append(row); save(out/f'case_{i:02d}.json',row); save(out/'rows.json',rows); print('finished',i,row['status'],row.get('marginals',{}).get('full_threshold_tv_lower_diagnostic'),flush=True)
    unchanged=hashes=={name:digest(ROOT/name) for name in sources}; assert unchanged
    save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=dependency))


if __name__=='__main__': main()
