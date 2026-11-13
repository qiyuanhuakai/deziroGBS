"""Frozen global mean and count-tail confirmation; sampler unchanged from round37."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,traceback
from fractions import Fraction as F
from time import perf_counter
import numpy as np
from iteration37_support import ROOT,trajectories
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.mean_witness import product_mean_interval,integer_count_tail_upper
from gbslab.gaussian_envelope import binomial_lower,target_count_tail_upper


def main():
    sys.set_int_max_str_digits(0);out=ROOT/'results/iteration38_confirmation';out.mkdir(exist_ok=False);names=['research/iteration38_confirmation_plan.md',*[f'experiments/{n}.py' for n in ('iteration38_confirm','iteration37_support','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={p:digest(ROOT/p) for p in names}
    for p in names:
        dest=out/'source_snapshot'/p;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/p,dest)
    shutil.copy2(ROOT/'research/iteration38_confirmation_plan.md',out/'frozen_plan.md');deps={}
    def reuse(p,it):
        sha=digest(ROOT/p);assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][p];deps[p]=sha;return ROOT/p
    inputs=np.load(reuse('results/iteration35_panel/inputs.npz',35));base=decode(json.loads(reuse('results/iteration35_panel/case_07.audit.json',35).read_text()));mom=decode(json.loads(reuse('results/iteration36_recovery/case_11.audit.json',36).read_text()))['moments'];reuse('results/iteration36_validation/validation.json',36);reuse('results/iteration35_panel/case_12.audit.json',35);lib=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31);transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper'];target_mean=tuple(x/144 for x in mom['mean_interval']);mean_alpha=F(1,1000);tail_alpha=F(1,16000);thresholds=(40,50,60,70,80,90,100);targets={}
    for k in thresholds:
        discrete=integer_count_tail_upper(k,144,mom['mean_interval'],mom['second_moment_interval']);old=target_count_tail_upper(k,144,mom['mean_interval'],mom['variance_interval']);targets[k]=dict(discrete=discrete,previous=old,upper=min(old,discrete['upper']))
    training=[]
    for it,folder,indices in ((36,'iteration36_recovery',range(12,16)),(37,'iteration37_full',range(4))):
        for i in indices:
            p=f'results/{folder}/case_{i:02d}.audit.json';r=decode(json.loads(reuse(p,it).read_text()));values=[F(a['clicks'],144) if a['outcome']=='completed' else F(0) for a in r['attempts']];interval=product_mean_interval(values,mean_alpha);training.append(dict(source=p,exploratory_only=True,interval=interval,mean_difference=max(F(0),interval['lower']-target_mean[1],target_mean[0]-interval['upper'])))
    save(out/'training.json',encode(training));jobs=[dict(selection=s,forest='static_mi_tree',budget=4,modes=144,attempts=96,seed=381440) for s in ('recent','pairwise','conditional_mi','click_influence')];save(out/'plan.json',encode(dict(jobs=jobs,source_hashes=hashes,dependencies=deps,mean_alpha_per_tail=mean_alpha,mean_tails=8,event_alpha=tail_alpha,event_tails=32,familywise_alpha=8*mean_alpha+32*tail_alpha,thresholds=thresholds,bits=40,failed_X=0)));save(out/'target_bounds.json',encode(dict(moments=mom,target_X_interval=target_mean,transfer_upper=transfer,tail_bounds=targets)));U=np.random.default_rng(381440).random((96,144));np.savez_compressed(out/'uniforms.npz',uniforms=U);rows=[]
    for i,job in enumerate(jobs):
        begin=perf_counter();row=dict(sequence=i,**job);print('starting',i,job,flush=True)
        try:
            outputs,rec=trajectories(inputs['shifted'],job,U,out/f'attempts/case_{i:02d}',lib);values=[F(a['clicks'],144) if a['outcome']=='completed' else F(0) for a in rec['attempts']];interval=product_mean_interval(values,mean_alpha);mean_lower=max(F(0),interval['lower']-target_mean[1],target_mean[0]-interval['upper']);events=[]
            for threshold in (*thresholds,None):
                k=sum(a['outcome']=='numerical_failure' if threshold is None else a['outcome']=='completed' and a['clicks']>=threshold for a in rec['attempts']);lower=binomial_lower(96,k,tail_alpha);upper=F(0) if threshold is None else targets[threshold]['upper'];events.append(dict(threshold=threshold,successes=k,attempts=96,sampler_probability_lower=lower,target_probability_upper=upper,full_TV_statistical_lower=max(F(0),lower-upper)))
            full_lower=max(mean_lower,max(e['full_TV_statistical_lower'] for e in events));rec.update(job=job,mean_interval=interval,mean_TV_statistical_lower=mean_lower,events=events,full_TV_statistical_lower=full_lower,shifted_target_TV_statistical_lower=max(F(0),full_lower-transfer));np.savez_compressed(out/f'case_{i:02d}.npz',outputs=outputs);save(out/f'case_{i:02d}.audit.json',encode(rec));row.update(status='completed',completed_attempts=sum(a['outcome']=='completed' for a in rec['attempts']),failed_attempts=sum(a['outcome']=='numerical_failure' for a in rec['attempts']),sample_mean_mapped_clicks=float(sum(values)*144/96),native_calls=rec['native_diagnostics']['probability_calls'],conditional_attempts=rec['conditional_attempts'],mean_TV_statistical_lower=float(mean_lower),full_TV_statistical_lower=float(full_lower))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-begin;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row,flush=True)
    unchanged=hashes=={p:digest(ROOT/p) for p in names};assert unchanged;save(out/'summary.json',dict(planned=4,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=deps))
if __name__=='__main__':main()
