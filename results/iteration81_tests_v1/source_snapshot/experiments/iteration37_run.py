"""Predeclared complete stopped laws and independent full-length confirmation."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,traceback,itertools
from fractions import Fraction as F
from time import perf_counter
import numpy as np
from iteration37_support import ROOT,trajectories,trace_details
from iteration22_external import digest,save
from iteration33_run import encode
from iteration35_run import certificate
from iteration36_run import envelope_records,compact
from verify_iteration34 import decode
from gbslab.certified_gaussian import ExactPairingModel
from gbslab.gaussian_envelope import GaussianEnvelope,binomial_lower,target_count_tail_upper
from gbslab.atom_certificate import selected_atoms
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel


def main():
    sys.set_int_max_str_digits(0);panel=sys.argv[1];assert panel in ('local','full');out=ROOT/f'results/iteration37_{panel}';out.mkdir(exist_ok=False)
    names=['research/iteration37_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration37_run','iteration37_support','iteration22_external','iteration33_run','iteration35_run','iteration36_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    shutil.copy2(ROOT/'research/iteration37_panel_plan.md',out/'frozen_plan.md');deps={}
    def reuse(p,it):
        sha=digest(ROOT/p);assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][p];deps[p]=sha;return ROOT/p
    inputs=np.load(reuse('results/iteration35_panel/inputs.npz',35));V=inputs['shifted'];base=decode(json.loads(reuse('results/iteration35_panel/case_07.audit.json',35).read_text()));reuse('results/iteration35_panel/case_12.audit.json',35);mom=decode(json.loads(reuse('results/iteration36_recovery/case_11.audit.json',36).read_text()))['moments'];reuse('results/iteration36_validation/validation.json',36);lib=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31)
    selectors=('recent','pairwise','conditional_mi','click_influence');jobs=([dict(start=s,modes=10,budget=4,selection=c,forest=f) for s in (0,54,134) for c in selectors for f in ('chain','static_mi_tree')] if panel=='local' else [dict(modes=144,budget=4,selection=c,forest='static_mi_tree',attempts=64,seed=372440) for c in selectors]);thresholds=(40,50,60,70,80,90,100);alpha=F(1,3200)
    save(out/'plan.json',encode(dict(panel=panel,jobs=jobs,source_hashes=hashes,dependencies=deps,event_thresholds=thresholds,per_event_one_sided_alpha=alpha,simultaneous_events=32,familywise_alpha=F(1,100),confidence_bits=40,cache_size=0,precision='auto',no_refills=True)))
    U=np.random.default_rng(372440).random((64,144));np.savez_compressed(out/'uniforms.npz',uniforms=U);laws={};rows=[]
    if panel=='local':
        model=ExactPairingModel(inputs['B'],inputs['F']);assert encode(certificate(model.core))==encode(base['core']) and model.operator_radius==base['operator_radius'];env=GaussianEnvelope(model)
        for start in (0,54,134):
            begin=perf_counter();law=env.law(tuple(range(start,start+10)));laws[start]=law;records=[r for r in envelope_records(env) if all(start<=j<start+10 for j in r['modes'])];assert len(records)==1023;save(out/f'target_{start:03d}.json',encode(dict(law=law,enclosures=records,seconds=perf_counter()-begin)));print('target ready',start,flush=True)
    for i,job in enumerate(jobs):
        begin=perf_counter();row=dict(sequence=i,**job);print('starting',panel,i,job,flush=True)
        try:
            if panel=='local':
                start=job['start'];order=tuple(range(start,start+10))+tuple(j for j in range(144) if not start<=j<start+10);prefixes=[h for depth in range(10) for h in itertools.product((0,1),repeat=depth)];first=[];reverse=[]
                with WireForestSampler(ThresholdModel(V),4,selection=job['selection'],forest=job['forest'],order=order,cache_size=0,precision='auto',library=lib) as sampler:
                    for h in prefixes:first.append(trace_details(sampler,h)[1])
                    save(out/f'case_{i:02d}.first.json',first)
                    for h in reversed(prefixes):reverse.append(trace_details(sampler,h)[1])
                    native=sampler.kernel.cache_info();diagnostics=sampler.cache_info()
                save(out/f'case_{i:02d}.reverse.json',reverse);keys=('prefix','q_hex','retained','edges');mismatches=[a['prefix'] for a,b in zip(first,reversed(reverse)) if any(a.get(k)!=b.get(k) for k in keys)];table={tuple(map(int,r['prefix'])):r['q_hex'] for r in first};chosen=[tuple(j for j in range(10) if mask>>j&1) for mask in range(1024)];target=laws[start]['intervals']
                def oracle(h):return None if table[h] is None else F(float.fromhex(table[h]))
                audits=[compact(selected_atoms(10,chosen,oracle,lambda c:target[sum(1<<j for j in c)],uniform_bits=bits)) for bits in (None,53)];rec=dict(job=job,order=order,audits=audits,reverse_mismatches=mismatches,native_diagnostics=native,sampler_diagnostics=diagnostics);row.update(reverse_mismatches=len(mismatches),conditional_attempts=2046,native_calls=native['probability_calls'],tv_lower=float(audits[1]['tv_lower']),tv_upper=float(audits[1]['tv_upper']),failed_prefixes=sum(r['q_hex'] is None for r in first));assert not mismatches
            else:
                outputs,rec=trajectories(V,job,U,out/f'attempts/case_{i:02d}',lib);np.savez_compressed(out/f'case_{i:02d}.npz',outputs=outputs);events=[]
                for threshold in (*thresholds,None):
                    k=sum(a['outcome']=='numerical_failure' if threshold is None else a['outcome']=='completed' and a['clicks']>=threshold for a in rec['attempts']);qlo=binomial_lower(64,k,alpha);upper=F(0) if threshold is None else target_count_tail_upper(threshold,144,mom['mean_interval'],mom['variance_interval']);events.append(dict(threshold=threshold,successes=k,attempts=64,sampler_probability_lower=qlo,target_probability_upper=upper,full_extended_TV_statistical_lower=max(F(0),qlo-upper)))
                rec.update(job=job,events=events);row.update(completed_attempts=sum(a['outcome']=='completed' for a in rec['attempts']),failed_attempts=sum(a['outcome']=='numerical_failure' for a in rec['attempts']),native_calls=rec['native_diagnostics']['probability_calls'],conditional_attempts=rec['conditional_attempts'],mean_completed_clicks=float(np.mean([a['clicks'] for a in rec['attempts'] if a['outcome']=='completed'])),full_tv_statistical_lower=float(max(e['full_extended_TV_statistical_lower'] for e in events)))
            save(out/f'case_{i:02d}.audit.json',encode(rec));row['status']='completed'
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-begin;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',panel,i,row,flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=deps))
if __name__=='__main__':main()
