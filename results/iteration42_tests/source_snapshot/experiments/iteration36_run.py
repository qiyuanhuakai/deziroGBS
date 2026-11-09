"""Direct exact-model marginals and predeclared full-output event tests."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,traceback
from pathlib import Path
from fractions import Fraction as F
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.set_int_max_str_digits(0)
from gbslab.certified_gaussian import ExactPairingModel
from gbslab.gaussian_envelope import GaussianEnvelope,binomial_lower,target_count_tail_upper
from gbslab.atom_certificate import selected_atoms
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel,NumericalProbabilityError
from gbslab.positive_completion import PositiveCompletionSamplingError
from iteration22_external import digest,save
from iteration33_run import encode
from iteration35_run import certificate,small_input
from verify_iteration34 import decode


def envelope_records(e):
    return [dict(modes=modes,vacuum_interval=e.vacua[modes],**{k:v for k,v in r.items() if k!='certificate'},certificate=certificate(r['certificate'])) for modes,r in sorted(e.certificates.items(),key=lambda x:(len(x[0]),x[0]))]


def compact(a):return {k:v for k,v in a.items() if k not in ('records','patterns','target_intervals')}


def main():
    out=ROOT/'results/iteration36_panel';out.mkdir(exist_ok=False);names=['experiments/iteration36_run.py','experiments/iteration35_run.py','experiments/iteration22_external.py','experiments/iteration33_run.py','experiments/verify_iteration34.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    shutil.copy2(ROOT/'research/iteration36_panel_plan.md',out/'frozen_plan.md');dependencies={}
    def reuse(p,it=35):
        sha=digest(ROOT/p);assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][p];dependencies[p]=sha;return ROOT/p
    inputs=np.load(reuse('results/iteration35_panel/inputs.npz'));primary=inputs['shifted'];base=decode(json.loads(reuse('results/iteration35_panel/case_07.audit.json').read_text()));reuse('results/iteration35_panel/case_12.audit.json');reuse('results/iteration35_validation/validation.json');library=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31)
    for first in (0,18,54,136):reuse(f'results/iteration35_panel/target_{first:03d}.json')
    for i in range(21,45):reuse(f'results/iteration35_panel/case_{i:02d}.audit.json')
    jobs=[dict(panel='small',modes=m,bits=bits) for m in (2,4,8) for bits in (64,128)];jobs+=[dict(panel='model',modes=144,bits=128)];jobs+=[dict(panel='block',start=s,modes=8) for s in (0,18,54,136)];jobs+=[dict(panel='moments',modes=144)];jobs+=[dict(panel='sampling',modes=144,budget=b,forest=f,attempts=64,seed=361440) for b in (0,4) for f in ('chain','static_mi_tree')]
    assert len(jobs)==16;thresholds=(40,50,60,70,80,90,100);alpha=F(1,3200);uniforms=np.random.default_rng(361440).random((64,144));np.savez_compressed(out/'uniforms.npz',uniforms=uniforms);save(out/'plan.json',encode(dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,event_thresholds=thresholds,per_event_one_sided_alpha=alpha,simultaneous_events=32,familywise_alpha=F(1,100),confidence_bits=40,statistical_premise='independent identically distributed uniform-grid attempts; fixed PRNG stream is recorded, not a proof of independence',cache_size=0,precision='auto',no_refills=True)));rows=[];main_env=None;moments=None
    for i,job in enumerate(jobs):
        start=perf_counter();row=dict(sequence=i,**job);print('starting',i,job,flush=True)
        try:
            record=dict(job=job);arrays={}
            if job['panel']=='small':
                B,noise=small_input(job['modes']);model=ExactPairingModel(B,noise,bits=job['bits']);e=GaussianEnvelope(model,bits=job['bits']);law=e.law(tuple(range(job['modes'])));r=e.click_moments();record.update(law=law,moments=r,enclosures=envelope_records(e),operator_radius=model.operator_radius,center_numerators=model.center_numerators.tolist(),center_denominator=model.center_denominator,core=certificate(model.core),noise=certificate(model.noise_certificate));arrays.update(B=B,F=noise);row.update(maximum_atom_width=float(max(b-a for a,b in law['intervals'])),enclosures=len(e.certificates))
            elif job['panel']=='model':
                model=ExactPairingModel(inputs['B'],inputs['F']);assert model.operator_radius==base['operator_radius'] and model.center_denominator==base['center_denominator'] and np.array_equal(model.center_numerators,np.array(base['center_numerators'],dtype=object));assert certificate(model.core)==base['core'] and certificate(model.noise_certificate)==base['noise'];main_env=GaussianEnvelope(model);record.update(core=certificate(model.core),noise=certificate(model.noise_certificate),operator_radius=model.operator_radius,center_numerators=model.center_numerators.tolist(),center_denominator=model.center_denominator,vacuum_interval=model.vacuum_interval);arrays.update(B=inputs['B'],F=inputs['F'],shifted=primary);row['same_exact_model_as_round35']=True
            elif job['panel']=='block':
                assert main_env is not None;first=job['start'];law=main_env.law(tuple(range(first,first+8)));target=law['intervals'];old=decode(json.loads((ROOT/f'results/iteration35_panel/target_{first:03d}.json').read_text()))['target_intervals'];lower=sum(max(F(0),a-d,c-b) for (a,b),(c,d) in zip(target,old))/2;upper=sum(max(abs(a-d),abs(b-c)) for (a,b),(c,d) in zip(target,old))/2;audits=[];chosen=[tuple(j for j in range(8) if mask>>j&1) for mask in range(256)];first_case=21+6*(0,18,54,136).index(first)
                for source_case in range(first_case,first_case+6):
                    previous=decode(json.loads((ROOT/f'results/iteration35_panel/case_{source_case:02d}.audit.json').read_text()));table={tuple(r['prefix']):r['q_hex'] for r in previous['oracle']}
                    def oracle(h):return None if table[h] is None else F(float.fromhex(table[h]))
                    for bits in (None,53):
                        audit=selected_atoms(8,chosen,oracle,lambda c:target[sum(1<<j for j in c)],uniform_bits=bits);audits.append(dict(source_case=source_case,budget=previous['job']['budget'],forest=previous['job']['forest'],**compact(audit)))
                record.update(law=law,audits=audits,shifted_target_TV_interval=(lower,upper));row.update(shifted_target_tv_lower=float(lower),shifted_target_tv_upper=float(upper),maximum_atom_width=float(max(b-a for a,b in target)),best_direct_tv_upper=float(min(a['tv_upper'] for a in audits if a['uniform_bits']==53)))
            elif job['panel']=='moments':
                assert main_env is not None;moments=main_env.click_moments();bounds={str(k):target_count_tail_upper(k,144,moments['mean_interval'],moments['variance_interval']) for k in thresholds};record.update(moments=moments,tail_probability_upper=bounds);records=envelope_records(main_env)
                for offset in range(0,len(records),256):save(out/f'certificates_{offset//256:02d}.json',encode(records[offset:offset+256]))
                save(out/'certificate_index.json',dict(count=len(records),shards=(len(records)+255)//256));row.update(enclosures=len(records),mean_interval=[float(x) for x in moments['mean_interval']],variance_interval=[float(x) for x in moments['variance_interval']])
            else:
                assert main_env is not None and moments is not None;directory=out/f'attempts/case_{i:02d}';directory.mkdir(parents=True,exist_ok=False);outputs=np.full((64,144),-1,dtype=np.int16);attempts=[];calls=0
                class Sequence:
                    def __init__(self,values):self.values=values;self.position=0
                    def random(self):
                        x=float(self.values[self.position]);self.position+=1;return x
                with WireForestSampler(ThresholdModel(primary),job['budget'],forest=job['forest'],selection='recent',precision='auto',cache_size=0,library=library) as sampler:
                    original=sampler.conditional_click_probability
                    def observed(prefix):
                        try:q=float(original(prefix));conditionals.append(dict(prefix=''.join(map(str,prefix)),q_hex=q.hex()));return q
                        except NumericalProbabilityError as error:conditionals.append(dict(prefix=''.join(map(str,prefix)),q_hex=None,error=str(error)));raise
                    sampler.conditional_click_probability=observed
                    for attempt in range(64):
                        begin=perf_counter();rng=Sequence(uniforms[attempt]);conditionals=[];a=dict(attempt=attempt)
                        try:outputs[attempt]=sampler.sample(1,rng)[0];a.update(outcome='completed',clicks=int(outputs[attempt].sum()),output=outputs[attempt].tolist())
                        except PositiveCompletionSamplingError as error:a.update(outcome='numerical_failure',failed_prefix=''.join(map(str,error.failed_prefix)),error=str(error))
                        calls+=len(conditionals);a.update(uniforms_consumed=rng.position,native_calls=len(conditionals),seconds=perf_counter()-begin);save(directory/f'attempt_{attempt:03d}.json',dict(**a,conditionals=conditionals));attempts.append(a)
                        if (attempt+1)%8==0:print('attempts',i,attempt+1,'completed',sum(a['outcome']=='completed' for a in attempts),flush=True)
                    info=sampler.kernel.cache_info();assert info['probability_calls']==calls
                events=[]
                for threshold in (*thresholds,None):
                    k=sum(a['outcome']=='numerical_failure' if threshold is None else a['outcome']=='completed' and a['clicks']>=threshold for a in attempts);qlo=binomial_lower(64,k,alpha);pupper=F(0) if threshold is None else target_count_tail_upper(threshold,144,moments['mean_interval'],moments['variance_interval']);events.append(dict(event='failure' if threshold is None else f'completed_clicks_at_least_{threshold}',threshold=threshold,successes=k,attempts=64,sampler_probability_lower=qlo,target_probability_upper=pupper,full_extended_TV_statistical_lower=max(F(0),qlo-pupper)))
                record.update(attempts=attempts,events=events,native_diagnostics=info);arrays['outputs']=outputs;row.update(completed_attempts=sum(a['outcome']=='completed' for a in attempts),failed_attempts=sum(a['outcome']=='numerical_failure' for a in attempts),native_calls=calls,mean_completed_clicks=float(np.mean([a['clicks'] for a in attempts if a['outcome']=='completed'])),full_tv_statistical_lower=float(max(e['full_extended_TV_statistical_lower'] for e in events)))
            np.savez_compressed(out/f'case_{i:02d}.npz',**arrays);save(out/f'case_{i:02d}.audit.json',encode(record));row['status']='completed'
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row['status'],row.get('full_tv_statistical_lower'),flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=16,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=dependencies))
if __name__=='__main__':main()
