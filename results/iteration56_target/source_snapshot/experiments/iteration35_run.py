"""Frozen exact physical targets, stopped prefix laws, and fixed full attempts."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,shutil,sys,traceback
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.set_int_max_str_digits(0)
from gbslab.certified_gaussian import HermitianCertificate,ExactPairingModel,certify_uncertainty
from gbslab.rational_gaussian import reciprocal_sqrt
from gbslab.atom_certificate import selected_atoms
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel,NumericalProbabilityError
from gbslab.positive_completion import PositiveCompletionSamplingError
from iteration22_external import save,digest
from iteration33_run import encode


def certificate(c,determinant=True):
    r=dict(real=c.real.tolist(),imag=c.imag.tolist(),denominator=c.denominator,R_real=c.R_real.tolist(),R_imag=c.R_imag.tolist(),scale=c.scale,diagnostics=c.diagnostics())
    if determinant:r['determinant_interval']=c.determinant_interval()
    return r


def small_input(m):
    rng=np.random.default_rng(350000+m);a=rng.integers(-2,3,(m,m));b=rng.integers(-2,3,(m,m));B=((a+a.T)+1j*(b+b.T))/(1<<(m.bit_length()+6));noise=rng.integers(-2,3,(2*m,m+1))/32.;return B,noise


def target_table(V):
    m=len(V)//2;vacua=[(F(1),F(1))];certs=[]
    for mask in range(1,1<<m):
        modes=[j for j in range(m) if mask>>j&1];ix=modes+[j+m for j in modes];A=V[np.ix_(ix,ix)];Q=np.array([[(F(float(A[i,j]))+int(i==j))/2 for j in range(len(ix))] for i in range(len(ix))],dtype=object);c=HermitianCertificate(Q);vacua.append(reciprocal_sqrt(c.determinant_interval(),256));certs.append(dict(mask=mask,**certificate(c)))
    raw=[];bounds=[]
    for clicks in range(1<<m):
        zeros=((1<<m)-1)^clicks;lo=hi=F(0);subset=clicks
        while True:
            a,b=vacua[zeros|subset]
            if subset.bit_count()%2:lo-=b;hi-=a
            else:lo+=a;hi+=b
            if not subset:break
            subset=(subset-1)&clicks
        raw.append((lo,hi));bounds.append((max(F(0),lo),min(F(1),hi)))
        assert bounds[-1][0]<=bounds[-1][1]
    return dict(vacua=vacua,raw_atom_intervals=raw,target_intervals=bounds,certificates=certs)


def main():
    out=ROOT/'results/iteration35_panel';out.mkdir(exist_ok=False);names=['experiments/iteration35_run.py','experiments/iteration22_external.py','experiments/iteration33_run.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(ROOT/'research/iteration35_panel_plan.md',out/'frozen_plan.md');dependencies={}
    def reuse(p,iteration):
        sha=digest(ROOT/p);assert sha==json.loads((ROOT/f'results/iteration{iteration}_manifest.json').read_text())['artifacts'][p];dependencies[p]=sha;return ROOT/p
    data=np.load(reuse('results/iteration28_panel/case_14.npz',28));old=np.load(reuse('results/iteration31_panel/large_input.npz',31))['covariance'];library=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31)
    jobs=[dict(panel='model',modes=m,bits=bits) for m in (2,4,8,144) for bits in (64,128)]
    jobs += [dict(panel='uncertainty',modes=144,bits=128,shift_exponent=e,family='original') for e in (None,-52,-50,-46,-40)]
    jobs += [dict(panel='uncertainty',modes=8,bits=bits,family=family) for family in ('vacuum','rank_one','invalid_half','thermal40') for bits in (64,128)]
    jobs += [dict(panel='prefix',modes=144,prefix_modes=8,start=start,budget=budget,forest=forest) for start in (0,18,54,136) for budget in (0,2,4) for forest in ('chain','static_mi_tree')]
    jobs += [dict(panel='sampling',modes=144,attempts=8,budget=budget,forest=forest,seed=351440) for budget in (0,4) for forest in ('chain','static_mi_tree')]
    assert len(jobs)==49;save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,primary_requested_shift=2.**-40,cache_size=0,precision='auto',exact_integer_string_digit_limit=0,auxiliary_target_tables=4));rows=[];primary=old+2.**-40*np.eye(len(old));np.savez_compressed(out/'inputs.npz',original=old,shifted=primary,B=data['full_pairing'],F=data['full_noise_factor']);targets={};primary_valid=False
    for i,job in enumerate(jobs):
        start=perf_counter();row=dict(sequence=i,**job);print('starting',i,job,flush=True)
        try:
            arrays={};record=dict(job=job)
            if job['panel']=='model':
                B,noise=(data['full_pairing'],data['full_noise_factor']) if job['modes']==144 else small_input(job['modes']);model=ExactPairingModel(B,noise,bits=job['bits']);arrays.update(B=B,F=noise,center_float=model.covariance_center_float);record.update(core=certificate(model.core),noise=certificate(model.noise_certificate),vacuum_interval=model.vacuum_interval,center_numerators=model.center_numerators.tolist(),center_denominator=model.center_denominator,operator_radius=model.operator_radius,core_trace_interval=(model.core_trace_lower,model.core_trace_upper))
                if job['modes']==144:record.update(original_comparison=model.compare_covariance(old),transfers={str(e):model.noise_transfer_bound(old,old+2.**e*np.eye(len(old))) for e in (-52,-50,-46,-40)})
                row.update(vacuum_lower=float(model.vacuum_interval[0]),vacuum_upper=float(model.vacuum_interval[1]),vacuum_width=float(model.vacuum_interval[1]-model.vacuum_interval[0]),core_residual=float(model.core.delta),center_radius=float(model.operator_radius))
            elif job['panel']=='uncertainty':
                m=job['modes'];family=job['family']
                if family=='original':V=old if job['shift_exponent'] is None else old+2.**job['shift_exponent']*np.eye(2*m)
                elif family=='vacuum':V=np.eye(2*m)
                elif family=='invalid_half':V=.5*np.eye(2*m)
                elif family=='thermal40':V=(1+2.**-40)*np.eye(2*m)
                else:
                    a=np.array([(-1 if j%3==1 else 1)/32 for j in range(2*m)]);V=np.eye(2*m)+np.outer(a,a)
                result=certify_uncertainty(V,bits=job['bits']);row['physicality_status']=result['status'];arrays['covariance']=V
                if 'certificate' in result:result['certificate']=certificate(result['certificate'],False)
                if 'witness' in result:arrays['witness']=result.pop('witness')
                record.update(result)
                if family=='original' and job['shift_exponent']==-40:primary_valid=result['status']=='strictly_physical'
            elif job['panel']=='prefix':
                assert primary_valid;first=job['start'];modes=tuple(range(first,first+8));ix=list(modes)+[j+144 for j in modes];V=primary[np.ix_(ix,ix)];order=modes+tuple(j for j in range(144) if j not in modes)
                if first not in targets:
                    begin=perf_counter();targets[first]=target_table(V);targets[first]['seconds']=perf_counter()-begin;save(out/f'target_{first:03d}.json',encode(targets[first]));np.savez_compressed(out/f'target_{first:03d}.npz',covariance=V,order=np.array(order));print('target intervals',first,targets[first]['seconds'],flush=True)
                target=targets[first]['target_intervals'];table={};calls=duplicates=0
                with WireForestSampler(ThresholdModel(primary),job['budget'],forest=job['forest'],selection='recent',order=order,precision='auto',cache_size=0,library=library) as sampler:
                    for mask in range(256):
                        prefix=()
                        for j in range(8):
                            calls+=1
                            try:q=float(sampler.conditional_click_probability(prefix));r=dict(prefix=list(prefix),q_hex=q.hex(),observations=1)
                            except NumericalProbabilityError as error:q=None;r=dict(prefix=list(prefix),q_hex=None,error=str(error),observations=1)
                            if prefix in table:
                                duplicates+=1;assert table[prefix]['q_hex']==r['q_hex'];table[prefix]['observations']+=1
                            else:table[prefix]=r
                            if q is None:break
                            bit=(mask>>j)&1
                            if (q if bit else 1-q)==0:break
                            prefix+=(bit,)
                    info=sampler.kernel.cache_info();assert info['probability_calls']==calls
                chosen=[tuple(j for j in range(8) if mask>>j&1) for mask in range(256)]
                def oracle(h):return None if table[h]['q_hex'] is None else F(float.fromhex(table[h]['q_hex']))
                audits=[]
                for bits in (None,53):
                    a=selected_atoms(8,chosen,oracle,lambda c:target[sum(1<<j for j in c)],uniform_bits=bits);audits.append({k:v for k,v in a.items() if k not in ('records','patterns','target_intervals')})
                record.update(oracle=list(table.values()),audits=audits,native_diagnostics=info);arrays.update(covariance=V,full_order=np.array(order));row.update(tv_lower=float(audits[-1]['tv_lower']),tv_upper=float(audits[-1]['tv_upper']),failure_mass=float(audits[-1]['seen_failure_mass']),native_calls=calls,unique_prefixes=len(table),duplicate_checks=duplicates)
            else:
                assert primary_valid;uniforms=np.random.default_rng(job['seed']).random((8,144));attempts=[];outputs=np.full((8,144),-1,dtype=np.int16);qrecords=[]
                class Uniforms:
                    def __init__(self,x):self.x=x;self.position=0
                    def random(self):
                        value=float(self.x[self.position]);self.position+=1;return value
                with WireForestSampler(ThresholdModel(primary),job['budget'],forest=job['forest'],selection='recent',precision='auto',cache_size=0,library=library) as sampler:
                    original=sampler.conditional_click_probability
                    def observed(prefix):
                        try:q=float(original(prefix));qrecords.append(dict(attempt=len(attempts),prefix=list(prefix),q_hex=q.hex()));return q
                        except NumericalProbabilityError as error:qrecords.append(dict(attempt=len(attempts),prefix=list(prefix),q_hex=None,error=str(error)));raise
                    sampler.conditional_click_probability=observed
                    for k in range(8):
                        rng=Uniforms(uniforms[k]);r=dict(attempt=k)
                        try:outputs[k]=sampler.sample(1,rng)[0];r['outcome']='completed'
                        except PositiveCompletionSamplingError as error:r.update(outcome='numerical_failure',failed_prefix=list(error.failed_prefix),error=str(error))
                        r['uniforms_consumed']=rng.position;attempts.append(r)
                    info=sampler.kernel.cache_info();assert info['probability_calls']==len(qrecords)
                arrays.update(uniforms=uniforms,outputs=outputs);record.update(attempts=attempts,conditionals=qrecords,native_diagnostics=info);row.update(completed_attempts=sum(r['outcome']=='completed' for r in attempts),failed_attempts=sum(r['outcome']!='completed' for r in attempts),native_calls=len(qrecords))
            np.savez_compressed(out/f'case_{i:02d}.npz',**arrays);save(out/f'case_{i:02d}.audit.json',encode(record));row['status']='completed'
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row['status'],row.get('physicality_status'),row.get('tv_upper'),flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,primary_valid=primary_valid,dependencies=dependencies))
if __name__=='__main__':main()
