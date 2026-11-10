"""Five fixed configurations, common attempt counts and asymmetric mean tests."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,sys,json,shutil,subprocess,time,traceback
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration37_support import ROOT,trajectories
from iteration39_independent import replay
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.mean_witness import product_mean_interval,integer_count_tail_upper
from gbslab.betting_mean import lower_bound as betting_lower,DEFAULT_STAKES
from gbslab.binomial_integer import lower_bound as binomial_lower
from gbslab.gaussian_envelope import target_count_tail_upper


def main():
    parser=argparse.ArgumentParser();parser.add_argument('panel',choices=('frugal','recent','conditional_mi'));label=parser.parse_args().panel;sys.set_int_max_str_digits(0);out=ROOT/f'results/iteration40_{label}';out.mkdir(exist_ok=False);deps={};names=['research/iteration40_confirmation_plan.md',*[f'experiments/{n}.py' for n in ('iteration40_confirm','iteration37_support','iteration39_independent','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def reuse(n,it):
        p=ROOT/n;sha=digest(p);assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][n];deps[n]=sha;return p
    V=np.load(reuse('results/iteration35_panel/inputs.npz',35))['shifted'];base=decode(json.loads(reuse('results/iteration35_panel/case_07.audit.json',35).read_text()));reuse('results/iteration35_panel/case_12.audit.json',35);mom=decode(json.loads(reuse('results/iteration36_recovery/case_11.audit.json',36).read_text()))['moments'];reuse('results/iteration36_validation/validation.json',36);mu=mom['mean_interval'];e2=mom['second_moment_interval'];targets={'X':tuple(x/144 for x in mu),'Y':((e2[0]-144*mu[1]+5184)/5184,(e2[1]-144*mu[0]+5184)/5184)};transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper'];tails={k:min(target_count_tail_upper(k,144,mu,mom['variance_interval']),integer_count_tail_upper(k,144,mu,e2)['upper']) for k in (40,50,60,70,80,90,100)};alpha=F(1,5700)
    if label=='frugal':
        exe=reuse('results/iteration39_build/frugal_extended.exe',39);reuse('results/iteration39_build/build.json',39)
        for p in exe.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix(),39)
        jobs=[dict(method='frugal',modes=144,cutoff=k,attempts=256,seed=401440) for k in (1,2,4)]
    else:
        lib=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31);jobs=[dict(method='forest',modes=144,selection=label,forest='static_mi_tree',budget=4,attempts=256,seed=401440)];U=np.random.default_rng(401440).random((256,144));np.savez_compressed(out/'uniforms.npz',uniforms=U)
    save(out/'plan.json',encode(dict(panel=label,jobs=jobs,source_hashes=hashes,dependencies=deps,alpha_per_tail=alpha,familywise_alpha=57*alpha,total_tails=57,stakes=DEFAULT_STAKES,bits=40,source_round39_pilot_not_confirmation=True)));save(out/'target_bounds.json',encode(dict(moments=mom,targets=targets,tail_bounds=tails,transfer_upper=transfer)));rows=[]
    for i,job in enumerate(jobs):
        begin=time.perf_counter();row=dict(sequence=i,**job);print('starting',label,i,job,flush=True)
        try:
            if label=='frugal':
                csv=out/f'case_{i:02d}.csv';np.savetxt(csv,V,delimiter=',',fmt='%.17g');cmd=[str(exe),'--input',str(csv),'--modes','144','--cutoff',str(job['cutoff']),'--samples','256','--seed','401440','--threads','1','--audit'];row['command']=cmd
                with (out/f'case_{i:02d}.stdout.json').open('w') as stdout,(out/f'case_{i:02d}.stderr.log').open('w') as stderr:p=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=7200)
                assert p.returncode==0;raw=json.loads((out/f'case_{i:02d}.stdout.json').read_text());assert raw['samples']==256;steps=replay(raw);counts=[s.count('1') for s in raw['outputs']];rec=dict(job=job,replayed_steps=steps);row.update(timings=raw['timings_seconds'],nonfinite=raw['audit_nonfinite_steps'],saturated=raw['audit_saturated_steps'])
            else:
                outputs,rec=trajectories(V,job,U,out/f'attempts/case_{i:02d}',lib);np.savez_compressed(out/f'case_{i:02d}.npz',outputs=outputs);counts=[a['clicks'] if a['outcome']=='completed' else None for a in rec['attempts']];row.update(native_calls=rec['native_diagnostics']['probability_calls'])
            values={'X':[F(0) if n is None else F(n,144) for n in counts],'Y':[F(0) if n is None else F((n-72)**2,5184) for n in counts]};intervals={name:product_mean_interval(xs,alpha) for name,xs in values.items()};intervals['Y']['product_lower_diagnostic']=intervals['Y']['lower'];intervals['Y']['lower']=betting_lower(values['Y'],alpha);intervals['Y']['lower_method']='fixed_uniform_betting_mixture';intervals['Y']['stakes']=DEFAULT_STAKES;intervals['Y']['weights']=tuple(F(1,7) for _ in range(7));diff={name:max(F(0),r['lower']-targets[name][1],targets[name][0]-r['upper']) for name,r in intervals.items()};events=[]
            for k in (*tails,*(('failure',) if label!='frugal' else ())):
                count=sum(n is None for n in counts) if k=='failure' else sum(n is not None and n>=k for n in counts);lower=binomial_lower(256,count,alpha);upper=F(0) if k=='failure' else tails[k];events.append(dict(event=k,successes=count,attempts=256,probability_lower=lower,target_upper=upper,TV_lower=max(F(0),lower-upper)))
            full=max(*diff.values(),*(e['TV_lower'] for e in events));rec.update(job=job,intervals=intervals,observable_TV_lower=diff,events=events,full_TV_statistical_lower=full,shifted_target_TV_statistical_lower=max(F(0),full-transfer));save(out/f'case_{i:02d}.audit.json',encode(rec));row.update(status='completed',completed_attempts=sum(n is not None for n in counts),failed_attempts=sum(n is None for n in counts),sample_mean_mapped_clicks=float(sum(values['X'])*144/256),sample_mean_mapped_square=float(sum(values['Y'])*5184/256),full_TV_statistical_lower=float(full))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-begin;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',label,i,row,flush=True)
    assert hashes=={n:digest(ROOT/n) for n in names} and all(digest(ROOT/n)==sha for n,sha in deps.items());save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
