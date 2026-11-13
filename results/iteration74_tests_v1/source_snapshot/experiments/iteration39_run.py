"""Frozen official Frugal confirmation and complete ten-mode threshold panels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,sys,shutil,subprocess,time,traceback,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from gbslab.frugal import frugal_sample
from gbslab.frugal_law import full_law,distance_interval,uniform_model_transfer_bound
from gbslab.mean_witness import product_mean_interval,integer_count_tail_upper
from gbslab.gaussian_envelope import target_count_tail_upper
from gbslab.binomial_integer import lower_bound
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('panel',choices=('global','local'));panel=parser.parse_args().panel;sys.set_int_max_str_digits(0)
    out=ROOT/f'results/iteration39_{panel}';out.mkdir(exist_ok=False)
    names=['research/iteration39_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration39_run','iteration39_independent','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:
        dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    def reuse(n,it=None):
        sha=digest(ROOT/n)
        if it is not None:assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][n]
        deps[n]=sha;return ROOT/n
    V=np.load(reuse('results/iteration35_panel/inputs.npz',35))['shifted'];exe=reuse('results/iteration39_build/frugal_extended.exe');build=json.loads(reuse('results/iteration39_build/build.json').read_text());assert digest(exe)==build['executable_sha256']
    for p in exe.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix())
    if panel=='global':
        base=decode(json.loads(reuse('results/iteration35_panel/case_07.audit.json',35).read_text()));reuse('results/iteration35_panel/case_12.audit.json',35);mom=decode(json.loads(reuse('results/iteration36_recovery/case_11.audit.json',36).read_text()))['moments'];reuse('results/iteration36_validation/validation.json',36);transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper'];mu=mom['mean_interval'];second=mom['second_moment_interval'];target_X=tuple(x/144 for x in mu);target_Y=((second[0]-134*mu[1]+4489)/5929,(second[1]-134*mu[0]+4489)/5929);assert 0<=target_Y[0]<=target_Y[1]<=1
        targets={k:min(target_count_tail_upper(k,144,mu,mom['variance_interval']),integer_count_tail_upper(k,144,mu,second)['upper']) for k in (40,50,60,70,80,90,100)};save(out/'target_bounds.json',encode(dict(moments=mom,X=target_X,Y=target_Y,tail_bounds=targets,transfer_upper=transfer)));jobs=[dict(modes=144,cutoff=k,samples=1024,seed=391440) for k in range(1,5)]
    else:
        oldexe=reuse('results/iteration38_frugal_runtime/frugal_adapter.exe',38)
        for p in oldexe.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix(),38)
        targets={s:decode(json.loads(reuse(f'results/iteration37_local/target_{s:03d}.json',37).read_text()))['law']['intervals'] for s in (0,54,134)};jobs=[dict(modes=10,start=s,cutoff=k,samples=257,seed=391001+i*5+k-1) for i,s in enumerate((0,54,134)) for k in range(1,6)]
    save(out/'plan.json',encode(dict(panel=panel,jobs=jobs,source_hashes=hashes,dependencies=deps,mean_alpha=F(1,2000),event_alpha=F(1,14000),familywise_alpha=F(1,100),timeout_seconds=7200,bits=40)))
    rows=[]
    for i,job in enumerate(jobs):
        print('starting',panel,i,job,flush=True);begin=time.perf_counter();row=dict(sequence=i,**job)
        try:
            m=job['modes'];start=job.get('start',0);ix=list(range(start,start+m))+list(range(144+start,144+start+m));local=V[np.ix_(ix,ix)];csv=out/f'case_{i:02d}.csv';np.savetxt(csv,local,delimiter=',',fmt='%.17g');np.savez_compressed(out/f'case_{i:02d}.npz',covariance=local)
            cmd=[str(exe),'--input',str(csv),'--modes',str(m),'--cutoff',str(job['cutoff']),'--samples',str(job['samples']),'--seed',str(job['seed']),'--threads','1','--audit']
            if panel=='local':cmd.append('--prefix-tree')
            row['command']=cmd
            with (out/f'case_{i:02d}.stdout.json').open('w') as stdout,(out/f'case_{i:02d}.stderr.log').open('w') as stderr:p=subprocess.run(cmd,stdout=stdout,stderr=stderr,timeout=7200)
            row['returncode']=p.returncode;assert p.returncode==0;rec=json.loads((out/f'case_{i:02d}.stdout.json').read_text());assert rec['samples']==job['samples'];steps=replay(rec);counts=[s.count('1') for s in rec['outputs']];result=dict(job=job,replayed_steps=steps)
            if panel=='global':
                intervals={name:product_mean_interval(values,F(1,2000)) for name,values in (('X',[F(n,144) for n in counts]),('Y',[F((n-67)**2,5929) for n in counts]))};diff={name:max(F(0),intervals[name]['lower']-target[1],target[0]-intervals[name]['upper']) for name,target in (('X',target_X),('Y',target_Y))};events=[]
                for k,upper in targets.items():
                    successes=sum(n>=k for n in counts);lower=lower_bound(len(counts),successes,F(1,14000));events.append(dict(threshold=k,successes=successes,sampler_probability_lower=lower,target_probability_upper=upper,TV_lower=max(F(0),lower-upper)))
                full=max(*diff.values(),*(r['TV_lower'] for r in events));result.update(intervals=intervals,observable_TV_lower=diff,events=events,full_TV_statistical_lower=full,shifted_target_TV_statistical_lower=max(F(0),full-transfer));row.update(mean_clicks=sum(counts)/len(counts),mean_centered_square=sum((n-67)**2 for n in counts)/len(counts),full_TV_statistical_lower=float(full),observable_TV_lower={k:float(v) for k,v in diff.items()})
            else:
                reverse_cmd=cmd.copy();reverse_cmd[reverse_cmd.index('--samples')+1]='0';reverse_cmd.remove('--audit');reverse_cmd.append('--reverse-prefixes')
                with (out/f'case_{i:02d}.reverse.json').open('w') as stdout,(out/f'case_{i:02d}.reverse.stderr.log').open('w') as stderr:reverse=subprocess.run(reverse_cmd,stdout=stdout,stderr=stderr,timeout=7200)
                assert reverse.returncode==0;other=json.loads((out/f'case_{i:02d}.reverse.json').read_text());a={r['prefix']:r for r in rec['prefix_tree']};b={r['prefix']:r for r in other['prefix_tree']};assert a==b
                old=frugal_sample(local,job['cutoff'],job['samples'],seed=job['seed'],executable=oldexe,threads=1);save(out/f'case_{i:02d}.legacy.json',encode(old));hist=[0]*1024
                for s in rec['outputs']:hist[sum((c=='1')<<j for j,c in enumerate(s))]+=1
                assert hist==list(old['histogram_counts']);laws={model:full_law(10,rec['prefix_tree'],model=model) for model in ('continuous','uint64')};tv={model:distance_interval(law['distribution'],targets[start]) for model,law in laws.items()};coupling=sum(abs(a-b) for a,b in zip(laws['continuous']['distribution'],laws['uint64']['distribution']))/2;assert coupling<=uniform_model_transfer_bound(10);legacy=tuple(F(float(x)) for x in old['distribution']);result.update(laws=laws,TV_intervals=tv,uniform_model_distance=coupling,uniform_model_transfer_bound=uniform_model_transfer_bound(10),legacy_half_L1=sum(abs(a-b) for a,b in zip(legacy,laws['continuous']['distribution']))/2,legacy_normalization_defect=sum(legacy)-1,reverse_prefixes_equal=True,legacy_histogram_equal=True);row.update(TV_uint64=float(tv['uint64'][0]),uniform_model_distance=float(coupling))
            save(out/f'case_{i:02d}.audit.json',encode(result));row.update(status='completed',timings=rec['timings_seconds'],replayed_steps=steps,nonfinite=rec['audit_nonfinite_steps'],saturated=rec['audit_saturated_steps'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-begin;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',panel,i,row,flush=True)
    assert hashes=={n:digest(ROOT/n) for n in names} and all(digest(ROOT/n)==sha for n,sha in deps.items());save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
