import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,shutil,sys,traceback
from fractions import Fraction as F
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.coupling_audit import audit,normalized_rationals
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel,NumericalProbabilityError
from iteration22_external import digest,save

def encode(value):
    if isinstance(value,F):return {'rational':str(value)}
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    return value

def compact(result):return {k:v for k,v in result.items() if k not in ('records','unresolved','target','distribution')}

def native_oracle(sampler,table):
    def call(prefix):
        assert prefix not in table
        try:
            d=sampler.conditional_details(prefix);q=float(d['click_probability']);table[prefix]=dict(prefix=prefix,q=F(q),hex=q.hex(),precision_used=d['precision_used'],refinement_attempted=d['refinement_attempted']);return F(q)
        except NumericalProbabilityError as e:table[prefix]=dict(prefix=prefix,q=None,error=str(e));return None
    return call

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['experiments/iteration33_run.py','experiments/iteration22_external.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(ROOT/'research/iteration33_panel_plan.md',out/'frozen_plan.md');dependencies={}
    def reuse(path,iteration):
        sha=digest(ROOT/path);assert sha==json.loads((ROOT/f'results/iteration{iteration}_manifest.json').read_text())['artifacts'][path];dependencies[path]=sha;return ROOT/path
    oldrows=json.loads(reuse('results/iteration32_panel/rows.json',32).read_text());library=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31)
    for i in range(24,48):reuse(f'results/iteration32_panel/case_{i:02d}.npz',32)
    jobs=[dict(panel='reused_forest',source_case=i,modes=8,start=oldrows[i]['start'],budget=oldrows[i]['budget'],forest=oldrows[i]['forest']) for i in range(24,48)]
    jobs += [dict(panel='thermal',modes=m,exponent=b,precision=precision,budget=4,forest='chain') for m in (6,8) for b in (8,40) for precision in ('longdouble','quad','auto')]
    jobs += [dict(panel='endpoint',modes=1,precision='auto',budget=0,forest='chain'),dict(panel='vacuum',modes=4,precision='auto',budget=4,forest='chain')]
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,partial_budgets=[0,8,32,128],grid_bits=53));rows=[]
    for i,job in enumerate(jobs):
        started=perf_counter();row=dict(sequence=i,**job);print('starting',i,job,flush=True)
        try:
            m=job['modes'];table={};replay_table={};raw_target=None
            if job['panel']=='reused_forest':
                data=np.load(ROOT/f"results/iteration32_panel/case_{job['source_case']:02d}.npz");raw_target=[F(float(x)) for x in data['target']];target,total=normalized_rationals(raw_target)
                for length in range(m):
                    for mask in range(1<<length):
                        prefix=tuple((mask>>j)&1 for j in range(length));q=float(data['conditionals'][(1<<length)-1+mask]);table[prefix]=dict(prefix=prefix,q=F(q),hex=q.hex())
                full=audit(target,lambda prefix:table[prefix]['q']);row['target_origin']='exact normalization of saved binary64 Gaussian table'
            else:
                v=1+2.**(-job['exponent']) if job['panel']=='thermal' else (1e17 if job['panel']=='endpoint' else 1.);click=(F(v)-1)/(F(v)+1);target=tuple(click**mask.bit_count()*(1-click)**(m-mask.bit_count()) for mask in range(1<<m));total=F(1);V=v*np.eye(2*m);np.savez_compressed(out/f'case_{i:02d}.npz',covariance=V);row.update(target_origin='analytic independent thermal law for exact supplied binary64 covariance',exact_click_probability=str(click),covariance_diagonal_hex=v.hex())
                with WireForestSampler(ThresholdModel(V),job['budget'],forest='chain',selection='recent',precision=job['precision'],library=library) as sampler:
                    full=audit(target,native_oracle(sampler,table));row['native_diagnostics']=sampler.kernel.cache_info()
                with WireForestSampler(ThresholdModel(V),job['budget'],forest='chain',selection='recent',precision=job['precision'],library=library) as sampler:
                    call=native_oracle(sampler,replay_table)
                    def visit(prefix):
                        if len(prefix)==m:return
                        q=call(prefix)
                        if q is None:return
                        if q<1:visit(prefix+(0,))
                        if q>0:visit(prefix+(1,))
                    visit(());row['replay_native_diagnostics']=sampler.kernel.cache_info()
                replay=audit(target,lambda prefix:replay_table[prefix]['q']);row['replay_extended_law_tv']=float((sum(abs(x-y) for x,y in zip(full['distribution'],replay['distribution']))+abs(full['failure_mass_seen']-replay['failure_mass_seen']))/2)
                row['replay_table_identical']=set(table)==set(replay_table) and all(table[p]['q']==replay_table[p]['q'] for p in table)
            grid=audit(target,lambda prefix:table[prefix]['q'],uniform_bits=53);partials=[]
            for bits in (None,53):
                for budget in sorted(set([0,8,32,128,full['queries']])):
                    result=audit(target,lambda prefix:table[prefix]['q'],maximum_queries=budget,uniform_bits=bits);partials.append(dict(budget=budget,**compact(result)))
            law_shift=(sum(abs(x-y) for x,y in zip(full['distribution'],grid['distribution']))+abs(full['failure_mass_seen']-grid['failure_mass_seen']))/2
            output=dict(job=job,target=target,raw_target=raw_target,original_target_mass=total,oracle=list(table.values()),replay_oracle=list(replay_table.values()),ideal=full,grid53=grid,partials=partials,grid_law_tv=law_shift)
            save(out/f'case_{i:02d}.audit.json',encode(output));row.update(status='completed',queries=full['queries'],failure_nodes=sum(r['q'] is None for r in table.values()),target_original_mass_error=float(total-1),target_normalization_l1_change=float(sum(abs(a-b) for a,b in zip(raw_target,target))) if raw_target is not None else 0.,metrics={k:float(full[k]) for k in ('extended_tv','coupling_upper','hybrid_target_upper','hybrid_proxy_upper','failure_mass_seen')},grid_law_tv=float(law_shift),partials=[dict(budget=p['budget'],uniform_bits=p['uniform_bits'],queries=p['queries'],coupling_upper=float(p['coupling_upper']),unresolved_matched_mass=float(p['unresolved_matched_mass']),complete=p['complete']) for p in partials])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-started;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row['status'],row.get('metrics'),flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=dependencies))
if __name__=='__main__':main()
