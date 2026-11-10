import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,math,shutil,sys,traceback
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.atom_certificate import selected_atoms
from gbslab.rational_gaussian import VacuumEnclosure,reciprocal_sqrt,outward
from gbslab.coupling_audit import normalized_rationals
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel,NumericalProbabilityError
from iteration22_external import digest,save
from iteration33_run import encode

def patterns(m,k):return [c for size in range(k+1) for c in combinations(range(m),size)]

def physical_covariance(job):
    m=job['modes']
    if job['family']=='thermal':return (1+2.**(-job['exponent']))*np.eye(2*m)
    a=np.array([(-1 if j%3==1 else 1)*2.**(-job['exponent']) for j in range(2*m)]);return np.eye(2*m)+np.outer(a,a)

def analytic_interval(job,k):
    m=job['modes'];exponent=job['exponent']
    if job['family']=='thermal':
        q=F(1,(1<<(exponent+1))+1);value=q**k*(1-q)**(m-k);return value,value
    lo=hi=F(0)
    for j in range(k+1):
        x=1+F(m-k+j,1<<(2*exponent));a,b=reciprocal_sqrt((x,x),256);coefficient=math.comb(k,j)
        if j%2:lo-=coefficient*b;hi-=coefficient*a
        else:lo+=coefficient*a;hi+=coefficient*b
    # Physical validity follows from the explicit classical-displacement model.
    return max(F(0),lo),min(F(1),hi)

def collect_native(V,job,chosen,library):
    table={};calls=0;duplicates=0
    with WireForestSampler(ThresholdModel(V),4,forest='chain',selection='recent',precision=job.get('precision','auto'),cache_size=0,library=library) as sampler:
        for pattern in chosen:
            clickset=set(pattern);prefix=()
            for j in range(job['modes']):
                calls+=1
                try:
                    q=float(sampler.conditional_click_probability(prefix));record=dict(prefix=''.join(map(str,prefix)),q_hex=q.hex());value=F(q)
                except NumericalProbabilityError as e:record=dict(prefix=''.join(map(str,prefix)),q_hex=None,error=str(e));value=None
                if prefix in table:
                    duplicates+=1
                    if table[prefix]['q_hex']!=record['q_hex']:raise ArithmeticError('Same fresh-trajectory prefix returned inconsistent conditionals: '+str(prefix))
                    table[prefix]['observations']+=1
                else:record['observations']=1;table[prefix]=record
                if value is None:break
                bit=int(j in clickset)
                if (value if bit else 1-value)==0:break
                prefix+=(bit,)
        diagnostics=sampler.kernel.cache_info()
    assert diagnostics['probability_calls']==calls
    return table,dict(native_calls=calls,duplicate_prefix_checks=duplicates,native_diagnostics=diagnostics)

def compact(r):return {k:v for k,v in r.items() if k not in ('records','patterns','target_intervals')}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['experiments/iteration34_run.py','experiments/iteration22_external.py','experiments/iteration33_run.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(ROOT/'research/iteration34_panel_plan.md',out/'frozen_plan.md');dependencies={}
    def reuse(path,iteration):
        sha=digest(ROOT/path);assert sha==json.loads((ROOT/f'results/iteration{iteration}_manifest.json').read_text())['artifacts'][path];dependencies[path]=sha;return ROOT/path
    library=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31);original=reuse('results/iteration31_panel/large_input.npz',31);oldrows=json.loads(reuse('results/iteration32_panel/rows.json',32).read_text());reuse('results/iteration33_panel/rows.json',33)
    for i in range(24,48):reuse(f'results/iteration32_panel/case_{i:02d}.npz',32)
    families=[('thermal',8),('rank_one',5),('rank_one',4)]
    jobs=[dict(panel='small',family=family,exponent=exponent,modes=m,max_clicks=m,precision='auto') for m in (2,4,8) for family,exponent in families]
    jobs += [dict(panel='reused',modes=8,source_case=i,max_clicks=8,start=oldrows[i]['start'],budget=oldrows[i]['budget'],forest=oldrows[i]['forest']) for i in range(24,48)]
    jobs += [dict(panel='large',modes=32,family=family,exponent=exponent,max_clicks=2,precision='auto') for family,exponent in families]
    jobs += [dict(panel='large',modes=144,family=family,exponent=exponent,max_clicks=1,precision='auto') for family,exponent in [('thermal',8),('thermal',40),('rank_one',5),('rank_one',4)]]
    jobs += [dict(panel='original',modes=144,max_clicks=1,precision='auto'),dict(panel='large',modes=144,family='thermal',exponent=40,max_clicks=1,precision='longdouble')]
    assert len(jobs)==42;save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,conditional_cache_size=0,target_interval_bits=256,gaussian_enclosure_bits=128));rows=[]
    for i,job in enumerate(jobs):
        start=perf_counter();row=dict(sequence=i,**job);print('starting',i,job,flush=True)
        try:
            m=job['modes'];chosen=patterns(m,job['max_clicks']) if job['panel']!='original' else [()]+[(j,) for j in range(0,144,9)];cert_data=None
            if job['panel']=='reused':
                data=np.load(ROOT/f"results/iteration32_panel/case_{job['source_case']:02d}.npz");p,total=normalized_rationals([float(x) for x in data['target']]);target={c:(p[sum(1<<j for j in c)],)*2 for c in chosen};table={}
                for length in range(m):
                    for mask in range(1<<length):
                        h=tuple((mask>>j)&1 for j in range(length));q=float(data['conditionals'][(1<<length)-1+mask]);table[h]=dict(prefix=''.join(map(str,h)),q_hex=q.hex(),observations=0)
                row.update(target_scope='exact normalized saved finite table, not a Gaussian rounding certificate',original_target_mass=str(total));stages=[0,2,4,6,8]
            else:
                V=np.load(original)['covariance'] if job['panel']=='original' else physical_covariance(job);begin=perf_counter();cert=VacuumEnclosure(V);row['certificate_seconds']=perf_counter()-begin;cert_atoms={c:cert.atom(c) for c in chosen if len(c)<=2};cert_data=dict(diagnostics=cert.diagnostics,atoms=[dict(pattern=c,interval=bounds) for c,bounds in cert_atoms.items()]);np.savez_compressed(out/f'case_{i:02d}.npz',covariance=V,R=cert.R)
                if job['panel']=='original':
                    target=cert_atoms;row['target_scope']='determinant-function intervals; Gaussian probability/TV interpretation requires quantum-valid covariance, not proved here'
                else:
                    by_count={k:analytic_interval(job,k) for k in range(job['max_clicks']+1)};target={c:by_count[len(c)] for c in chosen};row['target_scope']='physically valid analytic thermal or explicit classical rank-one displacement target'
                    for c,(lo,hi) in cert_atoms.items():
                        a,b=target[c]
                        if not lo<=a<=b<=hi:raise ArithmeticError('Gaussian enclosure did not contain analytic target interval: '+str(c))
                begin=perf_counter();table,info=collect_native(V,job,chosen,library);row.update(info);row['native_seconds']=perf_counter()-begin;stages=sorted(set([0,1,2,m])) if job['panel']=='small' else list(range(job['max_clicks']+1))
            stages=[k for k in stages if k<=job['max_clicks']];audits=[]
            def oracle(h):
                value=table[h]['q_hex'];return None if value is None else F(float.fromhex(value))
            for bits in (None,53):
                for k in stages:
                    subset=[c for c in chosen if len(c)<=k];r=selected_atoms(m,subset,oracle,lambda c:target[c],uniform_bits=bits);audits.append(dict(max_clicks=k,**compact(r)))
            output=dict(job=job,patterns=chosen,target_intervals=[target[c] for c in chosen],oracle=list(table.values()),gaussian_certificate=cert_data,audits=audits);save(out/f'case_{i:02d}.audit.json',encode(output));row.update(status='completed',unique_conditionals=len(table),selected_atoms=len(chosen),failure_nodes=sum(r['q_hex'] is None for r in table.values()),bounds=[dict(max_clicks=a['max_clicks'],uniform_bits=a['uniform_bits'],queries=a['queries'],tv_lower=float(a['tv_lower']),tv_upper=float(a['tv_upper']),confirmed_overlap=float(a['confirmed_overlap']),seen_failure_mass=float(a['seen_failure_mass']),target_selected_lower=float(a['target_selected_lower']),target_selected_upper=float(a['target_selected_upper'])) for a in audits])
            if cert_data is not None:row.update(whitening_residual=float(cert.delta),maximum_certificate_atom_width=float(max(hi-lo for lo,hi in cert_atoms.values())))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-start;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row['status'],row.get('bounds',[{}])[-1],flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=dependencies))
if __name__=='__main__':main()
