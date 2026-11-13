"""Audit positive results and preserved failed geometric preconditions."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.gaussian import haar_unitary
from gbslab.local_mps import local_circuit
from gbslab.banded_gaussian_mps import pairing_log_normalization,pairing_covariance
from iteration22_external import ARCHIVE_SHA,digest,save
from iteration22_global import load_family
from iteration24_reference import low_sector_gaussian,partial_state_bounds


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); out=args.output; out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/verify_iteration24.py','experiments/iteration22_external.py','experiments/iteration22_global.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'experiments').glob('iteration24*.py')),*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',hashes); checks=[]; dependencies={}
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    check('archive hash',digest(args.archive)==ARCHIVE_SHA)
    external_hashes={}
    for name in ('chain','posthoc','geometry'):
        panel=ROOT/f'results/iteration24_{name}'; plan=json.loads((panel/'plan.json').read_text())
        for f,sha in plan['source_hashes'].items(): check(name+' source '+f,digest(ROOT/f)==sha and digest(panel/'source_snapshot'/f)==sha)
        summary=json.loads((panel/'summary.json').read_text())
        dependencies.update(summary.get('dependencies',{}))
        with zipfile.ZipFile(args.archive) as z:
            for path in panel.glob('*inputs.json'):
                for entry in json.loads(path.read_text()):
                    if not entry['name'].startswith('PCO mapping/'): continue
                    raw=z.read(entry['name']); import hashlib
                    sha=hashlib.sha256(raw).hexdigest(); check(name+' external '+entry['name'],sha==entry['sha256']); external_hashes[entry['name']]=sha
    for name,sha in dependencies.items(): check('saved dependency '+name,digest(ROOT/name)==sha)
    rows=json.loads((ROOT/'results/iteration24_chain/rows.json').read_text()); plan=json.loads((ROOT/'results/iteration24_chain/plan.json').read_text())
    check('57 complete chain cases',len(rows)==len(plan['jobs'])==57 and all(r['status']=='passed' for r in rows))
    for row,job in zip(rows,plan['jobs']):
        i=row['sequence']; check(f'case {i} frozen job',all(row[k]==v for k,v in job.items()))
        path=ROOT/f'results/iteration24_chain/case_{i:02d}.npz'; check(f'case {i} file',digest(path)==row['output_sha256'])
        with np.load(path,allow_pickle=False) as a:
            B=a['forward']; C=a['reverse']; cf=complex(a['forward_log_scalar']); cr=complex(a['reverse_log_scalar'])
            check(f'case {i} bidirectional',np.max(np.abs(B-C))<1e-9 and abs(cf.real-cr.real)<1e-9 and abs(np.exp(1j*(cf.imag-cr.imag))-1)<1e-9)
            for direction in ('forward','reverse'):
                d=row['directions'][direction]; check(f'case {i} {direction} normalizable',np.linalg.norm(a[direction],2)<1-1e-12 and abs(pairing_log_normalization(a[direction])-d['normalized_log_vacuum_amplitude'])<1e-12)
                check(f'case {i} {direction} all convergence guards',len(d['steps'])==row['modes'] and all(max(s['left_internal_norm'],s['right_internal_norm'])<1-1e-12 and s['solve_residual']<1e-9 for s in d['steps']))
            if job['panel']=='known':
                rng=np.random.default_rng(job['seed']); U=haar_unitary(job['modes'],rng) if job['ensemble']=='haar' else local_circuit(job['modes'],12,rng)[1]
                original=-math.tanh(job['squeezing'])*(U@U.T)
                check(f'case {i} original input',np.array_equal(U,a['U']) and np.array_equal(original,a['original']))
            if job['panel']!='external': check(f'case {i} original recovery',np.max(np.abs(B-a['original']))<1e-9 and abs(cf.real-pairing_log_normalization(a['original']))<1e-9 and abs(np.exp(1j*cf.imag)-1)<1e-9)
            else: check('external covariance',np.max(np.abs(pairing_covariance(B)-a['covariance']))<1e-12)
    post=ROOT/'results/iteration24_posthoc'; gs=json.loads((post/'global.json').read_text())
    with np.load(ROOT/'results/iteration24_chain/case_56.npz',allow_pickle=False) as a: B=a['forward']; c=complex(a['forward_log_scalar'])
    with np.load(post/'global_pairing.npz',allow_pickle=False) as a: check('independent whole-chain Schur complement',gs['status']=='passed' and np.max(np.abs(B-a['pairing']))<1e-9)
    check('whole-chain determinant',abs(gs['log_determinant_abs']+2*c.real)<1e-8 and abs(complex(*gs['determinant_phase'])-np.exp(-2j*c.imag))<1e-8)
    direct_rows=[]
    for row in json.loads((post/'rows.json').read_text()):
        name=row['family']; path=post/(name+'.npz'); check(name+' low file',digest(path)==row['output_sha256'])
        with np.load(path,allow_pickle=False) as a: low=a['mps_low']; gauss=a['gaussian_low']; pairs=a['pairs']
        check(name+' complete two photon sector',pairs.shape==(144*145//2,2) and np.array_equal(pairs,np.array([(i,j) for i in range(144) for j in range(i,144)])))
        check(name+' Gaussian low coefficients',np.max(np.abs(gauss-low_sector_gaussian(B,pairing_log_normalization(B),pairs)))<1e-14)
        bound=partial_state_bounds(low,gauss)
        check(name+' bounds reproducible',all(abs(bound[k]-row[k])<1e-13 for k in bound if k!='scope'))
        check(name+' lower bounds respect ordering',row['vacuum_probability_gap']<=row['pnr_coarse_tv_lower_estimate']<=row['pure_trace_lower_estimate']+1e-12)
        tensors,_=load_family(name,ROOT/'results/iteration22_external',args.archive)
        ids=sorted({0,*[j+1 for j,(u,v) in enumerate(pairs) if u==v and u in (0,35,71,107,143)],*np.random.default_rng(240905).choice(len(low),32,replace=False).tolist()})
        errors=[]
        for index in ids:
            occ=np.zeros(144,int)
            if index: i,j=pairs[index-1]; occ[i]+=1; occ[j]+=1
            value=np.ones(1,complex); logscale=0.
            for A,n in zip(tensors,occ):
                value=value@A[:,n,:]; scale=float(np.max(np.abs(value)))
                if scale==0: break
                value/=scale; logscale+=math.log(scale)
            amplitude=0j if scale==0 else complex(value[0])*math.exp(logscale-.5*row['raw_log_norm_squared'])
            errors.append(abs(amplitude-low[index]))
        check(name+' direct raw contractions',max(errors)<1e-11); direct_rows.append(dict(family=name,positions=ids,errors=errors)); del tensors
    save(out/'direct_low_checks.json',direct_rows)
    geo=ROOT/'results/iteration24_geometry'; geom=json.loads((geo/'summary.json').read_text()); grams=json.loads((geo/'grams.json').read_text()); cuts=json.loads((geo/'cuts.json').read_text())
    check('geometry negative evidence preserved',geom['count']==301 and geom['passed']==156 and not geom['budget_conditions_passed'] and geom['conditional_pure_trace_upper_estimate'] is None)
    check('local nonisometry material',max(g['gram_pairing_error'] for g in grams)>.9 and geom['scale_log_error']>50)
    check('virtual thermal but wrong physical Schmidt spectra',len(cuts)==143 and all(c['thermal_covariance_residual']<1e-8 and c['physical_prefix_spectrum_error']>1e-8 for c in cuts))
    tests=ROOT/'results/iteration24_tests'; tr=json.loads((tests/'validation.json').read_text())
    check('eight new tests and inherited failures',tr['new_tests_passed']==8 and tr['unchanged_failure_ids'] and tr['unchanged_sources'])
    for name,sha in json.loads((tests/'source_hashes_at_start.json').read_text()).items(): check('test source '+name,digest(ROOT/name)==sha and digest(tests/'source_snapshot'/name)==sha)
    check('audit sources unchanged',hashes=={f:digest(ROOT/f) for f in sources})
    summary=dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,dependencies=dependencies,external_arrays=len(external_hashes),direct_low_amplitudes=sum(len(r['positions']) for r in direct_rows))
    save(out/'validation.json',summary); print(json.dumps({k:v for k,v in summary.items() if k not in ('checks','dependencies')},indent=2))
    if summary['count']!=summary['passed']: raise AssertionError('Preserve failed validation')


if __name__=='__main__': main()
