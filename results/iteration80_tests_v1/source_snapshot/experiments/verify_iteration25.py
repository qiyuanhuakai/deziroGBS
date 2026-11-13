"""Audit physical spectra, saved states, exact small boxes and comparisons."""
import argparse
from functools import lru_cache
import hashlib
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
from gbslab.banded_gaussian_mps import pairing_covariance,pairing_log_normalization
from gbslab.gaussian import haar_unitary
from gbslab.mps_probe import geometric_top_weights,mps_to_dense
from gbslab.mps_overlap import scaled_overlap
from iteration23_support import digest,save,load_mps,mps_fingerprint
from iteration24_reference import partial_state_bounds,low_sector_gaussian
from iteration22_external import ARCHIVE_SHA


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/verify_iteration25.py','experiments/iteration23_support.py','experiments/iteration24_reference.py','experiments/iteration22_external.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))})
    hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',hashes); checks=[]; dependencies={}; direct=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    for name in ('spectra','panel','panel_retry','compare'):
        panel=ROOT/f'results/iteration25_{name}'; plan=read(panel/'plan.json')
        for f,s in plan['source_hashes'].items(): check(name+' source '+f,digest(ROOT/f)==s and digest(panel/'source_snapshot'/f)==s)
        dependencies.update(plan.get('dependencies',{}))
        if (panel/'summary.json').exists(): dependencies.update(read(panel/'summary.json').get('dependencies',{}))
    aborted=read(ROOT/'results/iteration25_panel/abort.json'); failures=read(ROOT/'results/iteration25_panel/rows.json')
    check('first path failure preserved',aborted['source_hashes_unchanged'] and aborted['completed_rows']==len(failures)==17 and all(r['status']=='failed' and 'relative_to' in r['traceback'] for r in failures))
    check('identical jobs on absolute-path retry',read(ROOT/'results/iteration25_panel/plan.json')['jobs']==read(ROOT/'results/iteration25_panel_retry/plan.json')['jobs'])
    spectra=ROOT/'results/iteration25_spectra'; summary=read(spectra/'summary.json'); cuts=read(spectra/'cuts.json'); bounds=read(spectra/'bounds.json')
    check('all physical cuts and planned bounds',summary['cuts']==len(cuts)==143 and summary['bound_rows']==len(bounds)==21 and summary['unchanged_sources'])
    check('spectra bytes',digest(spectra/'spectra.npz')==summary['output_sha256'])
    with np.load(spectra/'spectra.npz',allow_pickle=False) as a:
        B=a['pairing']; V=a['covariance']; physical=a['physical_tails']; weights={i:a[f'weights{i}'] for i in range(1,144)}; parameters={i:a[f'parameters{i}'] for i in range(1,144)}
    with np.load(ROOT/'results/iteration24_chain/case_56.npz',allow_pickle=False) as a: check('frozen Gaussian target unchanged',np.array_equal(B,a['forward']))
    check('covariance reproduction',np.max(np.abs(V-pairing_covariance(B)))<1e-13)
    complement_errors=[]
    for c in cuts:
        i=c['cut']; q=parameters[i]; w=weights[i]
        check(f'cut {i} saved parameters',np.array_equal(q,c['parameters']) and c['side_modes']==min(i,144-i))
        check(f'cut {i} top weights',len(w)==2048 and np.all(w>=0) and np.all(np.diff(w)<=1e-15) and math.fsum(w)<=1+1e-12 and np.max(np.abs(w-geometric_top_weights(q,2048)))<1e-14)
        check(f'cut {i} thermal vacuum',abs(w[0]-math.prod(1-q))<1e-14)
        ids=list(range(i,144)) if c['physical_side']=='left' else list(range(i)); n=len(ids)
        block=V[np.ix_(ids+[144+j for j in ids],ids+[144+j for j in ids])]
        omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
        ev=np.linalg.eigvals(1j*omega@block); nu=np.sort(ev.real)[n:][::-1]; expected=np.sort(c['symplectic_values'])[::-1]
        error=float(np.max(np.abs(nu[:len(expected)]-expected))); complement_errors.append(error)
        check(f'cut {i} complementary physical spectrum',error<1e-10 and np.max(np.abs(ev.imag))<1e-10 and (len(nu)==len(expected) or np.max(np.abs(nu[len(expected):]-1))<1e-10))
    for r in bounds:
        ts=[max(0.,1-math.fsum(w[:r['rank']])) for w in weights.values()]; tail=math.fsum(physical[:,[4,6,8].index(r['dimension'])]); upper=math.sqrt(min(1.,math.fsum(ts)+tail))
        check(f"bound chi{r['rank']} d{r['dimension']}",max(abs(r['max_cut_tail']-max(ts)),abs(r['schmidt_tail_sum']-math.fsum(ts)),abs(r['physical_tail_union']-tail),abs(r['pure_rank_distance_lower_estimate']-math.sqrt(max(ts))),abs(r['original_projector_trace_upper_estimate']-upper))<1e-12)
    panel=ROOT/'results/iteration25_panel_retry'; rows=read(panel/'rows.json'); plan=read(panel/'plan.json'); total=read(panel/'summary.json')
    check('all twenty runs terminal',len(rows)==len(plan['jobs'])==total['finished']==20 and total['accepted']==13 and total['budget_missed']==7 and total['other']==0 and total['unchanged_sources'])
    hafnian_checks=0
    for r,j in zip(rows,plan['jobs']):
        label=f"run {r['sequence']}"; check(label+' frozen configuration',all(r[k]==v for k,v in j.items()))
        check(label+' saved bytes',digest(ROOT/r['state']['path'])==r['state']['sha256'])
        state=load_mps(ROOT/r['state']['path']); check(label+' arrays',mps_fingerprint(state)==r['state']['array_fingerprint'])
        norm=scaled_overlap(state,state); check(label+' unit norm and rank',not norm['zero'] and abs(norm['log_abs'])<1e-10 and max(a.shape[2] for a in state)<=r['rank'] and all(a.shape[1]==r['dimension'] for a in state))
        info=r['info']; check(label+' budget and status',abs(math.sqrt(min(1.,math.fsum(info['cut_tail_estimates'])+math.fsum(info['marginal_tail_estimates'])))-r['construction_trace_budget'])<1e-12 and r['status']==('accepted' if r['construction_trace_budget']<=.1 else 'budget_missed') and r['validation_status']=='passed')
        if r['panel']=='quality':
            with np.load(panel/f"run_{r['sequence']:02d}.reference.npz",allow_pickle=False) as a: original=a['pairing']; box=a['gaussian_box']; U=a['U']
            expectedU=haar_unitary(r['modes'],np.random.default_rng(r['seed'])); check(label+' original input',np.array_equal(U,expectedU) and np.array_equal(original,-(U*np.tanh(np.linspace(.05,.5,r['modes'])))@U.T))
            dense=mps_to_dense(state); actual=math.sqrt(max(0.,1-abs(np.vdot(box,dense))**2)); check(label+' actual Gaussian distance',abs(actual-r['actual_original_distance'])<1e-12 and actual<=r['construction_trace_budget']+1e-10)
            # Repeated-index hafnian recursion pairs individual labels, not occupation recurrence.
            @lru_cache(None)
            def haf(labels):
                if not labels: return 1.+0j
                if len(labels)%2: return 0j
                return sum(original[labels[0],labels[k]]*haf(labels[1:k]+labels[k+1:]) for k in range(1,len(labels)))
            rng=np.random.default_rng(250900+r['sequence']); occupations=[(0,)*r['modes']]
            for _ in range(20):
                occ=[0]*r['modes']
                for site in rng.integers(0,r['modes'],size=int(rng.integers(1,6))*2): occ[int(site)]+=1
                if max(occ)<r['dimension']: occupations.append(tuple(occ))
            for occ in occupations:
                labels=tuple(i for i,n in enumerate(occ) for _ in range(n)); amp=math.exp(pairing_log_normalization(original))*haf(labels)/math.sqrt(math.prod(math.factorial(n) for n in occ))
                check(label+' hafnian '+str(occ),abs(amp-box[occ])<1e-13); hafnian_checks+=1
        else:
            with np.load(panel/f"run_{r['sequence']:02d}.low.npz",allow_pickle=False) as a: low=a['mps_low']; gauss=a['gaussian_low']; pairs=a['pairs']
            check(label+' full low sector',pairs.shape==(10440,2) and np.array_equal(pairs,np.array([(i,j) for i in range(144) for j in range(i,144)])))
            check(label+' Gaussian low formula',np.max(np.abs(gauss-low_sector_gaussian(B,pairing_log_normalization(B),pairs)))<1e-14)
            partial=partial_state_bounds(low,gauss); check(label+' low bounds',all(abs(partial[k]-r['low_sector'][k])<1e-13 for k in partial if k!='scope'))
            spec=next(x for x in bounds if x['rank']==r['rank'] and x['dimension']==4)
            check(label+' original physical tails',np.max(np.abs(np.array(info['cut_tail_estimates'])-[max(0.,1-math.fsum(weights[i][:r['rank']])) for i in range(1,144)]))<1e-10 and abs(math.fsum(info['marginal_tail_estimates'])-spec['physical_tail_union'])<1e-12)
            ids=sorted({0,*[k+1 for k,(i,j) in enumerate(pairs) if i==j and i in (0,35,71,107,143)],*np.random.default_rng(250901).choice(len(low),32,replace=False).tolist()}); errors=[]
            for index in ids:
                occ=np.zeros(144,int)
                if index: i,j=pairs[index-1]; occ[i]+=1; occ[j]+=1
                value=np.ones(1,complex)
                for A,n in zip(state,occ): value=value@A[:,n,:]
                errors.append(abs(complex(value[0])-low[index]))
            check(label+' direct low contraction',max(errors)<1e-12); direct.append(dict(rank=r['rank'],positions=ids,errors=errors))
        del state
    compare=ROOT/'results/iteration25_compare'; cr=read(compare/'rows.json'); cs=read(compare/'summary.json')
    expected_pairs={f'new_{r}:{f}' for r in (32,64,128,200) for f in ('own_recurrence','author_pco','author_hafnian')}|{f'new_{a}:new_{b}' for a in (32,64,128,200) for b in (32,64,128,200) if a<b}
    check('all eighteen comparison pairs',cs['finished']==cs['passed']==18 and {r['pair'] for r in cr}==expected_pairs and cs['unchanged_sources'])
    for r in cr: check(r['pair']+' two full distance routes',r['status']=='passed' and abs(1-r['pure_trace_distance_from_difference']**2-r['fidelity_from_overlap'])<1e-9 and max(abs(r['log_norm_qr_residual_left']),abs(r['log_norm_qr_residual_right']))<1e-9)
    check('original archive hash',digest(args.archive)==ARCHIVE_SHA); external={}
    with zipfile.ZipFile(args.archive) as z:
        for f in compare.glob('*_inputs.json'):
            for e in read(f):
                if e['name'].startswith('PCO mapping/'):
                    sha=hashlib.sha256(z.read(e['name'])).hexdigest(); check('archive '+e['name'],sha==e['sha256']); external[e['name']]=sha
    tests=ROOT/'results/iteration25_tests'; tr=read(tests/'validation.json')
    check('new tests and preserved baseline',tr['tests_run']==385 and tr['passed']==351 and tr['new_tests_passed']==5 and tr['unchanged_failure_ids'] and tr['unchanged_sources'])
    for name,s in read(tests/'source_hashes_at_start.json').items(): check('test source '+name,digest(ROOT/name)==s and digest(tests/'source_snapshot'/name)==s)
    for name,s in dependencies.items(): check('dependency '+name,digest(ROOT/name)==s)
    check('audit sources unchanged',hashes=={f:digest(ROOT/f) for f in sources})
    save(out/'direct_low_checks.json',direct)
    result=dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,dependencies=dependencies,external_arrays=len(external),direct_low_amplitudes=sum(len(r['positions']) for r in direct),independent_hafnian_coefficients=hafnian_checks,max_complement_spectrum_error=max(complement_errors))
    save(out/'validation.json',result); print(json.dumps({k:v for k,v in result.items() if k not in ('checks','dependencies')},indent=2))
    assert result['count']==result['passed'], 'Preserve failed audit results'


if __name__=='__main__': main()
