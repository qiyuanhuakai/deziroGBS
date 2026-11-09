"""Independent low-click probabilities and preserved platform evidence audit."""
import argparse
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import sys
import zipfile
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.gaussian import haar_unitary
from gbslab.exact import ThresholdModel
from gbslab.mps_probe import mps_to_dense
from gbslab.mps_overlap import scaled_overlap
from iteration23_support import digest,save,load_mps,mps_fingerprint
from iteration22_global import load_family
from iteration22_external import ARCHIVE_SHA


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args(); out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    sources=sorted({'experiments/verify_iteration26.py','experiments/iteration23_support.py','experiments/iteration22_global.py','experiments/iteration22_external.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))}); hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dest=out/'source_snapshot'/f; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dest)
    save(out/'source_hashes.json',hashes); checks=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    panel=ROOT/'results/iteration26_panel'; rows=read(panel/'rows.json'); plan=read(panel/'plan.json'); summary=read(panel/'summary.json'); dependencies=summary['dependencies'].copy()
    for name in ('panel','native'):
        location=ROOT/f'results/iteration26_{name}'; frozen=read(location/'plan.json')
        for f,s in frozen['source_hashes'].items(): check(name+' source '+f,digest(ROOT/f)==s and digest(location/'source_snapshot'/f)==s)
        check(name+' frozen configuration',digest(location/'frozen_plan.md')==digest(ROOT/'research/iteration26_panel_plan.md'))
    check('31 planned complete runs',len(rows)==len(plan['jobs'])==summary['planned']==summary['finished']==summary['passed']==31 and summary['unchanged_sources'])
    with np.load(panel/'gaussian.npz',allow_pickle=False) as z: B=z['pairing']; gaussian=z['probabilities']; patterns=z['patterns']
    expected_patterns=np.array([(-1,-1),*[(i,-1) for i in range(144)],*[(i,j) for i in range(144) for j in range(i+1,144)]])
    check('10441 complete threshold patterns',np.array_equal(patterns,expected_patterns))
    singular=np.linalg.svd(B,compute_uv=False); p0=math.exp(.5*math.fsum(np.log1p(-singular**2))); check('independent Gaussian vacuum',abs(p0-gaussian[0])<1e-13)
    independent=[p0]
    with mp.workdps(75):
        p0mp=mp.mpf(p0); q=[p0mp/mp.sqrt(1-abs(mp.mpc(B[i,i]))**2) for i in range(144)]
        independent.extend(float(x-p0mp) for x in q)
        for i,j in patterns[145:]:
            A=mp.matrix([[mp.mpc(B[i,i]),mp.mpc(B[i,j])],[mp.mpc(B[j,i]),mp.mpc(B[j,j])]])
            determinant=mp.det(mp.eye(2)-A*A.H); assert abs(determinant.imag)<mp.mpf('1e-70')
            independent.append(float(p0mp/mp.sqrt(determinant.real)-q[i]-q[j]+p0mp))
    independent=np.asarray(independent); gaussian_error=float(np.max(abs(independent-gaussian)))
    check('all Gaussian events independent local high precision',gaussian_error<1e-13 and np.all(independent>=0)); np.savez_compressed(out/'independent_gaussian_events.npz',probabilities=independent,absolute_errors=abs(independent-gaussian))
    direct=[]; full_errors=[]; direct_amplitudes=0
    oldrows=read(ROOT/'results/iteration25_panel_retry/rows.json'); states={f"new_{r['rank']}":r['state'] for r in oldrows if r['panel']=='candidate'}
    for row,job in zip(rows,plan['jobs']):
        i=row['sequence']; label=f'case {i}'; check(label+' configuration',all(row[k]==v for k,v in job.items()) and row['status']=='passed')
        path=panel/f'case_{i:02d}.npz'; check(label+' byte hash',digest(path)==row['output_sha256'])
        with np.load(path,allow_pickle=False) as z: ids=z['patterns']; p=z['mps_probabilities']; q=z['gaussian_probabilities']
        lower=.5*(math.fsum(abs(p-q))+abs(math.fsum(p)-math.fsum(q)))
        check(label+' positive masses and unreconditioned remainder',np.all(p>=0) and np.all(q>=0) and max(math.fsum(p),math.fsum(q))<=1+1e-10 and abs(max(0,1-math.fsum(p))-row['mps_remainder'])<1e-13 and abs(max(0,1-math.fsum(q))-row['gaussian_remainder'])<1e-13)
        check(label+' threshold bound reproduction',abs(lower-row['threshold_coarse_tv_lower_estimate'])<1e-13 and abs(abs(p[0]-q[0])-row['vacuum_probability_gap'])<1e-13 and lower>=abs(p[0]-q[0])-1e-13)
        if row['panel']=='quality':
            tensors=load_mps(panel/f'case_{i:02d}.state.npz')
            with np.load(panel/f'case_{i:02d}.reference.npz',allow_pickle=False) as z: U=z['U']; smallB=z['pairing']; saved_p=z['full_mps']; saved_q=z['full_gaussian']
            m,d=row['modes'],row['dimension']; expectedU=haar_unitary(m,np.random.default_rng(row['seed']))
            check(label+' independent original input',np.array_equal(expectedU,U) and np.array_equal(-(U*np.tanh(np.linspace(.05,.5,m)))@U.T,smallB))
            dense=mps_to_dense(tensors); full_p=np.zeros(1<<m)
            for occ in np.ndindex(dense.shape): full_p[sum(1<<j for j,n in enumerate(occ) if n)]+=abs(dense[occ])**2
            full_q=ThresholdModel(pairing_covariance(smallB)).full_distribution(); index=[sum(1<<int(j) for j in pair if j>=0) for pair in ids]
            check(label+' complete finite and Gaussian distributions',max(np.max(abs(full_p-saved_p)),np.max(abs(full_q-saved_q)),np.max(abs(full_p[index]-p)),np.max(abs(full_q[index]-q)))<1e-11)
            tv=.5*float(np.sum(abs(full_p-full_q))); full_errors.append(tv-lower)
            check(label+' coarse lower below full TV',lower<=tv+1e-11 and abs(tv-row['full_threshold_tv'])<1e-12)
        else:
            family=row['family']; check(label+' shared Gaussian events',np.array_equal(ids,patterns) and np.array_equal(q,gaussian))
            if family.startswith('new_'): tensors=load_mps(ROOT/states[family]['path'])
            else: tensors,_=load_family(family,ROOT/'results/iteration22_external',args.archive)
            check(label+' input array fingerprint',mps_fingerprint(tensors)==row['input_array_fingerprint'])
            norm=scaled_overlap(tensors,tensors); check(label+' independent full norm',not norm['zero'] and abs(norm['log_abs']-row['mps_input_log_norm_squared'])<1e-9)
            chosen=sorted({0,1,36,72,108,144,*np.random.default_rng(260902).choice(np.arange(145,len(ids)),20,replace=False).tolist()}); errors=[]
            for index in chosen:
                sites=[int(k) for k in ids[index] if k>=0]; probability=0.
                for occupations in itertools.product(*[range(1,tensors[j].shape[1]) for j in sites]):
                    counts=dict(zip(sites,occupations)); v=np.ones(1,complex); logscale=0.; zero=False
                    for site,A in enumerate(tensors):
                        v=v@A[:,counts.get(site,0),:]; scale=float(np.max(abs(v)))
                        if scale==0: zero=True; break
                        v/=scale; logscale+=math.log(scale)
                    if not zero: probability+=abs(v[0])**2*math.exp(2*logscale-norm['log_abs'])
                    direct_amplitudes+=1
                errors.append(abs(probability-p[index])); check(label+f' direct raw event {index}',errors[-1]<1e-11)
            direct.append(dict(family=family,positions=chosen,errors=errors)); del tensors
    native=ROOT/'results/iteration26_native'; ns=read(native/'summary.json'); probe=read(native/'probe.json'); build=read(native/'build.json'); nplan=read(native/'plan.json')
    check('legacy build failure retained',ns['legacy_build_returncode']==1 and 'libquadmath' in (native/'legacy.stderr.log').read_text(encoding='utf-8'))
    check('Windows native build bytes',ns['windows_compile_returncode']==0 and digest(native/'libbatched_forest.dll')==build['library_sha256'])
    check('native ABI safely rejected',ns['native_status']=='abi_mismatch_rejected' and not ns['kernel_created'] and not probe['scalar_format_compatible'] and probe['native_formats']['long_double_bytes']==16 and probe['ctypes_long_double_bytes']==probe['numpy_long_double_bytes']==8 and probe['native_formats']['long_double_mantissa_bits']==64 and probe['numpy_long_double_mantissa_bits']==53 and 'format mismatch' in probe['adapter_error'])
    check('compiler identity',digest(nplan['compiler'])==nplan['compiler_sha256'])
    for key,record in nplan['dependencies'].items(): check('compiler dependency '+key,record['exists'] and digest(record['path'])==record['sha256'])
    path=read(native/'relative_path.json'); result=read(native/'path_result.json')
    check('relative output path repaired',path['returncode']==0 and path['validated'] and path['state_exists'] and result['validation_status']=='passed' and digest(ROOT/result['state']['path'])==result['state']['sha256'])
    tests=ROOT/'results/iteration26_tests'; tr=read(tests/'validation.json'); passed=read(tests/'passed_test_ids.json'); baseline='results/iteration25_tests/validation.json'; dependencies[baseline]=digest(ROOT/baseline); old=read(ROOT/baseline)
    check('all old failures actually executed and passed',len(tr['resolved_failure_ids'])==4 and set(tr['resolved_failure_ids'])=={x['test'] for x in old['failure_details']} and all(x in passed for x in tr['resolved_failure_ids']))
    check('old import error resolved into thirteen passing tests',tr['previously_unimportable_module_tests_passed']==13 and sum(x.startswith('test_iteration5_quality.') for x in passed)==13)
    check('full suite accurately scoped',tr['tests_run']==408 and tr['passed']==379 and tr['failures']==tr['errors']==0 and tr['new_tests_passed']==11 and tr['skipped_cases']==29 and tr['skipped_classes']==7 and tr['executed_gate_passed'] and not tr['full_gate_passed'] and tr['unchanged_sources'])
    for name,sha in read(tests/'source_hashes_at_start.json').items(): check('test source '+name,digest(ROOT/name)==sha and digest(tests/'source_snapshot'/name)==sha)
    check('external archive hash',digest(args.archive)==ARCHIVE_SHA); external={}
    with zipfile.ZipFile(args.archive) as z:
        for file in panel.glob('*_inputs.json'):
            for e in read(file):
                if e['name'].startswith('PCO mapping/'):
                    sha=hashlib.sha256(z.read(e['name'])).hexdigest(); check('external '+e['name'],sha==e['sha256']); external[e['name']]=sha
    for name,sha in dependencies.items(): check('dependency '+name,digest(ROOT/name)==sha)
    check('audit sources unchanged',hashes=={f:digest(ROOT/f) for f in sources})
    save(out/'direct_event_checks.json',direct)
    report=dict(count=len(checks),passed=sum(c['passed'] for c in checks),checks=checks,dependencies=dependencies,independent_gaussian_events=len(independent),independent_gaussian_max_error=gaussian_error,independent_gaussian_local_decimal_precision=75,direct_mps_events=sum(len(d['positions']) for d in direct),direct_mps_amplitudes=direct_amplitudes,external_arrays=len(external),small_full_tv_min_margin=min(full_errors))
    save(out/'validation.json',report); print(json.dumps({k:v for k,v in report.items() if k not in ('checks','dependencies')},indent=2)); assert report['count']==report['passed'], 'Preserve failed audit'


if __name__=='__main__': main()
