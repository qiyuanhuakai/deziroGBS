"""Independent archive, full-count histogram, bound and public API checks."""
from pathlib import Path
import os,json,hashlib,shutil,sys,math,statistics,argparse,py_compile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_click_branches import NativeBranchClickSampler
from gbslab.native_pnr_branches import NativeBranchPNRSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=[]
def check(ok,label):
    checks.append(label);assert ok,label
for name in ('iteration71_branch_pilot_v1','iteration71_pnr_branch_pilot_v1','iteration71_pnr_branch_pilot_v2','iteration71_pnr_holdout','iteration71_frugal_comparison'):
    folder=ROOT/'results'/name;manifest=json.loads((folder/'manifest.json').read_text())['artifacts']
    for relative,h in manifest.items():check(H(folder/relative)==h,name+'/'+relative)
    if (folder/'plan.json').exists():
        plan=json.loads((folder/'plan.json').read_text())
        for relative,h in plan.get('source_hashes',{}).items():check(H(ROOT/relative)==h,'frozen source '+relative)
testdir=ROOT/'results/iteration71_tests';tests=json.loads((testdir/'validation.json').read_text());check(tests['passed'] and tests['failures']==tests['errors']==0 and tests['branch_click_passed']==7 and tests['branch_pnr_passed']==6,'complete native test suite')
for relative,h in json.loads((testdir/'source_hashes.json').read_text()).items():
    check(H(ROOT/relative)==h,'tested source unchanged '+relative);check(H(testdir/'source_snapshot'/relative)==h,'test snapshot '+relative)
small=ROOT/'results/iteration71_branch_pilot_v1';summary=json.loads((small/'summary.json').read_text());check(summary['passed'] and not summary['failures'],'click experiment passed')
for row in summary['small']:
    case,budget=row['case'],row['budget'];data=np.load(small/f'input_{case}.npz');law=np.load(small/f'law_{case}_{budget}.npz')['law'];target=np.array([float(p) for p in json.loads((small/f'target_{case}.json').read_text())['probabilities']]);check(np.array_equal(data['target'],target),'high precision target cast');tv=float(abs(law-target).sum()/2);check(abs(tv-row['tv'])<1e-15,'induced TV recomputed');check(abs(law.sum()-1)<1e-12 and law.min()>=0,'induced law normalization')
    if budget>=2**(row['modes']-1):check(tv<2e-10,'uncompressed full law')
v1=json.loads((ROOT/'results/iteration71_pnr_branch_pilot_v1/summary.json').read_text());v2=json.loads((ROOT/'results/iteration71_pnr_branch_pilot_v2/summary.json').read_text());check(sum(r.get('returncode',0)!=0 for r in v1['rows'])==5,'five retained v1 failures');check(v2['all_batches_succeeded'],'v2 pilot complete');check(len(v1['checks'])==len(v2['checks'])==24,'independent physical moment checks')
hold=ROOT/'results/iteration71_pnr_holdout';plan=json.loads((hold/'plan.json').read_text());result=json.loads((hold/'summary.json').read_text());radius=math.sqrt(math.log(8/.01)/(2*2000000));check(radius==plan['radius']==result['radius'],'predeclared simultaneous bound allocation');seeds=[];recomputed=[]
for row in result['rows']:
    case,method=row['case'],row['method'];data=np.load(ROOT/f'results/iteration71_pnr_branch_pilot_v2/input_{case}.npz');codes=data['codes'];cutoff=int(data['cutoff']);base=cutoff+1;law=data['target'];lookup=np.full(base**4,-1,np.int32);lookup[codes]=np.arange(len(codes));powers=np.array([base**j for j in range(4)],np.uint64);hist=np.zeros(len(law),np.int64);photons=0;maximum=0;extreme=0
    for batch in range(20):
        status=json.loads((hold/f'status_{case}_{method}_{batch}.json').read_text());check(status['returncode']==0,'holdout batch complete');seeds.append(status['seed']);counts=np.load(hold/f'counts_{case}_{method}_{batch}.npz')['counts'];check(counts.shape==(100000,4) and counts.dtype==np.uint32,'uncut uint32 counts');totals=counts.sum(axis=1);inside=totals<=cutoff;labels=np.full(len(counts),len(codes),np.int32);labels[inside]=lookup[(counts[inside].astype(np.uint64)@powers).astype(np.int64)];check(labels.min()>=0,'reference simplex mapping');hist+=np.bincount(labels,minlength=len(law));photons+=int(totals.sum());maximum=max(maximum,int(totals.max()));extreme+=int(sum(totals>=128))
    stored=np.load(hold/f'histogram_{case}_{method}.npz');check(np.array_equal(stored['histogram'],hist),'all 2M raw counts equal histogram');old=np.load(ROOT/'results/iteration71_pnr_branch_pilot_v2'/(f'counts_{case}_2_1.npz' if method=='branches' else f'baseline_{case}.npz'))['histogram'];event=old/old.sum()>law;check(np.array_equal(stored['fixed_event'],event),'event fixed from earlier pilot');empirical=hist/hist.sum();tv=float(abs(empirical-law).sum()/2);lower=max(0,float(empirical[event].sum()-law[event].sum())-radius);upper=min(1,tv+radius+float(law[-1]));check(abs(tv-row['empirical_tv'])<1e-15 and np.allclose([lower,upper],row['nominal_full_tv_interval'],rtol=0,atol=1e-15),'independent full PNR bounds');check(maximum==row['maximum_photons'] and extreme==row['events_ge128'] and photons/2000000==row['mean_photons'],'uncut tail and mean statistics');recomputed.append(dict(case=case,method=method,lower=lower,upper=upper,tv=tv))
check(len(set(seeds))==80 and min(b-a for a,b in zip(sorted(seeds),sorted(seeds)[1:]))>=2**40,'nonoverlapping row seed ranges')
check(recomputed[0]['upper']<recomputed[1]['lower'],'case3 strict quality improvement');check(recomputed[2]['upper']<.01,'case4 nominal upper below one percent')
runtime=[Path(shutil.which('g++')).parent];replays=0
for case in (3,4):
    d=np.load(ROOT/f'results/iteration71_pnr_branch_pilot_v2/input_{case}.npz');seed=json.loads((hold/f'status_{case}_branches_0.json').read_text())['seed'];expected=np.load(hold/f'counts_{case}_branches_0.npz')['counts']
    with NativeBranchPNRSampler(d['v'],d['x'],ROOT/'results/iteration71_pnr_branch_pilot_v2/pnr_branches.dll',noise_variance=d['variance'],workers=4,runtime_dirs=runtime) as sampler:
        actual=sampler.sample(2048,seed)['counts'];check(np.array_equal(expected[:len(actual)],actual),'public PNR replay');replays+=len(actual)
for case in (3,4,5):
    d=np.load(small/f'input_{case}.npz')
    with NativeBranchClickSampler(d['v'],d['x'],small/'branches.dll',mean=d['mean'],noise_variance=d['variance'],workers=4,runtime_dirs=runtime) as sampler:
        for budget in (1,2,8):
            expected=np.load(small/f'samples_{case}_{budget}.npz')['clicks'];actual=sampler.sample(2048,711001+case,branch_budget=budget)['clicks'];check(np.array_equal(actual,expected[:len(actual)]),'public click replay');replays+=len(actual)
bench=json.loads((ROOT/'results/iteration71_frugal_comparison/summary.json').read_text());check(bench['passed'] and len(bench['rows'])==51,'51 matched timing rows');medians={}
for panel in sorted(set(r['panel'] for r in bench['rows'])):
    medians[panel]={method:statistics.median(r['compute_seconds'] for r in bench['rows'] if r['panel']==panel and r['method']==method) for method in sorted(set(r['method'] for r in bench['rows'] if r['panel']==panel))}
for p in (ROOT/'experiments').glob('*iteration71*.py'):py_compile.compile(str(p),doraise=True)
shutil.copy2(__file__,OUT/Path(__file__).name);save('validation.json',dict(passed=True,checks=len(checks),raw_pnr_records_recomputed=8000000,public_records_replayed=replays,full_suite=tests,medians=medians,pnr_bounds=recomputed,source_sha256=H(Path(__file__))));save('checks.json',checks)
print(json.dumps(dict(passed=True,checks=len(checks),raw_pnr_records_recomputed=8000000,public_records_replayed=replays)),flush=True)
