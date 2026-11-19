"""Independent saved-data audit. Does not invoke any sampling implementation.

Rebuild all60 confidence events, domain ranges, analytic photon-cap tails,
timing medians/replays and source/artifact hashes. Matrix-exponential evidence
and upstream numerical diagnostics remain scoped floating-point checks.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,hashlib,json,math,statistics,zipfile
from fractions import Fraction
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
checks=0
def check(ok,why):
 global checks
 checks+=1
 if not ok:raise AssertionError(why)
def bounds(labels,target,tail=0.,pilot=4096):
 n=len(labels);hist=np.bincount(labels,minlength=len(target));ph=np.bincount(labels[:pilot],minlength=len(target));validation=np.bincount(labels[pilot:],minlength=len(target));event=ph/pilot>target;tv=float(np.abs(hist/n-target).sum()/2);gap=float(validation[event].sum()/(n-pilot)-target[event].sum());return dict(empirical_tv=tv,lower=max(0,gap-math.sqrt(math.log(6000)/(2*(n-pilot)))),upper=min(1,tv+math.sqrt(math.log(6000)/(2*n))+tail))
def click_labels(counts,failed,m):
 labels=np.zeros(len(counts),int)
 for j in range(m):labels|=(counts[:,j]>0).astype(int)<<j
 labels[failed]=1<<m;return labels
def cap_tail(k,r,cap):
 q=Fraction(r)**2
 return 1-sum((Fraction(math.comb(k+l,k))*(1-q)**(k+1)*q**l for l in range((cap-k)//2+1)),Fraction(0))

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
for folder in sorted((ROOT/'results').glob('iteration92*')):
 if folder.is_dir() and (folder/'manifest.json').exists():
  entries=read(folder/'manifest.json')
  if 'artifacts' in entries:entries=entries['artifacts']
  for name,h in entries.items():check(sha(folder/name)==h,f'artifact {folder.name}/{name}')
for tag in ('build_v1','build_v2','gates_v1','gates_v2','quality_v1','baseline_quality_v1'):
 plan=read(ROOT/f'results/iteration92_{tag}/plan.json')
 for name,h in plan.get('hashes',{}).items():check(sha(ROOT/name)==h,f'plan hash {name}')
check(not read(ROOT/'results/iteration92_gates_v1/validation.json')['passed'],'failed v1 preserved')
check(len(read(ROOT/'results/iteration92_gates_v1/validation.json')['failures'])==5,'v1 five failed tests')
check(read(ROOT/'results/iteration92_gates_v2/validation.json')['passed'],'v2 gates')
identity=read(ROOT/'results/iteration92_identity_v1/validation.json');check(identity['passed'] and identity['checks']==72,'exponential identities');check(identity['source_sha256']==sha(ROOT/'experiments/iteration92_identity_v1.py'),'identity source')
suite=read(ROOT/'results/iteration92_tests_v1/validation.json');check(suite['passed'] and suite['unchanged_sources'] and suite['coherent92_passed']==8 and suite['precision92_passed']==1,'full suite')
for name,h in read(ROOT/'results/iteration92_tests_v1/source_hashes.json').items():check(sha(ROOT/name)==h,'suite source '+name)

base=ROOT/'results/iteration92_quality_v1';plan=read(base/'plan.json')['quality'];quality=[];streams=set();used_domains=0;records=0
for j,case in enumerate(plan['cases']):
 arrays=[]
 for offset in range(0,plan['records'],plan['batch_size']):
  with np.load(base/f'raw_{j}_{offset}.npz') as z:raw={k:z[k] for k in z.files}
  counts=raw['counts'];stats=raw['statistics'];errors=raw['errors'];check(np.all(errors==0),'fresh quality failures');check(np.array_equal(counts.sum(1),stats[:,0]),'photon stats');check(np.array_equal(stats[:,5],stats[:,0]+1),'waiting draw count');check(np.array_equal(stats[:,6],stats[:,0]),'detector draw count');check(np.all(stats[:,4]<=8.1e-18),'root residual')
  for row in range(len(counts)):
   for purpose,col in ((1,5),(2,6)):
    if stats[row,col]:
     domain=(plan['experiment_base']+j,0,offset+row,purpose);check(domain not in streams,'random domain reuse');streams.add(domain);used_domains+=1
  arrays.append(counts);records+=len(counts)
 counts=np.concatenate(arrays)[:,:case['detected_modes']];m=case['detected_modes'];failed=np.zeros(len(counts),bool)
 if 'reference88' in case:
  q=case['reference88'];refbase=ROOT/'results/iteration88_small_pnr_reference_v1';z=np.load(refbase/f'case_{q}.npz');info=read(refbase/f'case_{q}.json');target=np.r_[z['target'].ravel(),float(info['tail_mass']),0];tail=float(info['tail_upper']);ctarget=np.r_[z['target_click'],0];labels=np.full(len(counts),65**2,int);inside=(counts<=64).all(1);labels[inside]=counts[inside,0]*65+counts[inside,1]
 else:
  q=case['reference91'];k=case['k'];z=np.load(ROOT/f'results/iteration91_fock_quality_v1/law_{q}_{k}.npz');ix=np.argsort(z['keys']);keys=z['keys'][ix];prob=z['probabilities'][ix];tail=float(cap_tail(k,case['r'],case['cutoff']));target=np.r_[prob,tail,0];packed=np.zeros(len(counts),np.uint64)
  for mode in range(m):packed|=counts[:,mode].astype(np.uint64)<<(8*mode)
  labels=np.searchsorted(keys,packed);valid=labels<len(keys);valid[valid]&=keys[labels[valid]]==packed[valid];labels[~valid]=len(keys);ctarget=np.zeros((1<<m)+1);keyclick=np.zeros(len(keys),int)
  for mode in range(m):keyclick|=(((keys>>(8*mode))&255)>0).astype(int)<<mode
  ctarget[:1<<m]=np.bincount(keyclick,weights=prob,minlength=1<<m);ctarget[0]+=tail
 pnr=bounds(labels,target,tail);click=bounds(click_labels(counts,failed,m),ctarget,tail);click['lower']=max(0,click['lower']-tail)
 with np.load(base/f'hist_{j}.npz') as z:
  check(np.array_equal(np.bincount(labels,minlength=len(target)),z['pnr_histogram']),'saved PNR histogram');check(np.max(abs(target-z['pnr_target']))<3e-15,'saved PNR reference');check(np.max(abs(ctarget-z['click_target']))<3e-15,'saved click reference')
 tailcap=cap_tail(case['k'],case['r'],plan['photon_cap']);quality.append(dict(name=case['name'],pnr=pnr,click=click,analytic_cap_tail=float(tailcap),analytic_cap_tail_fraction=[str(tailcap.numerator),str(tailcap.denominator)],target_reference_tail=tail))
check(records==393216,'new record count');check(len(streams)==used_domains,'injective consumed domain ranges')

historical=[]
for row in read(ROOT/'results/iteration88_small_pnr_quality_v1/summary.json')['rows']:
 j=row['case_index'];d=row['retained'];z=np.load(ROOT/f'results/iteration88_small_pnr_quality_v1/combined_{j}_{d}.npz');counts=z['counts'];failed=~z['valid'];info=read(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.json');ref=np.load(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.npz');target=np.r_[ref['target'].ravel(),float(info['tail_mass']),0];labels=np.full(len(counts),65**2,int);inside=(counts<=64).all(1);labels[inside]=counts[inside,0]*65+counts[inside,1];labels[failed]=len(target)-1;historical.append(dict(name=row['case'],retained=d,pnr=bounds(labels,target,float(info['tail_upper']))))

baseline=[];bb=ROOT/'results/iteration92_baseline_quality_v1';oldrows=read(bb/'summary.json')['rows']
for j in range(2):
 ref=np.load(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.npz');info=read(ROOT/f'results/iteration88_small_pnr_reference_v1/case_{j}.json');target=np.r_[ref['target'].ravel(),float(info['tail_mass']),0];ctarget=np.r_[ref['target_click'],0];tail=float(info['tail_upper']);z=np.load(bb/f'pe_raw_{j}.npz');counts=z['counts'];failed=z['errors']!=0;labels=np.full(len(counts),65**2,int);inside=(counts<=64).all(1);labels[inside]=counts[inside,0]*65+counts[inside,1];labels[failed]=len(target)-1;pnr=bounds(labels,target,tail);click=bounds(click_labels(counts,failed,2),ctarget,tail);check(pnr==oldrows[2*j]['pnr'] and click==oldrows[2*j]['click'],'upstream confidence');baseline.append(dict(name=('weak','strong')[j],method='photonic-engines',pnr=pnr,click=click))
 z=np.load(bb/f'frugal_raw_{j}.npz');bits=z['clicks'];check(not bits[:,2:].any(),'Frugal vacuum outputs');click=bounds(bits[:,0].astype(int)+2*bits[:,1],ctarget,tail);check(click==oldrows[2*j+1]['click'],'Frugal confidence');baseline.append(dict(name=('weak','strong')[j],method='Frugal K4',click=click))
check(len(quality)*4+len(historical)*2+2*4+2*2==60,'one60-event family')
up=read(ROOT/'results/iteration92_pe_pilot_v1/plan.json')
with zipfile.ZipFile(ROOT/'results/iteration92_pe_pilot_v1/upstream_source.zip') as archive:
 for name,h in up['upstream_hashes'].items():check(hashlib.sha256(archive.read(name)).hexdigest()==h,'upstream snapshot '+name)

timings={}
for version in ('v1','v2'):
 folder=ROOT/f'results/iteration92_timing_{version}';result=read(folder/'summary.json');rows=result['rows'];check(result['passed'],'timing completed')
 for median in result['medians']:
  selected=[r for r in rows if r['case']==median['case'] and r['method']==median['method']];check(len(selected)==3 and statistics.median(r['seconds'] for r in selected)==median['seconds'],'timing median');previous=None
  for row in selected:
   with np.load(folder/f"raw_{row['case']}_{row['rep']}_{row['method']}.npz") as z:raw={k:z[k] for k in z.files}
   if previous is not None:
    for key in raw:check(np.array_equal(raw[key],previous[key]),'timing fixed control replay')
   previous=raw
 timings[version]=result['medians']
scaling=read(ROOT/'results/iteration92_scaling_v1/summary.json')
for row in scaling['medians']:
 selected=[r for r in scaling['rows'] if r['modes']==row['modes'] and r['threads']==row['threads']];check(len(selected)==3 and statistics.median(r['seconds'] for r in selected)==row['seconds'],'scaling median');raws=[np.load(ROOT/f"results/iteration92_scaling_v1/raw_{row['modes']}_{row['threads']}_{i}.npz") for i in range(3)]
 for i in (1,2):
  for key in raws[0].files:check(np.array_equal(raws[0][key],raws[i][key]),'scaling replay')
 for z in raws:z.close()
for m in (2,4,6,16,64,1152):
 a=np.load(ROOT/f'results/iteration92_scaling_v1/raw_{m}_1_0.npz');b=np.load(ROOT/f'results/iteration92_scaling_v1/raw_{m}_16_0.npz')
 for key in a.files:check(np.array_equal(a[key],b[key]),'scaling worker replay')
 a.close();b.close()
report=dict(passed=True,checks=checks,records=records,consumed_domains=used_domains,joint_events=60,joint_confidence=.99,new_quality=quality,historical88=historical,baselines=baseline,timings=timings,scaling=scaling['medians'],full_suite=suite,identity=dict(checks=72,max_error=identity['maximum_error'],max_error_cutoff36=max(r['error'] for r in identity['rows'] if r['cutoff']==36)),scope=__doc__)
(out/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');(out/'manifest.json').write_text(json.dumps({'validation.json':sha(out/'validation.json')},indent=2)+'\n');print(json.dumps(dict(passed=True,checks=checks,records=records,domains=used_domains)),flush=True)
