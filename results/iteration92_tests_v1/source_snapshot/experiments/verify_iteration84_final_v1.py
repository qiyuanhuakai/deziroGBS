"""Independent iteration84 archive, dot, CDF and formal timing verification.
No native library loaded. The RNG-correction verifier is independently retained.
Count-CDF upper bounds never become complete PNR TV upper bounds.
"""
from pathlib import Path
import os,json,argparse,hashlib,math,statistics
from fractions import Fraction
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
checks=[]
def check(ok,label):
 assert ok,label
 checks.append(label)
save('plan.json',dict(scope=__doc__,verifier_sha256=H(Path(__file__))))
for folder in sorted((ROOT/'results').glob('iteration84*')):
 if not folder.is_dir() or folder==OUT:continue
 if (folder/'manifest.json').exists():
  for name,digest in read(folder/'manifest.json').get('artifacts',{}).items():check(H(folder/name)==digest,'frozen artifact '+folder.name+'/'+name)
 if (folder/'plan.json').exists():
  for name,digest in read(folder/'plan.json').get('hashes',{}).items():check(H(ROOT/name)==digest,'executed input '+name)
tests=read(ROOT/'results/iteration84_tests_v1/validation.json');check(tests['passed'] and tests['rng84_passed']==7 and tests['exactsum84_passed']==6 and tests['failures']==tests['errors']==0 and tests['unchanged_sources'],'full tests and new domain/dot gates')
for name,digest in read(ROOT/'results/iteration84_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
corrected=read(ROOT/'results/iteration84_rng_verification_v1/validation.json');check(corrected['passed'] and corrected['unique_domains']==26624,'independent RNG correction audit');check(read(ROOT/'results/iteration84_full_quality_v1/aborted.json')['completed_sectors']==0,'unsafe legacy run stopped before assigning a dataset')
profile=ROOT/'results/iteration84_profile_v1';old=ROOT/'results/iteration83_weights_pilot_v1';fast=ROOT/'results/iteration84_exactsum_pilot_v1'
for s in range(2):
 d=np.load(profile/f'sector_{s}.npz');ref=np.load(old/f'sector_0_32_{s}.npz');check(np.array_equal(d['clicks'],ref['clicks']) and np.array_equal(d['conditional'],ref['conditional']) and np.array_equal(d['statistics'][:,:6],ref['statistics']),'profile retained arithmetic '+str(s));row=read(profile/f'row_{s}.json');times=d['statistics'][:,6:].sum(0);check(np.max(abs(times/np.sum(times)-row['phase_fractions']))<1e-14,'phase fractions recomputed')
for r in read(fast/'arithmetic_checks.json')['checks']:
 values=[[Fraction(v) for v in row] for row in r['inputs']];expected=sum((a+b)*c for a,b,c in values);scale=sum((abs(a)+abs(b))*abs(c) for a,b,c in values);actual=Fraction(r['high'])+Fraction(r['low']);check(abs(actual-expected)<=scale*Fraction(1,2**210),'independent rational dot '+str(r['case']))
for r in read(fast/'prefix_summary.json')['rows']:
 d=np.load(fast/f"prefix_{r['budget']}_{r['sector']}.npz");ref=np.load(old/f"prefix_{r['budget']}_{r['sector']}.npz");check(r['success'] and np.array_equal(d['law'],ref['law']) and r['law_difference_tv']==0,'whole prefix law unchanged '+str((r['budget'],r['sector'])))
for r in read(fast/'summary.json')['rows']:
 suffix=f"{r['rep']}_{r['budget']}_{r['sector']}";d=np.load(fast/f'sector_{suffix}.npz');ref=np.load(old/f'sector_{suffix}.npz');check(r['success'] and not d['errors'][:,0].any(),'fast full pilot completed');check(np.array_equal(d['clicks'],ref['clicks']) and np.array_equal(d['conditional'],ref['conditional']) and np.array_equal(d['statistics'][:,:6],ref['statistics']),'full frozen pilot replay '+suffix);check(d['statistics'][:,7].sum()==r['fallback_dots'] and d['statistics'][:,6].sum()==r['exact_dots'],'exact dot counters')
# Independently obtain all low count coefficients by Fourier inversion of the
# Gaussian generating function, not the experiment's power-series recurrence.
base=ROOT/'results/iteration84_photon_cdf_v1';data=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');eigenvalues=np.concatenate([np.linalg.eigvalsh(x) for x in data['x']]);stored=read(base/'reference.json');p=np.load(base/'reference.npz');check(np.max(abs(eigenvalues-p['eigenvalues']))<1e-13,'independent count spectrum');grid=np.exp(2j*np.pi*np.arange(4096)/4096);generating=np.exp(-.5*np.log(1+(1-grid[:,None])*eigenvalues[None,:]).sum(1));fourier=np.fft.fft(generating)/4096;difference=float(abs(fourier[:1025]-p['photon_pmf']).max());check(difference<5e-14,'independent Fourier count coefficients');check(abs(float(stored['trace_mean'])-eigenvalues.sum()/2)<1e-10 and abs(float(stored['trace_variance'])-np.dot(eigenvalues,1+eigenvalues)/2)<1e-9,'count trace moments');radius=float(stored['chernoff_radius']);log_tail=-.5*np.log(1+(1-radius)*eigenvalues).sum()-1025*math.log(radius);check(abs(log_tail-math.log(float(stored['tail_upper'])))<1e-9,'independent Chernoff tail')
countchecks=[]
for row in read(base/'summary.json')['rows']:
 method=row['method'];d=np.load(ROOT/f'results/iteration84_pnr_regenerated_v1/{method}_combined.npz');counts=d['counts'].sum(1).astype(int);h=np.bincount(counts,minlength=1025);cdf=np.cumsum(h)/len(counts);ks=float(np.max(abs(cdf-p['photon_cdf'])));eps=math.sqrt(math.log(2000)/(2*len(counts)));lower=max(0.,ks-eps);upper=min(1.,ks+eps+float(stored['tail_upper']));check(abs(ks-row['empirical_count_cdf_distance'])<1e-14 and abs(lower-row['nominal_joint99_population_count_cdf_lower'])<1e-14 and abs(upper-row['nominal_joint99_population_count_cdf_upper'])<1e-14,'independent population count-CDF interval '+method);countchecks.append(dict(method=method,empirical_ks=ks,nominal_joint99_count_cdf_lower=lower,nominal_joint99_count_cdf_upper=upper))
save('count_cdf_checks.json',dict(fourier_maximum_error=difference,rows=countchecks,count_only_improvement_lower=countchecks[0]['nominal_joint99_count_cdf_lower']/countchecks[1]['nominal_joint99_count_cdf_upper']))
def philox(c,k):
 c=list(c);k=list(k)
 for _ in range(10):
  a=c[0]*0xd2511f53;b=c[2]*0xcd9e8d57;c=[(b>>32)^c[1]^k[0],b&0xffffffff,(a>>32)^c[3]^k[1],a&0xffffffff];k=[(k[0]+0x9e3779b9)&0xffffffff,(k[1]+0xbb67ae85)&0xffffffff]
 return c
timingbase=ROOT/'results/iteration84_exactsum_timing_v1';timing=read(timingbase/'summary.json');check(timing['completed'] and len(timing['rows'])==12,'all twelve formal panels');speed=[]
for budget in (32,64):
 for rep in range(3):
  pair=[r for r in timing['rows'] if r['rep']==rep and r['budget']==budget];check(len(pair)==2,'paired formal panel retained')
  if not all(r['success'] for r in pair):continue
  for s in range(2):
   a=np.load(timingbase/f'sector_{rep}_old_{budget}_{s}.npz');b=np.load(timingbase/f'sector_{rep}_exact_{budget}_{s}.npz');check(not a['errors'][:,0].any() and not b['errors'][:,0].any() and np.all(a['statistics'][:,5]==1152) and np.all(b['statistics'][:,5]==1152),'all timed paths complete');check(np.array_equal(a['clicks'],b['clicks']) and np.array_equal(a['conditional'],b['conditional']),'timed exact replay')
   for row in range(4):
    draws=[]
    for block in range(576):
     words=philox([block,0,row,(s<<16)|5],[0x84003001+rep,0]);draws.extend([((words[0]+(words[1]<<32))>>11)*2**-53,((words[2]+(words[3]<<32))>>11)*2**-53])
    check(np.array_equal(np.array(draws)<a['conditional'][row],a['clicks'][row]),'independent domain draws and timed decisions')
 for method in ('old','exact'):
  rows=[r for r in timing['rows'] if r['method']==method and r['budget']==budget];speed.append(dict(method=method,budget=budget,records=4,successful=sum(r['success'] for r in rows),median_seconds=statistics.median(r['seconds'] for r in rows) if all(r['success'] for r in rows) else None))
save('speed_checks.json',dict(rows=speed));save('validation.json',dict(passed=True,checks=len(checks),full_suite=tests,correction_verifier_checks=corrected['checks'],corrected_quality=corrected['verified'],count_cdf=countchecks,timing=speed,limitations=['Count-CDF upper bounds are not full PNR TV upper bounds.','RNG domain separation is not a mathematical iid proof.','References use floating input covariances, not interval arithmetic.','Archived legacy PNR confidence guarantees remain withdrawn.']));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Iteration84 independent verification passed',len(checks),flush=True)
