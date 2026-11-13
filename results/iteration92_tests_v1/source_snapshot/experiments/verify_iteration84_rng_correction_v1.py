"""Independent verification of RNG repair and regenerated PNR error bounds.
No native library loaded. Check every input/ledger, reproduce reference Philox
words, reconstruct Gaussian moment inputs and recalculate bounded witnesses.
"""
from pathlib import Path
import os,json,argparse,hashlib,math
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
folders=('iteration84_rng_overlap_audit_v2','iteration84_rng_gates_v1','iteration84_rng_tests_v1','iteration84_pnr_regenerated_v1','iteration84_pnr_witness_v1')
save('plan.json',dict(scope=__doc__,verifier_sha256=H(Path(__file__)),folders=folders))
for name in folders:
 folder=ROOT/'results'/name
 for file,digest in read(folder/'manifest.json')['artifacts'].items():check(H(folder/file)==digest,'artifact '+name+'/'+file)
 for file,digest in read(folder/'plan.json')['hashes'].items():check(H(ROOT/file)==digest,'executed input '+file)
gate=read(ROOT/'results/iteration84_rng_gates_v1/validation.json');check(gate['passed'] and gate['known_answers']==55 and gate['native_addresses']==262144 and gate['domain_duplicates']==0,'domain gates passed');test=read(ROOT/'results/iteration84_rng_tests_v1/validation.json');check(test['passed'] and test['tests_run']==7 and test['failures']==0 and test['errors']==0 and test['unchanged_sources'],'focused RNG regressions')
def philox(counter,key):
 c=list(counter);k=list(key)
 for _ in range(10):
  low0=(c[0]*0xd2511f53)&0xffffffff;high0=(c[0]*0xd2511f53)>>32;low1=(c[2]*0xcd9e8d57)&0xffffffff;high1=(c[2]*0xcd9e8d57)>>32;c=[high1^c[1]^k[0],low1,high0^c[3]^k[1],low0];k=[(k[0]+0x9e3779b9)&0xffffffff,(k[1]+0xbb67ae85)&0xffffffff]
 return c
for r in read(ROOT/'results/iteration84_rng_gates_v1/known_answer_checks.json')['rows']:check(philox(r['counter'],r['key'])==r['expected']==r['actual'],'independent Philox KAT')
words=np.load(ROOT/'results/iteration84_rng_gates_v1/stream_words.npz')['words']
for draw,actual in enumerate(words):
 c=philox([draw//2,0,77,(1<<16)|3],[0x84001001,0]);lane=(draw%2)*2;check(int(actual)==c[lane]+(c[lane+1]<<32),'independent stream word '+str(draw))
base=ROOT/'results/iteration84_pnr_regenerated_v1';wit=ROOT/'results/iteration84_pnr_witness_v1';plan=read(base/'plan.json');summary=read(base/'summary.json');reference=read(ROOT/'results/iteration83_bias_profile_v1/summary.json');real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];N=int(real.sum());realcdf=np.cumsum(real)/N;domains=set();witnessrows=read(wit/'summary.json')['rows'];data=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');mu=0.;var=0.
for x in data['x']:
 n=len(x)//2;normal=(x[:n,:n]+x[n:,n:]+1j*(x[n:,:n]-x[:n,n:]))/2;anomalous=(x[:n,:n]-x[n:,n:]+1j*(x[n:,:n]+x[:n,n:]))/2;mu+=float(np.trace(normal).real);var+=float(np.trace(normal@(normal+np.eye(n))).real+np.sum(abs(anomalous)**2))
stored=read(wit/'reference_photons.json');check(abs(mu-stored['mean'])<1e-10 and abs(var-stored['variance'])<1e-9,'independent complex total photon moments')
core=ROOT/'results/iteration84_correlated_pnr_pilot_v1'
for s in range(2):
 d=np.load(core/f'input_{s}.npz');R,t,L=d['rotation'],d['t'],d['noise_factor'];U=R[:n,:n]+1j*R[n:,:n];P=(R*np.r_[(1-t)/(1+t),(1+t)/(1-t)])@R.T;check(np.max(abs(P+2*R@L@L.T@R.T-(np.eye(2*n)+2*data['x'][s])))<1e-11,'full correlated covariance retained '+str(s));check(np.max(abs(data['v']@U-d['v']))<1e-12,'pure core optical rotation '+str(s))
def audit_bound(values,mean,variance,r,maximum=None):
 eps=math.sqrt(math.log(2000)/(2*len(values)));check(abs(eps-r['dkw_radius'])<1e-15,'declared ten-event radius');check(abs(mean-r['reference_mean'])<1e-9 and abs(variance-r['reference_variance'])<1e-8,'reference witness moments');check(abs(values.mean()-r['empirical_mean'])<1e-12,'regenerated empirical mean')
 if r['direction'] is None:check(r['nominal_joint99_full_tv_lower']==0,'zero bound retained as inconclusive');return 0.
 if r['direction']=='decreasing':check(maximum is not None,'bounded count for decreasing witness');values=maximum-values;mean=maximum-mean
 lo,hi=r['clip_lower'],r['clip_upper'];emp=np.clip((values.astype(float)-lo)/(hi-lo),0,1).mean();stop=(math.hypot(math.sqrt(variance),hi-mean)-(hi-mean))/2;ref=(mean-lo-stop)/(hi-lo);lower=max(0.,ref-emp-eps);check(abs(emp-r['empirical_expectation'])<1e-13 and abs(ref-r['reference_expectation_lower'])<1e-9 and abs(lower-r['nominal_joint99_full_tv_lower'])<1e-9,'independent regenerated clipped witness');return lower
verified=[]
for row in summary['rows']:
 method=row['method'];panel=next(r for r in plan['panels'] if r['method']==method);parts=[]
 for s in range(2):
  d=np.load(base/f'{method}_{s}.npz');counts,stats,errors=d['counts'],d['statistics'],d['errors'];check(counts.shape==(panel['records'],1152) and counts.dtype==np.uint32,'full integer output '+str((method,s)));check(np.array_equal(counts.sum(1),stats[:,0]),'integer total statistics');ledger=read(base/f'ledger_{method}_{s}.json');check(len(ledger)==4*panel['records'],'complete domain ledger')
  for entry in ledger:
   e,sec,r,purpose=entry['experiment'],entry['sector'],entry['record'],entry['purpose'];check(e==panel['experiment'] and sec==s and 0<=r<panel['records'] and 1<=purpose<=4,'domain belongs to declared dataset');identity=(e&0xffffffff,e>>32,r,(sec<<16)|purpose);check(identity not in domains,'no native-key/domain duplicate');domains.add(identity);check(entry['draws_used']==int(stats[r,11+purpose]) and entry['blocks_used']==(entry['draws_used']+1)//2,'actual consumption matches ledger')
  if row['success']:check(not errors[:,0].any() and np.all(stats[:,1]==1) and np.array_equal(stats[:,12],stats[:,0]+1) and np.array_equal(stats[:,13],stats[:,0]),'all paths complete and consumed streams checked')
  parts.append(counts.astype(np.uint64))
 r=next(r for r in witnessrows if r['method']==method)
 if not row['success']:check(not r['success'],'failed dataset not assigned law');continue
 d=np.load(base/f'{method}_combined.npz');counts,bits=d['counts'],d['clicks'];check(np.array_equal(counts,parts[0]+parts[1]) and np.array_equal(bits,counts>0),'independent sector PNR sum then threshold');lowers=[]
 for region in r['click_regions']:
  first,end=region['first'],region['end'];ref=next(q for q in reference['reference'] if (q['first'],q['end'])==(first,end));lowers.append(audit_bound(bits[:,first:end].sum(1).astype(int),ref['mean'],ref['variance'],region,end-first))
 photonlower=audit_bound(counts.sum(1).astype(int),mu,var,r['photons']);h=np.bincount(bits.sum(1).astype(int),minlength=1153);check(np.array_equal(h,d['total_histogram']),'count histogram');ks=float(np.max(abs(np.cumsum(h)/len(bits)-realcdf)));lower=max(0.,ks-math.sqrt(math.log(2000)/(2*len(bits)))-math.sqrt(math.log(2000)/(2*N)));check(abs(lower-r['experimental_comparison']['nominal_joint99_full_bit_tv_lower'])<1e-14,'independent experimental CDF bound');verified.append(dict(method=method,records=len(bits),mean_photons=float(counts.sum(1).mean()),mean_clicks=float(bits.sum(1).mean()),nominal_joint99_pnr_tv_lower=photonlower,nominal_joint99_click_tv_lower=max(lowers),nominal_joint99_experimental_bit_tv_lower=lower))
check(len(domains)==summary['unique_domains'] and summary['domain_duplicates']==0,'all experiment domains covered');save('validation.json',dict(passed=True,checks=len(checks),unique_domains=len(domains),verified=verified,scope=__doc__,limitations=['Domain separation does not prove mathematical iid.','Nominal confidence depends on stated iid and numerical-reference assumptions.','Lower bounds cannot establish a full-law accuracy ordering.']));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Independent corrected validation passed',len(checks),flush=True)
