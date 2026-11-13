"""Regenerated PNR accuracy after explicit stream-domain separation.
Ten-event nominal joint99 family: three methods times full/last128 clicks and
total photons, plus the shared experimental click CDF. iid and fixed numerical
reference assumptions remain explicit; domain separation alone is not iid proof.
"""
from pathlib import Path
import os,argparse,json,hashlib,math
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
base=ROOT/'results/iteration84_pnr_regenerated_v1';profilepath=ROOT/'results/iteration83_bias_profile_v1/summary.json';realpath=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';files=[Path(__file__),base/'summary.json',base/'plan.json',profilepath,realpath,inp]+sorted(base.glob('*.npz'))+sorted(base.glob('ledger_*.json'));save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},regions=[[0,1152],[1024,1152]],delta_per_event=.001));profile=read(profilepath);real=np.load(realpath)['total_click_histogram'];N=int(real.sum());realcdf=np.cumsum(real)/N;realradius=math.sqrt(math.log(2000)/(2*N));xs=np.load(inp)['x'];mu=0.;var=0.
for x in xs:
 n=len(x)//2;g=np.eye(2*n)+2*x;mu+=(np.trace(g)-2*n)/4;var+=(np.sum(g*g.T)-2*n)/8
save('reference_photons.json',dict(mean=mu,variance=var,scope='real Gaussian trace invariants; two independent spectra; numerical float64 reference'))
domains=set();ledger_count=0
for path in sorted(base.glob('ledger_*.json')):
 for r in read(path):
  identity=(r['experiment'],r['sector'],r['record'],r['purpose']);assert identity not in domains;domains.add(identity);assert 0<=r['record']<2**32 and 0<=r['sector']<2**16 and 1<=r['purpose']<=4 and r['draws_used']<2**64;assert r['blocks_used']==(r['draws_used']+1)//2;ledger_count+=1
save('independent_domain_audit.json',dict(passed=True,domains=ledger_count,duplicates=0,scope='Disjoint actual-use domain records; no iid theorem claimed.'))
def bound(counts,mean,variance,size):
 n=len(counts);eps=math.sqrt(math.log(2000)/(2*n));end=max(512,int(counts.max())) if size is None else size;hist=np.bincount(counts,minlength=end+1);best=dict(nominal_joint99_full_tv_lower=0.,direction=None);choices=[('increasing',hist,mean)]
 if size is not None:choices.append(('decreasing',hist[::-1],size-mean))
 for direction,h,reference_mean in choices:
  integral=np.r_[0,np.cumsum(1-np.cumsum(h)/n)]
  for U in range(1,end+1):
   L=np.arange(U);width=U-L;emp=(integral[U]-integral[L])/width;stop=(math.hypot(math.sqrt(variance),U-reference_mean)-(U-reference_mean))/2;ref=(reference_mean-L-stop)/width;values=ref-emp-eps;j=int(np.argmax(values))
   if values[j]>best['nominal_joint99_full_tv_lower']:best=dict(nominal_joint99_full_tv_lower=float(values[j]),direction=direction,clip_lower=int(L[j]),clip_upper=U,reference_expectation_lower=float(ref[j]),empirical_expectation=float(emp[j]))
 return dict(records=n,reference_mean=mean,reference_variance=variance,empirical_mean=float(counts.mean()),empirical_variance=float(counts.var(ddof=1)),dkw_radius=eps,**best)
rows=[]
for panel in read(base/'summary.json')['rows']:
 method=panel['method']
 if not panel['success']:rows.append(dict(method=method,success=False,reason='Failed complete batch; no valid sampling law assigned'));continue
 parts=[]
 for s in range(2):
  d=np.load(base/f'{method}_{s}.npz');assert not d['errors'][:,0].any() and np.all(d['statistics'][:,1]==1);assert np.array_equal(d['counts'].sum(1),d['statistics'][:,0]);assert np.array_equal(d['statistics'][:,12],d['statistics'][:,0]+1) and np.array_equal(d['statistics'][:,13],d['statistics'][:,0]);parts.append(d['counts'].astype(np.uint64))
 d=np.load(base/f'{method}_combined.npz');counts=d['counts'];bits=d['clicks'];assert np.array_equal(counts,parts[0]+parts[1]) and np.array_equal(bits,counts>0);record=dict(method=method,success=True,click_regions=[])
 for first,end in ((0,1152),(1024,1152)):
  ref=next(r for r in profile['reference'] if (r['first'],r['end'])==(first,end));record['click_regions'].append(dict(first=first,end=end,**bound(bits[:,first:end].sum(1).astype(int),ref['mean'],ref['variance'],end-first)))
 h=np.bincount(bits.sum(1).astype(int),minlength=1153);assert np.array_equal(h,d['total_histogram']);cdf=np.cumsum(h)/len(bits);diff=abs(cdf-realcdf);j=int(np.argmax(diff));record['experimental_comparison']=dict(threshold=j,empirical_ks=float(diff[j]),nominal_joint99_full_bit_tv_lower=max(0.,float(diff[j])-record['click_regions'][0]['dkw_radius']-realradius),experimental_records=N,experimental_dkw_radius=realradius);record['photons']=bound(counts.sum(1).astype(int),mu,var,None);rows.append(record);save(f'row_{method}.json',record);print(method,'mean photons',record['photons']['empirical_mean'],'PNR lower',record['photons']['nominal_joint99_full_tv_lower'],'click lower',max(r['nominal_joint99_full_tv_lower'] for r in record['click_regions']),'experimental lower',record['experimental_comparison']['nominal_joint99_full_bit_tv_lower'],flush=True)
save('summary.json',dict(rows=rows,scope=__doc__,confidence='nominal joint99 conditional on iid generated/experimental records and fixed numerical Gaussian reference moments; explicit stream separation checked; not a proof of true iid or interval arithmetic'));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
