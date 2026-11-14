"""Independent artifact, random-domain, inclusive-quality and detector-tree audit.
No native DLL is loaded. Reconstruct complete rows, probability references and
confidence formulas; verify all replayed integer outputs and native source hashes.
"""
from pathlib import Path
import os,json,hashlib,argparse,math,statistics,functools,itertools
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
checks=[]
def check(v,label):
 if not v:raise AssertionError(label)
 checks.append(label)
save('plan.json',dict(driver_sha256=H(Path(__file__)),scope=__doc__))
for folder in sorted((ROOT/'results').glob('iteration85*')):
 if not folder.is_dir() or folder==OUT:continue
 if (folder/'manifest.json').exists():
  for name,h in read(folder/'manifest.json')['artifacts'].items():check(H(folder/name)==h,'artifact '+str(folder.name)+'/'+name)
 if (folder/'plan.json').exists():
  for name,h in read(folder/'plan.json').get('hashes',{}).items():check(H(ROOT/name)==h,'input '+name)
suite=read(ROOT/'results/iteration85_tests_v1/validation.json');check(suite['passed'] and suite['tree85_passed']==6 and suite['failures']==suite['errors']==0 and suite['unchanged_sources'],'complete suite and six new tests')
for name,h in read(ROOT/'results/iteration85_tests_v1/source_hashes.json').items():check(H(ROOT/name)==h,'test source '+name)
base=ROOT/'results/iteration85_full_quality_v1';quality=ROOT/'results/iteration85_joint_quality_v1';plan=read(base/'plan.json');summary=read(base/'summary.json');qsummary=read(quality/'summary.json');delta=plan['confidence']['delta_per_event'];check(delta*30<=.010000000001,'family budget');domains=set();method_data={}
for panel in summary['rows']:
 name=panel['method'];N=panel['records'];parts=[];validparts=[]
 for s in range(2):
  raw=[np.load(base/f'{name}_{s}_{i}.npz') for i in range(0,N,64)];errors=np.concatenate([r['errors'] for r in raw]);stats=np.concatenate([r['statistics'] for r in raw]);out=np.concatenate([r['clicks'] if name.startswith('exact') else r['counts'] for r in raw]);valid=(errors[:,0]==0)&(stats[:,5]==1152 if name.startswith('exact') else stats[:,1]==1);parts.append(out);validparts.append(valid)
  for entry in read(base/f'ledger_{name}_{s}.json'):
   e,sector,r,purpose=[entry[k] for k in ('experiment','sector','record','purpose')];address=(e,sector,r,purpose);check(address not in domains,'domain unique');domains.add(address);check(e==panel['experiment'] and sector==s and 0<=r<N,'domain fields')
   expected=1152 if purpose==5 else int(stats[r,11+purpose]);check(entry['words_generated']==expected,'random consumption')
  if not name.startswith('exact'):check(np.array_equal(out[valid].sum(1),stats[valid,0]) and np.array_equal(stats[valid,12],stats[valid,0]+1) and np.array_equal(stats[valid,13],stats[valid,0]),'photon draw identities')
 valid=validparts[0]&validparts[1];bits=np.bitwise_or(*parts) if name.startswith('exact') else parts[0].astype(np.uint64)+parts[1]>0;archived=np.load(quality/f'complete_{name}.npz');check(np.array_equal(valid,archived['valid']) and np.array_equal(bits,archived['clicks']),'all successful and failed rows retained');row=next(r for r in qsummary['rows'] if r['method']==name);check(row['failed_records']==int((~valid).sum()),'failure count');method_data[name]=(archived,row)
 for group,ids in enumerate(plan['groups']):
  target=np.r_[np.load(quality/f'reference_{group}.npz')['target'],0.];labels=sum(bits[:,port].astype(int)*(1<<j) for j,port in enumerate(ids));labels[~valid]=256;hist=np.bincount(labels,minlength=257);pilot=128;ph=np.bincount(labels[:pilot],minlength=257);event=ph/pilot>target;validation=np.bincount(labels[pilot:],minlength=257);lower=max(0.,float(validation[event].sum()/(N-pilot)-target[event].sum())-math.sqrt(math.log(1/delta)/(2*(N-pilot))));upper=min(1.,float(abs(hist/N-target).sum()/2)+math.sqrt(math.log(1/delta)/(2*N)));r=row['local_click'][group];check(abs(lower-r['nominal_joint99_marginal_tv_lower'])<1e-14 and abs(upper-r['nominal_joint99_marginal_tv_upper'])<1e-14,'independent inclusive local TV bounds');check(hist.sum()==N and hist[256]==row['failed_records'],'failure class target zero')
 for r in row['click_regions']:
  first,end=r['first'],r['end'];values=bits[:,first:end].sum(1).astype(int);values[~valid]=end-first+1;eps=math.sqrt(math.log(2/delta)/(2*N));check(abs(eps-r['dkw_radius'])<1e-15,'DKW family radius')
  if r['lower']>0:
   width=end-first+1;z=values if r['direction']=='increasing' else width-values;mu=r['reference_mean'] if r['direction']=='increasing' else width-r['reference_mean'];L,U=r['clip_lower'],r['clip_upper'];emp=float(np.clip((z-L)/(U-L),0,1).mean());stop=(math.sqrt(r['reference_variance']+(U-mu)**2)-(U-mu))/2;ref=(mu-L-stop)/(U-L);check(abs(emp-r['empirical_expectation'])<1e-12 and abs(ref-emp-eps-r['lower'])<1e-12,'independent clipped full-law witness')
real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];realcdf=np.r_[np.cumsum(real)/real.sum(),1.]
for name,(data,row) in method_data.items():
 values=data['clicks'].sum(1).astype(int);values[~data['valid']]=1153;cdf=np.cumsum(np.bincount(values,minlength=1154))/len(values);distance=float(abs(cdf-realcdf).max());radius=math.sqrt(math.log(2/delta)/(2*len(values)))+math.sqrt(math.log(2/delta)/(2*real.sum()));r=row['experimental_click_count'];check(abs(distance-r['empirical_cdf_distance'])<1e-14 and abs(min(1.,distance+radius)-r['nominal_joint99_cdf_upper'])<1e-14,'independent experimental count comparison')
check(len(domains)==summary['unique_domains'] and summary['domain_duplicates']==0,'complete actual-use domain registry')
for group in range(3):
 data=np.load(quality/f'reference_{group}.npz');vacuum=np.ones(256)
 for mask in range(1,256):
  ids=[j for j in range(8) if mask>>j&1];ids+=[j+8 for j in ids]
  for cov in data['covariances']:vacuum[mask]/=np.sqrt(np.linalg.det((np.eye(len(ids))+cov[np.ix_(ids,ids)])/2))
 law=np.zeros(256)
 for mask in range(256):
  zero=255^mask;sub=mask
  while True:
   law[mask]+=(-1)**sub.bit_count()*vacuum[zero|sub]
   if sub==0:break
   sub=(sub-1)&mask
 check(abs(law-data['target']).max()<2e-12 and abs(data['target'].sum()-1)<1e-13,'independent local Gaussian law')
regional=read(quality/'regional_photon_reference.json');data,row=method_data['core0']
for i,r in enumerate(row['photon_regions']):
 pmf=np.array([float(p) for p in regional['probabilities'][i]]);K=len(pmf)-1;values=data['counts'][:,r['first']:r['end']].sum(1).astype(int);values[~data['valid']]=K+1;h=np.bincount(np.minimum(values,K+1),minlength=K+2);tail=float(regional['tail_upper'][i]);ks=max(float(abs(np.cumsum(h[:-1])/len(values)-np.cumsum(pmf)).max()),tail);eps=math.sqrt(math.log(2/delta)/(2*len(values)));check(abs(ks-r['empirical_count_cdf_distance'])<1e-14 and abs(min(1.,ks+eps+tail)-r['nominal_joint99_count_cdf_upper'])<1e-14,'independent regional count CDF')
pilot=ROOT/'results/iteration85_tree_pilot_v1';probes=read(pilot/'probe_checks.json');check(probes['passed'],'tree probes passed')
for r in probes['checks']:
 z=np.load(pilot/f"probe_{r['case']}_{r['drawcase']}_{r['retained']}.npz");v,x,mean,modes=z['v'],z['x'],z['mean'],list(z['modes']);a=np.r_[v.real.T,-v.imag.T];b=np.r_[v.imag.T,v.real.T];rates=[]
 for mode in range(len(v)):
  fs=modes+[mode];total=0.
  for choices in itertools.product((0,1),repeat=len(fs)):
   F=np.column_stack([(a if bit==0 else b)[:,index] for index,bit in zip(fs,choices)]);mu=F.T@mean;cov=F.T@x@F
   @functools.lru_cache(None)
   def wick(ids):
    if not ids:return 1.
    first,*rest=ids;value=mu[first]*wick(tuple(rest))
    for j,index in enumerate(rest):value+=cov[first,index]*wick(tuple(rest[:j]+rest[j+1:]))
    return value
   total+=wick(tuple(j for i in range(len(fs)) for j in (i,i)))
  rates.append(total/2**len(fs))
 rates=np.array(rates)/sum(rates);check(abs(rates-z['new']).max()<2e-11 and abs(rates-z['old']).max()<2e-11,'independent real-Wick detector law');check(np.array_equal(z['selected'][:,0],z['selected'][:,1]),'same probe interval selections')
for rep in range(2):
 for s in range(2):
  a=np.load(pilot/f'replay_{rep}_old_{s}.npz');b=np.load(pilot/f'replay_{rep}_tree_{s}.npz');check(np.array_equal(a['counts'],b['counts']) and np.array_equal(a['statistics'],b['statistics'][:,:16]) and np.array_equal(a['errors'],b['errors']),'complete pilot replay')
replay=ROOT/'results/iteration85_tree_quality_replay_v1';check(read(replay/'summary.json')['passed'],'all2048 quality records replayed')
for sector in range(2):
 for offset in range(0,2048,64):
  old=np.load(base/f'core0_{sector}_{offset}.npz');new=np.load(replay/f'core0_{sector}_{offset}.npz');check(np.array_equal(old['counts'],new['counts']) and np.array_equal(old['statistics'],new['statistics'][:,:16]) and np.array_equal(old['errors'],new['errors']),'entire quality dataset replay')
timing=ROOT/'results/iteration85_tree_timing_v1';ts=read(timing/'summary.json');speed=[]
for count in (4,64):
 for rep in range(3):
  for s in range(2):
   a=np.load(timing/f'sector_{rep}_old_{count}_{s}.npz');b=np.load(timing/f'sector_{rep}_tree_{count}_{s}.npz');check(not a['errors'].any() and not b['errors'].any() and np.all(a['statistics'][:,1]==1) and np.all(b['statistics'][:,1]==1),'all timed trajectories complete');check(np.array_equal(a['counts'],b['counts']) and np.array_equal(a['statistics'],b['statistics'][:,:16]),'formal replay')
 for method in ('old','tree'):
  rows=[r for r in ts['rows'] if r['method']==method and r['records']==count];check(len(rows)==3 and all(r['success'] for r in rows),'three serial successful panels');speed.append(dict(method=method,records=count,median_seconds=statistics.median(r['seconds'] for r in rows)))
save('validation.json',dict(passed=True,checks=len(checks),full_suite=suite,unique_domains=len(domains),quality=qsummary['rows'],timing=speed,scope=__doc__,limitations=['Domain separation does not prove iid.','Numerical references are not interval-certified.','Marginal upper bounds do not bound full PNR/click TV.','Explicit native failures remain part of the output law.']));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Iteration85 independent checks passed',len(checks),flush=True)
