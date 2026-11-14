"""Independent raw-data, random-domain, projection, confidence and timing audit.
No native sampler calls. Reused data remain reused; PRNG separation is not iid.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,math,statistics
import numpy as np
from verify_iteration87_final_v1 import words
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return bool(np.allclose(a,b,rtol=2e-12,atol=2e-12))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);checks=0
 def check(v,detail):
  nonlocal checks
  checks+=1
  if not v:raise AssertionError(detail)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 panels=['window_pilot_v1','pnr_pilot_v1','small_pnr_reference_v1','small_pnr_quality_v1','project_pilot_v1','project_pilot_v2','project_gates_v1','joint_quality_v1','failure_diagnostic_v1','direct_timing_v1','pnr_timing_v1','tests_v1']+[f'full_quality_{n}_v1' for n in ('block16','project16','block64','project64')]
 for panel in panels:
  folder=ROOT/f'results/iteration88_{panel}';check(folder.is_dir(),panel)
  if (folder/'manifest.json').is_file():
   for name,h in read(folder/'manifest.json')['artifacts'].items():check(H(folder/name)==h,f'artifact {panel}/{name}')
  if (folder/'plan.json').is_file():
   for name,h in read(folder/'plan.json').get('hashes',{}).items():check(H(ROOT/name)==h,f'frozen source/input {name}')
 tests=read(ROOT/'results/iteration88_tests_v1/validation.json');check(tests['passed'] and tests['unchanged_sources'] and tests['failures']==0 and tests['errors']==0 and tests['projected88_passed']==5,'complete suite and five new native tests')
 for name,h in read(ROOT/'results/iteration88_tests_v1/source_hashes.json').items():check(H(ROOT/name)==h,f'tested source {name}')
 check(read(ROOT/'results/iteration88_project_pilot_v1/build.json')['returncode']!=0,'preserved compile failure')
 wp=read(ROOT/'results/iteration88_window_pilot_v1/summary.json');check(wp['completed'],'window attempt retained')
 # Independent double NumPy forward map covers every matrix coordinate.
 probe=ROOT/'results/iteration88_project_pilot_v2'
 for r in read(probe/'projection.json')['rows']:
  z=np.load(probe/f"projection_{r['dimension']}_{r['scale']}.npz");Y=z['output'];F=z['forms'];P=Y@F;G=np.eye(2)+F.T@P;q=1/math.sqrt(np.linalg.det(G));T=Y+q/(1-q)*P@np.linalg.solve(G,P.T);check(np.max(abs(T-z['target']))<2e-12,'independent full moment reconstruction');check(np.max(abs(Y-z['base']))<2e-12,'inverse base reconstruction')
 gates=ROOT/'results/iteration88_project_gates_v1';g=read(gates/'summary.json');check(g['passed'],'full-model gates')
 for r in g['rows']:
  check(r['comparison']['equal_decisions_errors_state_counters'] and r['comparison']['oracle_probabilities_enclosed'] and r['identical_projection_stats'],'matched numeric and guarded paths')
  b,s=r['budget'],r['sector'];old=np.load(gates/f'oracle_{b}_{s}.npz');new=np.load(gates/f'guarded_{b}_{s}.npz');check(np.array_equal(old['clicks'],new['clicks']) and np.array_equal(old['errors'],new['errors']) and np.array_equal(old['statistics'][:,[0,1,2,5]],new['statistics'][:,[0,1,2,5]]) and np.array_equal(old['statistics'][:,12:],new['statistics'][:,12:]),'raw matched gate');check(np.all(new['bounds'][:,:,0]<=old['conditional']) and np.all(old['conditional']<=new['bounds'][:,:,1]),'raw oracle enclosure')
  for name in ('serial','chunks'):
   raw=np.load(gates/f'{name}_{b}_{s}.npz')
   for key in raw:check(np.array_equal(raw[key],new[key][:7],equal_nan=True),'raw gate chunk/thread invariance')
 for r in g['prefix']:
  method,b=r['method'],r['budget'];left=np.load(gates/f'prefix_{method}_{b}_0.npz')['law'];right=np.load(gates/f'prefix_{method}_{b}_1.npz')['law'];labels=np.bitwise_or(np.arange(256)[:,None],np.arange(256)[None,:]);law=np.bincount(labels.ravel(),weights=np.outer(left,right).ravel(),minlength=256);saved=np.load(gates/f'prefix_combined_{method}_{b}.npz');check(close(law,saved['law']) and abs(abs(law-saved['target']).sum()/2-r['numerical_tv'])<1e-18,'first8 independent OR and numerical TV')
 plan=read(ROOT/'research/iteration88_full_quality_plan.json');check(plan['events']==4*8+14+1==47,'full family allocation');delta=.01/47;datasets={'pnr':np.load(ROOT/'results/iteration86_pnr_quality_v1/combined.npz')};domains=[];projection_totals={}
 def register(experiment,sector,purpose,first,count,tag):
  check(first>=0 and first+count<=2**32 and 0<=sector<2**16 and 0<purpose<2**16,'valid counter fields')
  check(experiment not in {0x88002001,0x88002002,0x88003001,0x88002201,0x88002202,*range(0x88002100,0x88002104)},'quality excluded pilot domains')
  for e,s,u,f,n,t in domains:check((experiment,sector,purpose)!=(e,s,u) or first+count<=f or f+n<=first,f'domain overlap {tag}/{t}')
  domains.append((experiment,sector,purpose,first,count,tag))
 for sector in (0,1):
  for purpose in range(1,5):register(0x86001001,sector,purpose,0,32768,'reused PNR86')
 def decision(raw,experiment,sector,offset):
  bits,p,b,k,s,e=(raw[key] for key in ('clicks','conditional','bounds','kind','statistics','errors'));n,m=bits.shape;u=words(experiment,sector,offset,n,m);reached=k!=255;ball=k==1;numeric=reached&~ball
  check(np.isfinite(p[reached]).all() and ((p[reached]>=0)&(p[reached]<=1)).all(),'midpoint range');check(np.all(b[:,:,0][reached]<=b[:,:,1][reached]),'interval order');check(np.all((u[ball]<b[:,:,0][ball])|(u[ball]>b[:,:,1][ball])),'resolved interval');check(np.array_equal(bits[ball].astype(bool),u[ball]<b[:,:,0][ball]),'Philox interval decisions');check(np.array_equal(bits[numeric].astype(bool),u[numeric]<p[numeric]),'Philox numeric decisions');complete=e[:,0]==0;check(np.all(s[complete,4]==s[complete,8]+s[complete,9]+s[complete,10]) and np.all(s[:,6]==s[:,7]+s[:,8]+s[:,10]),'completed wide and all ball accounting');check(np.all((k==1).sum(1)[complete]==s[complete,7]-s[complete,9]) and np.all((k==2).sum(1)[complete]==s[complete,4]),'completed kind accounting')
  # A throwing precision call has not incremented its completed-wide counter.
  for row in np.flatnonzero(~complete):check(s[row,8]+s[row,9]+s[row,10]-s[row,4] in (0,1) and np.all(k[row,int(e[row,1]):]==255),'preserved failed-step accounting')
  if s.shape[1]==20:check(np.all(s[:,12]==s[:,13]+s[:,14]) and np.all(s[:,14]==s[:,16:20].sum(1)) and np.all(s[:,15]<=1e-11),'projection success/rejection/residual')
  return (e[:,0]==0)&(s[:,5]==m)
 for job in plan['direct']:
  base=ROOT/f"results/iteration88_full_quality_{job['name']}_v1";check(read(base/'plan.json')['job']==job,'preregistered job');parts=[];good=[];stat=None
  for sector in range(2):
   register(int(job['experiment'],16),sector,5,0,job['records'],job['name']);bits=[];valid=[]
   for offset in range(0,job['records'],64):
    raw=np.load(base/f'raw_{sector}_{offset}.npz');bits.append(raw['clicks']);valid.append(decision(raw,int(job['experiment'],16),sector,offset));s=raw['statistics'];total=s.sum(0);total[11]=s[:,11].max()
    if len(total)==20:total[15]=s[:,15].max()
    if stat is None:stat=total
    else:
     prior=stat.copy();stat+=total;stat[11]=max(prior[11],total[11])
     if len(stat)==20:stat[15]=max(prior[15],total[15])
   parts.append(np.concatenate(bits));good.append(np.concatenate(valid))
  z=np.load(base/'combined.npz');check(np.array_equal(z['clicks'],np.bitwise_or(*parts)) and np.array_equal(z['valid'],good[0]&good[1]),'raw OR with inclusive failures');check(int((~z['valid']).sum())==read(base/'summary.json')['failed_records'],'failure denominator');datasets[job['name']]=z;projection_totals[job['name']]=stat.tolist()
 # Verify the separate small-PNR family, including every failed raw row.
 smallplan=read(ROOT/'research/iteration88_small_pnr_plan.json');small=ROOT/'results/iteration88_small_pnr_quality_v1';refs=ROOT/'results/iteration88_small_pnr_reference_v1';sdelta=.01/24;smallrows=read(small/'summary.json')['rows'];check(len(smallrows)*2==24,'small separate24-event family');failures={}
 for r in smallrows:
  j,R,N=r['case_index'],r['retained'],r['records'];experiment=0x88001000+16*j+R;check(r['experiment']==experiment and N==65536,'fixed small job');pilot=smallplan['pilot_records_for_lower_event'];counts=[];valid=[]
  for purpose in range(1,5):register(experiment,0,purpose,0,N,f'small {j}/{R}')
  for offset in range(0,N,1024):
   raw=np.load(small/f'raw_{j}_{R}_{offset}.npz');counts.append(raw['counts']);v=(raw['errors'][:,0]==0)&(raw['statistics'][:,1]==1);check(np.array_equal(raw['valid'],v),'small completion');valid.append(v)
  counts=np.concatenate(counts);valid=np.concatenate(valid);combined=np.load(small/f'combined_{j}_{R}.npz');check(np.array_equal(counts,combined['counts']) and np.array_equal(valid,combined['valid']),'small raw combination');check(int((~valid).sum())==r['failed'],'all small failures retained');failures[f'{j}/{R}']=r['failed'];h=np.load(small/f'hist_{j}_{R}.npz');target=h['target'];labels=np.full(N,65**2,int);inside=(counts<=64).all(1);labels[inside]=(65*counts[inside,0]+counts[inside,1]).astype(int);labels[~valid]=len(target)-1;hist=np.bincount(labels,minlength=len(target));ph=np.bincount(labels[:pilot],minlength=len(target));vh=np.bincount(labels[pilot:],minlength=len(target));event=ph/pilot>target
  check(np.array_equal(hist,h['histogram']) and np.array_equal(ph,h['pilot_histogram']) and np.array_equal(vh,h['validation_histogram']) and np.array_equal(event,h['event']),'small histograms and pilot event');rr=read(refs/f'case_{j}.json');tv=abs(hist/N-target).sum()/2;allow=float(rr['omitted_mixture_mass'])+float(rr['float_conversion_l1'])/2;upper=min(1,tv+math.sqrt(math.log(1/sdelta)/(2*N))+float(rr['tail_upper'])+allow);lower=max(0,vh[event].sum()/(N-pilot)-target[event].sum()-math.sqrt(math.log(1/sdelta)/(2*(N-pilot)))-allow);check(close(upper,r['nominal_joint99_full_pnr_tv_upper']) and close(lower,r['nominal_joint99_full_pnr_tv_lower']),'small complete PNR TV interval');click=(counts[:,0]>0).astype(int)+2*(counts[:,1]>0);click[~valid]=4;check(np.array_equal(np.bincount(click,minlength=5),h['click_histogram']),'small click coarsening')
  ref=np.load(refs/f'case_{j}.npz');check(close(target[:-2],ref['target'].ravel()) and close(target[-2],float(rr['tail_mass'])) and target[-1]==0,'small analytic reference and failure symbol')
  if j in (0,1):
   q=smallplan['cases'][j]['squeezing']**2;analytic=np.diag((1-q)*q**np.arange(65));check(close(ref['target'],analytic),'pure paired support and geometric law')
  elif j==3:check(close(ref['target'],np.outer(.5**(np.arange(65)+1),(2/3)**np.arange(65)/3)),'independent thermal product')
  else:
   q=smallplan['cases'][j]['squeezing']**2;eta=smallplan['cases'][j]['eta']
   for a,b in ((0,0),(1,3),(4,2),(10,10)):
    value=math.fsum((1-q)*q**k*math.comb(k,a)*math.comb(k,b)*eta**(a+b)*(1-eta)**(2*k-a-b) for k in range(max(a,b),257));check(close(value,ref['target'][a,b]),'independent positive lossy reference')
 # Quality intervals recomputed from the raw full-model combined records.
 quality=ROOT/'results/iteration88_joint_quality_v1';report=read(quality/'summary.json');check(report['passed'] and report['confidence']['events']==47,'full quality family');real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];realN=int(real.sum());realcdf=np.r_[np.cumsum(real)/realN,1.];order=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz')['order'];profile=read(ROOT/'results/iteration83_bias_profile_v1/summary.json')['reference']
 for name,z in datasets.items():
  result=report['rows'][name];N=len(z['valid']);pilot=2048 if name=='pnr' else next(j['pilot_records'] for j in plan['direct'] if j['name']==name);check(result['records']==N and result['pilot_records']==pilot,'full fixed sizes');good=z['valid'];bits=z['clicks'];eps=math.sqrt(math.log(2/delta)/(2*N));check(result['failed_records']==int((~good).sum()),'full quality failures')
  for kind,rows in (('click',result['local_click']),('pnr',result['optical_pnr'])):
   for j,r in enumerate(rows):
    h=np.load(quality/f'hist_{name}_{kind}_{j}.npz');target=h['target']
    if kind=='click':labels=(bits[:,plan['click_groups'][j]].astype(np.int64)*(2**np.arange(8))).sum(1)
    else:
     pair=plan['optical_pnr_pairs'][j];values=np.column_stack([z['counts'][:,order//8==mode].sum(1) for mode in pair]);inside=(values<=64).all(1);labels=np.full(N,65**2,int);labels[inside]=(65*values[inside,0]+values[inside,1]).astype(int)
    labels[~good]=len(target)-1;hist=np.bincount(labels,minlength=len(target));ph=np.bincount(labels[:pilot],minlength=len(target));vh=np.bincount(labels[pilot:],minlength=len(target));event=ph/pilot>target;check(np.array_equal(hist,h['histogram']) and np.array_equal(ph,h['pilot_histogram']) and np.array_equal(vh,h['validation_histogram']) and np.array_equal(event,h['event']),'full raw histograms');tv=abs(hist/N-target).sum()/2;gap=vh[event].sum()/(N-pilot)-target[event].sum();check(close(min(1,tv+math.sqrt(math.log(1/delta)/(2*N))+r['target_tail_upper']),r['nominal_joint99_tv_upper']) and close(max(0,gap-math.sqrt(math.log(1/delta)/(2*(N-pilot)))),r['nominal_joint99_tv_lower']),'full TV confidence')
  values=bits.sum(1).astype(int);values[~good]=1153;d=float(abs(np.cumsum(np.bincount(values,minlength=1154))/N-realcdf).max());radius=eps+math.sqrt(math.log(2/delta)/(2*realN));r=result['experimental_click_count'];check(close(d,r['empirical_cdf_distance']) and close(max(0,d-radius),r['nominal_joint99_cdf_lower']) and close(min(1,d+radius),r['nominal_joint99_cdf_upper']),'experimental CDF interval')
  for r in result['click_regions']:
   check(close(r['dkw_radius'],eps),'shared DKW budget')
   if r['lower']<=0:continue
   first,end=r['first'],r['end'];width=end-first+1;values=bits[:,first:end].sum(1).astype(int);values[~good]=width;ref=next(v for v in profile if (v['first'],v['end'])==(first,end));mu=ref['mean']
   if r['direction']=='decreasing':values=width-values;mu=width-mu
   L,U=r['clip_lower'],r['clip_upper'];emp=float(np.clip((values-L)/(U-L),0,1).mean());stop=(math.sqrt(ref['variance']+(U-mu)**2)-(U-mu))/2;check(close((mu-L-stop)/(U-L)-emp-eps,r['lower']),'full Gaussian TV lower witness')
  if name=='pnr':
   refs0=read(ROOT/'results/iteration85_joint_quality_v1/regional_photon_reference.json')
   for j,r in enumerate(result['photon_regions']):
    target=np.array([float(v) for v in refs0['probabilities'][j]]);K=len(target)-1;first,end=plan['count_regions'][j];values=z['counts'][:,first:end].sum(1).astype(int);values[~good]=K+1;hist=np.bincount(np.minimum(values,K+1),minlength=K+2);d=float(abs(np.cumsum(hist[:-1])/N-np.cumsum(target)).max());check(close(d,r['empirical_cdf_distance']) and close(min(1,d+eps+float(refs0['tail_upper'][j])),r['nominal_joint99_cdf_upper']),'PNR photon CDF')
 for b,r in report['projection_comparisons'].items():check(close(r['experimental_cdf_error_ratio_lower'],report['rows'][f'block{b}']['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows'][f'project{b}']['experimental_click_count']['nominal_joint99_cdf_upper']),'projection accuracy ratio')
 diagnostic=ROOT/'results/iteration88_failure_diagnostic_v1';dr=read(diagnostic/'summary.json');check(dr['passed'] and len(dr['rows'])==sum(report['rows'][n]['failed_records'] for n in ('project16','project64')),'every projected failure diagnosed')
 import mpmath as mp
 with mp.workdps(120):
  for r in dr['rows']:
   tag=f"{r['method']}_{r['sector']}_{r['record']}";A=mp.matrix([[mp.mpf(x) for x in row] for row in read(diagnostic/f'gram_{tag}.json')]);k=r['retained'];terms=[]
   for mask in range(1<<(k+1)):
    ids=[i for j in range(k+1) if mask>>j&1 for i in (2*j,2*j+1)];terms.append(1/mp.sqrt(mp.det(mp.matrix([[A[i,j] for j in ids] for i in ids]))) if ids else mp.mpf(1))
   mass=mp.fsum((-1)**m.bit_count()*terms[m] for m in range(1<<k));zero=mp.fsum((-1)**m.bit_count()*terms[m|(1<<k)] for m in range(1<<k));prob=(mass-zero)/mass;check(abs(prob-mp.mpf(r['independent100_digit_probability']))<mp.mpf('1e-80'),'120-digit failure probability');check(r['identical_replay'] and r['numeric']['returncode']!=0 and r['guarded']['returncode']!=0,'failure preserved across numeric and guarded paths')
   raw=np.load(diagnostic/f'guarded_{tag}.npz');base=ROOT/f"results/iteration88_full_quality_{r['method']}_v1";offset=r['record']//64*64;frozen=np.load(base/f"raw_{r['sector']}_{offset}.npz")
   for key in raw:check(np.array_equal(raw[key][0],frozen[key][r['record']-offset],equal_nan=True),'failed record exact diagnostic replay')
 for r in report['frugal_comparisons']:check(close(r['error_ratio_lower'],r['frugal_standalone_tv']/r['pnr_full_trajectory_marginal_tv_upper']),'scoped Frugal error ratio')
 for n,r in report['direct_to_pnr_experimental_error_ratio_lower'].items():check(close(r,report['rows'][n]['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows']['pnr']['experimental_click_count']['nominal_joint99_cdf_upper']),'direct to PNR accuracy ratio')
 for tag in ('direct','pnr'):
  base=ROOT/f'results/iteration88_{tag}_timing_v1';t=read(base/'summary.json');check(t['passed'],'serial timing success')
  if tag=='direct':
   for r in t['medians']:check(close(r['median_seconds'],statistics.median(x['seconds'] for x in t['rows'] if x['method']==r['method'] and x['records']==r['records'])),'direct timing median')
  else:
   for R,m in t['medians'].items():check(close(m,statistics.median(x['seconds'] for x in t['rows'] if x['retained']==int(R))),'PNR timing median')
  for c in t['checks']:check(c['identical_repetitions'],'deterministic timing replay')
 counts=dict(reused_full_pnr=262144,fresh_direct=sum(j['records']*2 for j in plan['direct']),fresh_small_pnr=12*65536*4);check(sum(d[4] for d in domains)==sum(counts.values()),'unique stream counts')
 final=dict(passed=True,checks=checks,full_suite=tests,stream_counts=counts,projection_statistics=projection_totals,small_pnr_failures=failures,scope=__doc__,limits=['Domain separation does not establish iid.','Two99-percent families have combined confidence at least98 percent.','Numerical references are not certified state intervals.','No full1152-pattern TV upper bound; preserved failures are part of the output law.']);save('validation.json',final);save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}));print(json.dumps(dict(passed=True,checks=checks,stream_counts=counts,small_pnr_failures=failures)),flush=True)
if __name__=='__main__':main()
