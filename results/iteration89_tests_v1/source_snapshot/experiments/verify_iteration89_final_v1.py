"""Independent hashes, frozen replay, random domains, decisions and confidence.
No sampler calls; numerical moment/physical checks are scoped to saved states.
"""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib,math,statistics
import numpy as np
from verify_iteration87_final_v1 import words
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return bool(np.allclose(a,b,atol=2e-12,rtol=2e-12))
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);checks=0
 def check(v,detail):
  nonlocal checks
  checks+=1
  if not v:raise AssertionError(detail)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 panels=['pilot_v1','gates_v1','quality_block16_v1','quality_physical16_v1','quality_pnr5_v1','pnr_diagnostic_v1','pnr_state_audit_v1','latent_physicality_v1','remaining_state_v1','pnr_replay_v1','direct_timing_v1','pnr_timing_v1','joint_quality_v1','figures_v1','tests_v1']
 for panel in panels:
  base=ROOT/f'results/iteration89_{panel}';check(base.is_dir(),panel)
  if (base/'manifest.json').is_file():
   for n,h in read(base/'manifest.json')['artifacts'].items():check(H(base/n)==h,f'artifact {panel}/{n}')
  for n,h in read(base/'plan.json').get('hashes',{}).items() if (base/'plan.json').is_file() else []:check(H(ROOT/n)==h,f'frozen input/source {n}')
 tests=read(ROOT/'results/iteration89_tests_v1/validation.json');check(tests['passed'] and tests['failures']==0 and tests['errors']==0 and tests['unchanged_sources'] and tests['physical89_passed']==4 and tests['puretree89_passed']==3,'complete suite seven new tests')
 for n,h in read(ROOT/'results/iteration89_tests_v1/source_hashes.json').items():check(H(ROOT/n)==h,f'tested source {n}')
 gates=ROOT/'results/iteration89_gates_v1';check(read(gates/'summary.json')['passed'],'gates')
 for r in read(gates/'small.json')['rows']:
  z=np.load(gates/f"law_{r['case']}_{r['budget']}.npz");check(z['law'].min()>=0 and abs(z['law'].sum()-1)<2e-11 and close(abs(z['law']-z['target']).sum()/2,r['numerical_tv']),'small whole-law probabilities')
 for b in (16,64):
  for sector in (0,1):
   old=np.load(gates/f'oracle_{b}_{sector}.npz');new=np.load(gates/f'guarded_{b}_{sector}.npz');check(np.array_equal(old['clicks'],new['clicks']) and np.array_equal(old['errors'],new['errors']) and np.array_equal(old['statistics'][:,12:],new['statistics'][:,12:]),'physical matched path');check(np.all(new['bounds'][:,:,0]<=old['conditional']) and np.all(old['conditional']<=new['bounds'][:,:,1]),'numeric probabilities in saved bounds')
   for tag in ('serial','chunks'):
    z=np.load(gates/f'{tag}_{b}_{sector}.npz')
    for k in z:check(np.array_equal(z[k],new[k][:7],equal_nan=True),'thread/chunk values')
 for r in read(gates/'pure_detector.json')['rows']:
  z=np.load(gates/f"pure_{r['n']}_{r['length']}_{r['displaced']}.npz");check(np.array_equal(z['selected'][:,0],z['selected'][:,1]),'pure boundary detector decisions');check(np.all(z['rates']>=0) and close(z['rates'].sum(),1),'pure positive normalized rates')
 replay=ROOT/'results/iteration89_pnr_replay_v1';rs=read(replay/'summary.json');check(rs['passed'],'PNR replay summary');failed=smallN=fullN=0
 for row in rs['rows']:
  if row['case']=='full1152':new=np.load(replay/f"full_{row['sector']}_{row['offset']}.npz");old=np.load(ROOT/f"results/iteration86_pnr_quality_v1/raw_{row['sector']}_{row['offset']}.npz");fullN+=len(new['valid'])
  else:new=np.load(replay/f"small_{row['case']}_{row['retained']}_{row['offset']}.npz");old=np.load(ROOT/f"results/iteration88_small_pnr_quality_v1/raw_{row['case']}_{row['retained']}_{row['offset']}.npz");smallN+=len(new['valid']);failed+=int((~new['valid']).sum())
  check(np.array_equal(new['counts'],old['counts']) and np.array_equal(new['errors'],old['errors']) and np.array_equal(new['statistics'][:,:16],old['statistics'][:,:16]),'raw frozen PNR outputs and first16 statistics');check(np.array_equal(new['valid'],(old['errors'][:,0]==0)&(old['statistics'][:,1]==1)),'inclusive replay validity')
 check(smallN==786432 and fullN==8192 and failed==207,'replay scope exact, failures retained')
 plan=read(ROOT/'research/iteration89_quality_plan.json');check(plan['events']==sum(j['events'] for j in plan['jobs'])+1==53,'53-event allocation');delta=.01/53;datasets={};domains=[];fitstats={}
 for job in plan['jobs']:
  name=job['name'];base=ROOT/job['source'] if job['reused'] else ROOT/f'results/iteration89_quality_{name}_v1';combined=np.load(base/'combined.npz');datasets[name]=combined;N=len(combined['valid']);check(N==job['records'],'fixed job size')
  if job['reused']:continue
  check(read(base/'plan.json')['job']==job,'frozen new job');parts=[];good=[];stats_total=None;batch=256 if job['kind']=='pnr' else 64;experiment=int(job['experiment'],16)
  for sector in (0,1):
   for purpose in (range(1,5) if job['kind']=='pnr' else (5,)):
    address=(experiment,sector,purpose,0,N);check(experiment not in (0x86001001,0x88004002,0x89003001,0x89003002,*range(0x89002001,0x89002007)),'new domains excluded reused/pilots');check(not any((experiment,sector,purpose)==d[:3] for d in domains),'new domain overlap');domains.append(address)
   output=[];valid=[]
   for offset in range(0,N,batch):
    z=np.load(base/f'raw_{sector}_{offset}.npz');s=z['statistics'];e=z['errors'];v=(e[:,0]==0)&(s[:,1]==1 if job['kind']=='pnr' else s[:,5]==1152);valid.append(v);output.append(z['counts'] if job['kind']=='pnr' else z['clicks'])
    if job['kind']=='pnr':check(np.array_equal(z['valid'],v),'PNR completion')
    else:
     u=words(experiment,sector,offset,len(v),1152);k=z['kind'];b=z['bounds'];p=z['conditional'];ball=k==1;numeric=(k!=255)&~ball;check(np.all((u[ball]<b[:,:,0][ball])|(u[ball]>b[:,:,1][ball])),'resolved decision intervals');check(np.array_equal(z['clicks'][ball].astype(bool),u[ball]<b[:,:,0][ball]),'independent Philox interval decisions');check(np.array_equal(z['clicks'][numeric].astype(bool),u[numeric]<p[numeric]),'independent Philox numeric decisions');check(np.all(s[v,4]==s[v,8]+s[v,9]+s[v,10]) and np.all(s[:,6]==s[:,7]+s[:,8]+s[:,10]),'decision path accounting');check(np.all((k==1).sum(1)[v]==s[v,7]-s[v,9]) and np.all((k==2).sum(1)[v]==s[v,4]),'completed precision kinds')
     if s.shape[1]==21:check(np.all(s[:,12]==s[:,13]+s[:,14]) and np.all(s[:,14]==s[:,16:21].sum(1)) and np.all(s[:,15]<=1e-11),'five fit rejection reasons and residual')
     total=s.sum(0)
     if stats_total is None:stats_total=total
     else:stats_total+=total
   parts.append(np.concatenate(output));good.append(np.concatenate(valid))
  values=parts[0].astype(np.uint64)+parts[1] if job['kind']=='pnr' else np.bitwise_or(*parts);check(np.array_equal(combined['valid'],good[0]&good[1]),'two-spectrum inclusive failure')
  if job['kind']=='pnr':check(np.array_equal(combined['counts'],values) and np.array_equal(combined['clicks'],values>0),'PNR sum before threshold')
  else:check(np.array_equal(combined['clicks'],values),'two-spectrum OR');fitstats[name]=stats_total.tolist()
 # All confidence statements are recomputed from original combined arrays.
 quality=ROOT/'results/iteration89_joint_quality_v1';report=read(quality/'summary.json');check(report['passed'] and report['confidence']['events']==53,'quality completed');real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];realN=int(real.sum());realcdf=np.r_[np.cumsum(real)/realN,1.];order=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz')['order'];profile=read(ROOT/'results/iteration83_bias_profile_v1/summary.json')['reference']
 for job in plan['jobs']:
  name=job['name'];z=datasets[name];r=report['rows'][name];N=job['records'];pilot=job['pilot_records'];valid=z['valid'];bits=z['clicks'];eps=math.sqrt(math.log(2/delta)/(2*N));check(r['records']==N and r['pilot_records']==pilot and r['failed_records']==int((~valid).sum()),'confidence denominators')
  for kind,items in (('click',r['local_click']),('pnr',r['optical_pnr'])):
   for j,item in enumerate(items):
    h=np.load(quality/f'hist_{name}_{kind}_{j}.npz');target=h['target']
    if kind=='click':labels=np.sum(bits[:,plan['click_groups'][j]].astype(np.int64)*2**np.arange(8),axis=1)
    else:
     values=np.column_stack([z['counts'][:,order//8==mode].sum(1) for mode in plan['optical_pnr_pairs'][j]]);inside=(values<=64).all(1);labels=np.full(N,65**2,int);labels[inside]=(65*values[inside,0]+values[inside,1]).astype(int)
    labels[~valid]=len(target)-1;hist=np.bincount(labels,minlength=len(target));ph=np.bincount(labels[:pilot],minlength=len(target));vh=np.bincount(labels[pilot:],minlength=len(target));event=ph/pilot>target;check(np.array_equal(hist,h['histogram']) and np.array_equal(ph,h['pilot_histogram']) and np.array_equal(vh,h['validation_histogram']) and np.array_equal(event,h['event']),'independent histograms and pilot event');tv=abs(hist/N-target).sum()/2;gap=vh[event].sum()/(N-pilot)-target[event].sum();check(close(min(1,tv+math.sqrt(math.log(1/delta)/(2*N))+item['target_tail_upper']),item['nominal_joint99_tv_upper']) and close(max(0,gap-math.sqrt(math.log(1/delta)/(2*(N-pilot)))),item['nominal_joint99_tv_lower']),'TV upper/lower')
  values=bits.sum(1).astype(int);values[~valid]=1153;distance=abs(np.cumsum(np.bincount(values,minlength=1154))/N-realcdf).max();radius=eps+math.sqrt(math.log(2/delta)/(2*realN));item=r['experimental_click_count'];check(close(distance,item['empirical_cdf_distance']) and close(max(0,distance-radius),item['nominal_joint99_cdf_lower']) and close(min(1,distance+radius),item['nominal_joint99_cdf_upper']),'experimental click count CDF')
  for item in r['click_regions']:
   check(close(item['dkw_radius'],eps),'shared CDF event')
   if item['lower']<=0:continue
   first,end=item['first'],item['end'];width=end-first+1;values=bits[:,first:end].sum(1).astype(int);values[~valid]=width;ref=next(x for x in profile if (x['first'],x['end'])==(first,end));mu=ref['mean']
   if item['direction']=='decreasing':values=width-values;mu=width-mu
   L,U=item['clip_lower'],item['clip_upper'];emp=np.clip((values-L)/(U-L),0,1).mean();stop=(math.sqrt(ref['variance']+(U-mu)**2)-(U-mu))/2;check(close((mu-L-stop)/(U-L)-emp-eps,item['lower']),'whole-law clipped witness')
  if job['kind']=='pnr':
   ref=read(ROOT/'results/iteration85_joint_quality_v1/regional_photon_reference.json')
   for j,item in enumerate(r['photon_regions']):
    target=np.array([float(v) for v in ref['probabilities'][j]]);first,end=plan['count_regions'][j];K=len(target)-1;values=z['counts'][:,first:end].sum(1).astype(int);values[~valid]=K+1;hist=np.bincount(np.minimum(values,K+1),minlength=K+2);distance=abs(np.cumsum(hist[:-1])/N-np.cumsum(target)).max();check(close(distance,item['empirical_cdf_distance']) and close(min(1,distance+eps+float(ref['tail_upper'][j])),item['nominal_joint99_cdf_upper']),'photon CDF bound')
 for r in report['frugal_comparisons']:check(close(r['error_ratio_lower'],r['frugal_standalone_tv']/r['pnr_full_trajectory_marginal_tv_upper']),'scoped Frugal ratio')
 for n,ratio in report['direct_to_pnr_experimental_error_ratio_lower'].items():check(close(ratio,report['rows'][n]['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows']['pnr3']['experimental_click_count']['nominal_joint99_cdf_upper']),'direct-to-PNR error ratio')
 for old,tag in (('block16','block'),('unchecked16','unchecked')):check(close(report['projection_comparisons'][tag+'_to_physical_cdf_error_ratio_lower'],report['rows'][old]['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows']['physical16']['experimental_click_count']['nominal_joint99_cdf_upper']),'physical accuracy comparison')
 for tag in ('direct','pnr'):
  base=ROOT/f'results/iteration89_{tag}_timing_v1';t=read(base/'summary.json');check(t['passed'],'serial timing completed')
  for r in t['medians']:
   key='records' if tag=='direct' else 'retained';check(close(r['median_seconds'],statistics.median(v['seconds'] for v in t['rows'] if v['method']==r['method'] and v[key]==r[key])),'formal median')
  for c in t['checks']:check(c.get('identical_repetitions',c.get('identical_old_new',False)),'formal replay')
 audit=read(ROOT/'results/iteration89_pnr_state_audit_v1/summary.json');check(audit['passed'] and len(audit['rows'])==5,'five direct-matrix PNR diagnostics');check(all(not r['direct_in_range'] for r in audit['rows']),'saved numerical-state negative endpoints retained')
 import mpmath as mp
 mp.mp.dps=110
 latent=read(ROOT/'results/iteration89_latent_physicality_v1/summary.json');check(latent['passed'] and len(latent['rows'])==18,'latent counterexample scope')
 for r in latent['rows']:
  squeeze=mp.mpf(r['squeezing']);eta=mp.mpf(r['measured_fraction']);noise=mp.mpf(r['added_classical_variance']);q=-squeeze/(1+squeeze)+noise;p=squeeze/(1-squeeze)+noise
  for tag,scale,denom in (('initial',mp.mpf(1),False),('latent',mp.mpf(1),True),('remaining',1-eta,True)):
   a=scale*q/(1+eta*q) if denom else q;b=scale*p/(1+eta*p) if denom else p
   minimum=(a+b+1-mp.sqrt((a-b)**2+1))/2
   check(abs(minimum-mp.mpf(r[tag+'_uncertainty_minimum']))<mp.mpf('1e-85'),'independent analytic uncertainty minimum')
   if tag!='latent':check(minimum>=-mp.mpf('1e-100'),'physical analytic initial/remaining state')
 remaining=ROOT/'results/iteration89_remaining_state_v1';r=read(remaining/'summary.json');z=np.load(remaining/'state.npz');X=z['effective_covariance'];n=len(X)//2;omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);Q=X+np.eye(2*n)/2+1j*omega/2;w=z['witness'];rayleigh=np.vdot(w,Q@w)
 check(r['passed'] and r['selected_outputs']==522 and r['retained_ports']==[566,594] and r['next_mode']==632,'frozen remaining optical scope')
 check(close(rayleigh.real,float(r['independent80_digit_rayleigh_real'])) and abs(rayleigh.imag)<1e-12 and rayleigh.real<-.03,'independent negative remaining-output witness')
 A=z['compressed_optical'];R=np.block([[A.real,-A.imag],[A.imag,A.real]]);old=np.load(ROOT/'results/iteration88_failure_diagnostic_v1/state_project16_1_685.npz');check(close(R@old['covariance']@R.T,X),'actual optical covariance transformation');check(close(np.linalg.svd(z['optical'],compute_uv=False).max(),r['maximum_singular_value']) and r['maximum_singular_value']<=1,'contractive remaining optical map')
 final=dict(passed=True,checks=checks,full_suite=tests,small_pnr_replayed=smallN,small_failures_preserved=failed,full_pnr_replayed=fullN//2,fresh_streams=sum(d[4] for d in domains),fit_statistics_sum=fitstats,scope=__doc__,limits=['No new sample count from replay.','Disjoint PRNG domains do not prove iid.','Canonical latent-base interior check is sufficient but not necessary for physical remaining outputs; numerical, not a state interval certificate.','No full1152-pattern TV upper bound.']);save('validation.json',final);save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}));print(json.dumps({k:final[k] for k in ('passed','checks','small_pnr_replayed','small_failures_preserved','full_pnr_replayed','fresh_streams')}),flush=True)
if __name__=='__main__':main()
