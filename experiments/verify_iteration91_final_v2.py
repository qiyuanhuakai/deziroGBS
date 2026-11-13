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
 panels=['build_v1','build_v2','build_v3','cache_audit_v2','cache_audit_v3','cache_audit_v4','sparse_gates_v1','zero_pivot_regression_v1','optical_gates_v1','optical_gates_v2','fock_build_v1','fock_quality_v1','quality_compact16_v1','quality_compact16_fixed_v1','joint_quality_v1','joint_quality_v2','timing_v1','timing_v2','fock_timing_v1','fock_scaling_v1','figures_v1','figures_v2','tests_v1','tests_v2']
 for panel in panels:
  base=ROOT/f'results/iteration91_{panel}';check(base.is_dir(),panel)
  if (base/'manifest.json').is_file():
   for n,h in read(base/'manifest.json')['artifacts'].items():check(H(base/n)==h,f'artifact {panel}/{n}')
  for n,h in read(base/'plan.json').get('hashes',{}).items() if (base/'plan.json').is_file() else []:check(H(ROOT/n)==h,f'frozen input/source {n}')
 tests=read(ROOT/'results/iteration91_tests_v2/validation.json');check(tests['passed'] and tests['failures']==0 and tests['errors']==0 and tests['unchanged_sources'] and tests['optical91_passed']==4 and tests['fock91_passed']==5 and tests['sparse91_passed']==4,'complete suite thirteen new tests')
 for n,h in read(ROOT/'results/iteration91_tests_v2/source_hashes.json').items():check(H(ROOT/n)==h,f'tested source {n}')
 gates=ROOT/'results/iteration91_optical_gates_v2';check(read(gates/'summary.json')['passed'],'gates')
 for r in read(gates/'small.json')['rows']:
  z=np.load(gates/f"law_{r['case']}_{r['budget']}.npz");check(z['law'].min()>=0 and abs(z['law'].sum()-1)<2e-11 and close(abs(z['law']-z['target']).sum()/2,r['numerical_tv']),'small whole-law probabilities')
 for b in (16,64):
  for sector in (0,1):
   old=np.load(gates/f'oracle_{b}_{sector}.npz');new=np.load(gates/f'guarded_{b}_{sector}.npz');check(np.array_equal(old['clicks'],new['clicks']) and np.array_equal(old['errors'],new['errors']) and np.array_equal(old['statistics'][:,12:],new['statistics'][:,12:]),'physical matched path');check(np.all(new['bounds'][:,:,0]<=old['conditional']) and np.all(old['conditional']<=new['bounds'][:,:,1]),'numeric probabilities in saved bounds')
   for tag in ('serial','chunks'):
    z=np.load(gates/f'{tag}_{b}_{sector}.npz')
    for k in z:check(np.array_equal(z[k],new[k][:7],equal_nan=True),'thread/chunk values')
 plan=read(ROOT/'research/iteration91_quality_plan_v2.json');check(plan['events']==sum(j['events'] for j in plan['jobs'])+1==69,'69-event allocation');delta=.01/69;datasets={};domains=[];fitstats={}
 for job in plan['jobs']:
  name=job['name'];base=ROOT/job['source'] if job['reused'] else ROOT/f'results/iteration91_quality_{name}_v1';combined=np.load(base/'combined.npz');datasets[name]=combined;N=len(combined['valid']);check(N==job['records'],'fixed job size')
  if job['reused']:continue
  check(read(base/'plan.json')['job']==job,'frozen new job');parts=[];good=[];stats_total=None;batch=256 if job['kind']=='pnr' else 64;experiment=int(job['experiment'],16)
  for sector in (0,1):
   for purpose in (range(1,5) if job['kind']=='pnr' else (5,)):
    address=(experiment,sector,purpose,0,N);check(experiment not in (0x86001001,0x88004002,0x89003001,0x89003002,0x91004001,0x90004001,0x91003001,0x91003002,0x91005001,0x91005002,0x90002001,0x90003001,0x90003002,*range(0x89002001,0x89002007)),'new domains excluded reused/pilots');check(not any((experiment,sector,purpose)==d[:3] for d in domains),'new domain overlap');domains.append(address)
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
 quality=ROOT/'results/iteration91_joint_quality_v2';report=read(quality/'summary.json');check(report['passed'] and report['confidence']['events']==69,'quality completed');real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];realN=int(real.sum());realcdf=np.r_[np.cumsum(real)/realN,1.];order=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz')['order'];profile=read(ROOT/'results/iteration83_bias_profile_v1/summary.json')['reference']
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
 for old,tag in (('block16','block'),('physical16','canonical')):check(close(report['projection_comparisons'][tag+'_to_optical_cdf_error_ratio_lower'],report['rows'][old]['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows']['optical16']['experimental_click_count']['nominal_joint99_cdf_upper']),'physical accuracy comparison')
 for old,tag in (('block16','block'),('optical16','optical')):check(close(report['projection_comparisons'][tag+'_to_compact_cdf_error_ratio_lower'],report['rows'][old]['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows']['compact16']['experimental_click_count']['nominal_joint99_cdf_upper']),'compact accuracy comparison')

 for old,tag in (('block16','block'),('optical16','optical'),('compact16','compact')):check(close(report['projection_comparisons'][tag+'_to_fixed_cdf_error_ratio_lower'],report['rows'][old]['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows']['compact16_fixed']['experimental_click_count']['nominal_joint99_cdf_upper']),'corrected compact quality comparison')
 failed=read(ROOT/'results/iteration91_tests_v1/validation.json');check(not failed['passed'] and failed['failures']==1 and 'model mismatch' in failed['failure_details'][0]['traceback'],'superseded regression failure preserved');check(read(ROOT/'results/iteration91_sparse_gates_v1/summary.json')['passed'],'four zero-pivot gates')
 zb=ROOT/'results/iteration91_zero_pivot_regression_v1';zero=read(zb/'summary.json');check(zero['passed'],'saved zero-pivot counterexample')
 for row in zero['rows']:
  z=np.load(zb/f"{row['version']}_suffix_{row['suffix']}.npz");e=np.linalg.norm(z['map'].conj().T@z['map']-z['target_gram']);check(close(e,row['gram_error']) and (e>0.9 if row['version']=='v2' else e==0),'sparse v2 error and v3 correction independently reconstructed')
 audit=ROOT/'results/iteration91_cache_audit_v4';ar=read(audit/'summary.json');check(ar['passed'],'packed cache complete audit');suffixes=0
 for row in ar['audits']:
  z=np.load(audit/f"maps_{row['case']}.npz");v=z['v'];n=v.shape[1];errors=[]
  for j in range(len(v)):
   k=min(n,len(v)-j);a0=z['maps'][z['offsets'][j]:z['offsets'][j+1]].reshape(k,n);g=v[j:].conj().T@v[j:];e=np.linalg.norm(a0.conj().T@a0-g)/(1+np.linalg.norm(g));check(e<2e-14 and close(e,z['errors'][j]),'exported suffix Gram without rank cutoff');errors.append(e);suffixes+=1
  required=32*sum(min(n,len(v)-j)*n-min(n,len(v)-j)*(min(n,len(v)-j)-1)//2 for j in range(len(v)));check(row['setup']['bytes']==required and close(max(errors),row['maximum_independent_gram_error']),'packed allocation and audit summary')
 check(read(audit/'rejected_v1.json')['gram_residual']>1e-3,'superseded wrong-adjoint cache explicitly rejected')
 for sector in (0,1):
  old=np.load(audit/f'old_{sector}.npz');new=np.load(audit/f'new_{sector}.npz')
  for name in ('clicks','errors','conditional'):check(np.array_equal(old[name],new[name]),'64-record old/new frozen decisions and probabilities')
  check(np.array_equal(old['statistics'][:,12:15],new['statistics'][:,12:15]),'unchanged frozen fit decisions')

 from fractions import Fraction as F
 import itertools,mpmath as mp
 mp.mp.dps=110
 def frac(x):return F(int(x[0]),int(x[1]))
 fbase=ROOT/'results/iteration91_fock_quality_v1';fq=read(fbase/'summary.json');check(fq['passed'] and len(fq['comparisons'])==20,'20 complete conditional comparisons');law_count=0
 for case in fq['cases']:
  number=case['case'];cut=case['cutoff'];m=case['modes'];rr=F(case['r']);s=rr*rr;u=np.load(fbase/f'input_{number}.npz')['u'];check(np.linalg.norm(u.conj().T@u-np.eye(2))<1e-12,'complex optical isometry');bank={};mass={}
  for row in case['native']:
   k=row['k'];z=np.load(fbase/f'law_{number}_{k}.npz');J=(cut-k)//2;trials=J+k+1
   # Independent negative-binomial/binomial-tail identity, no producer series.
   tail=sum((F(math.comb(trials,h))*(1-s)**h*s**(trials-h) for h in range(k+1)),F(0));mass[k]=1-tail;check(frac(row['exact_mass'])==1-tail,'exact binomial complement equals negative-binomial mass');check(abs(z['probabilities'].sum()-float(1-tail))<2e-11 and np.all(z['probabilities']>=0),'full retained numerical mass and positivity');check(len(np.unique(z['keys']))==len(z['keys']),'unique Fock occupations');total=sum(((z['keys']>>(8*j))&255).astype(int) for j in range(m));check(np.all(total<=cut) and np.all(total>=k) and np.all((total-k)%2==0),'whole total-photon layers');bank[k]=(z['keys'],z['probabilities']);law_count+=1
  for row in (r for r in fq['comparisons'] if r['case']==number):
   k=row['target_photons'];L=row['lattice'];a0=k%L;j=k//L;t=F(j,j+1);check(F(a0)+L*t/(1-t)==k,'positive lattice mixture preserves mean');inp_tv=1-(1-t)*t**j;check(frac(row['input_trace_distance'])==inp_tv,'input trace distance exact');z=np.load(fbase/f'comparison_{number}_{k}_{L}.npz');keys=z['keys'];target=np.zeros(len(keys));source_keys,pp=bank[k];target[np.searchsorted(keys,source_keys)]=pp;model=np.zeros_like(target);total_mass=F(0)
   for kk in range(a0,cut+1,L):
    weight=(1-t)*t**((kk-a0)//L);sk,pp=bank[kk];model[np.searchsorted(keys,sk)]+=float(weight)*pp;total_mass+=weight*mass[kk]
   check(np.array_equal(target,z['target']) and np.allclose(model,z['model'],atol=1e-17,rtol=1e-14),'complete model rebuilt from independently weighted raw layers');ta=1-mass[k];tb=1-total_mass;check(ta==frac(row['target_tail']) and tb==frac(row['model_tail']),'model and target exact tails');D=abs(target-model).sum()/2;lo=D+float(abs(ta-tb))/2;hi=min(D+float(ta+tb)/2,float(inp_tv));check(close(lo,row['pnr_tv_lower']) and close(hi,row['pnr_tv_upper']),'PNR disjoint-total tail bracket');labels=sum((((keys>>(8*j0))&255)>0).astype(int)*(1<<j0) for j0 in range(m));tc=np.bincount(labels,weights=target,minlength=1<<m);mc=np.bincount(labels,weights=model,minlength=1<<m);Dc=abs(tc-mc).sum()/2;check(close(max(0,Dc-float(ta+tb)/2),row['click_tv_lower']) and close(min(1,Dc+float(ta+tb)/2,float(inp_tv)),row['click_tv_upper']),'click merged-tail bracket');check(np.allclose(tc,z['target_click']) and np.allclose(mc,z['model_click']),'threshold labels independently rebuilt')
   if L>k:check(inp_tv==0 and np.array_equal(target,model),'explicit occupied core has no analytic closure error')
  for row in (r for r in fq['references'] if r['case']==number):
   key=row['key'];k=row['k'];n=[(key>>(8*j))&255 for j in range(m)];l=(sum(n)-k)//2;coefficient=mp.mpc(0)
   # Independent explicit allocations, rather than producer convolution.
   for split in itertools.product(*(range(q+1) for q in n)):
    if sum(split)!=k+l:continue
    term=mp.mpc(1)
    for j0,a0 in enumerate(split):term*=mp.binomial(n[j0],a0)*mp.mpc(complex(u[j0,0]))**a0*mp.mpc(complex(u[j0,1]))**(n[j0]-a0)
    coefficient+=term
   rmp=mp.mpf(case['r']);prob=abs(coefficient)**2*mp.factorial(k+l)*mp.factorial(l)/mp.fprod(mp.factorial(q) for q in n)*mp.binomial(k+l,k)*(1-rmp*rmp)**(k+1)*rmp**(2*l);check(abs(float(prob)-row['native'])<3e-13 and abs(prob-mp.mpf(row['reference']))<mp.mpf('1e-85'),'independent110-digit allocation amplitude')
  for row in (r for r in fq['samples'] if r['case']==number):
   k=row['k'];z=np.load(fbase/f'samples_{number}_{k}.npz');comparison=np.load(fbase/f'comparison_{number}_{k}_8.npz');keys=comparison['keys'];cdf=np.cumsum(comparison['target'].astype(np.longdouble));draws=words(0x91005001+number,0,0,65536,1).ravel();check(np.array_equal(draws,z['draws']),'independent conditional Philox words');idx=np.searchsorted(cdf,draws,side='right');valid=idx<len(keys);check(np.array_equal(z['failures'],~valid) and np.array_equal(z['keys'][valid],keys[idx[valid]]),'all native PNR draws and explicit tail outcomes')
 # Accuracy order differs between PNR and threshold readout on the same target.
 def item(k,L):return next(r for r in fq['comparisons'] if r['case']==0 and r['target_photons']==k and r['lattice']==L)
 check(item(4,2)['pnr_tv_upper']<item(4,1)['pnr_tv_lower'] and item(4,2)['click_tv_lower']>item(4,1)['click_tv_upper'],'strict PNR improvement with strict click degradation')
 check(item(6,3)['pnr_tv_lower']>item(6,1)['pnr_tv_upper'],'more occupation labels need not improve measured PNR law')

 timing=ROOT/'results/iteration91_timing_v2';t=read(timing/'summary.json');check(t['passed'],'serial complete-model timing')
 setup=read(timing/'setup_trials.json')
 for method,value in setup['medians'].items():check(close(value,statistics.median(x['seconds'] for x in setup['rows'] if x['method']==method)),'serial setup median')
 for row in t['medians']:check(close(row['median_seconds'],statistics.median(r['seconds'] for r in t['rows'] if (r['method'],r['cache'],r['records'])==(row['method'],row['cache'],row['records']))),'serial median')
 for method in ('optical90','compact91'):
  for N in (4,64):
   for sector in (0,1):
    z=np.load(timing/f'raw_0_{method}_cold_{N}_{sector}.npz')
    for rep in range(3):
     for cache in ('cold','warm'):
      other=np.load(timing/f'raw_{rep}_{method}_{cache}_{N}_{sector}.npz')
      for key in z:check(np.array_equal(z[key],other[key],equal_nan=True),'formal repetitions and cache modes preserve records')
 ft=read(ROOT/'results/iteration91_fock_timing_v1/summary.json');check(ft['passed'],'conditional timing')
 for row in ft['medians']:check(close(row['median_seconds'],statistics.median(r['seconds'] for r in ft['rows'] if (r['case'],r['k'],r['cache'],r['records'])==(row['case'],row['k'],row['cache'],row['records']))),'conditional medians')
 scale=read(ROOT/'results/iteration91_fock_scaling_v1/summary.json');check(scale['passed'] and len(scale['rows'])==9,'conditional scaling coverage')
 for row in scale['rows']:
  m=row['modes'];cut=row['cutoff'];count=sum(math.comb(n+m-1,m-1) for n in range(4,cut+1,2));check(count==row['patterns'] and row['returned_array_bytes']==16*count,'enumerated layer combinatorics');check(close(row['median_native_seconds'],statistics.median(x['seconds'] for x in row['repetitions'])),'scaling serial median');s=F(.3)**2;exact=1-sum((F(math.comb(l+4,4))*(1-s)**5*s**l for l in range((cut-4)//2+1)),F(0));check(exact==frac(row['tail']),'scaling exact truncation tail')
 final=dict(passed=True,checks=checks,full_suite=tests,fresh_streams=sum(d[4] for d in domains),fit_statistics_sum=fitstats,cache_suffixes=suffixes,fock_laws=law_count,amplitude_references=len(fq['references']),conditional_comparisons=20,scope=__doc__,limits=['No whole1152-pattern TV upper bound.','Conditional finite-occupation core is not a general GBS sampler.','Analytic tail bounds do not certify floating-point amplitude errors.','PRNG domain separation does not prove iid.']);save('validation.json',final);save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}));print(json.dumps({k:final[k] for k in ('passed','checks','fresh_streams','cache_suffixes','fock_laws','amplitude_references','conditional_comparisons')}),flush=True)
if __name__=='__main__':main()

