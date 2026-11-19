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
 panels=['build_v1','optical_pilot_v1','optical_gates_v1','charge_certificate_v1','parity_gates_v1','parity_scope_v1','quality_optical16_v1','joint_quality_v1','timing_v1','figures_v1','tests_v1']
 for panel in panels:
  base=ROOT/f'results/iteration90_{panel}';check(base.is_dir(),panel)
  if (base/'manifest.json').is_file():
   for n,h in read(base/'manifest.json')['artifacts'].items():check(H(base/n)==h,f'artifact {panel}/{n}')
  for n,h in read(base/'plan.json').get('hashes',{}).items() if (base/'plan.json').is_file() else []:check(H(ROOT/n)==h,f'frozen input/source {n}')
 tests=read(ROOT/'results/iteration90_tests_v1/validation.json');check(tests['passed'] and tests['failures']==0 and tests['errors']==0 and tests['unchanged_sources'] and tests['optical90_passed']==4 and tests['parity90_passed']==3,'complete suite seven new tests')
 for n,h in read(ROOT/'results/iteration90_tests_v1/source_hashes.json').items():check(H(ROOT/n)==h,f'tested source {n}')
 gates=ROOT/'results/iteration90_optical_gates_v1';check(read(gates/'summary.json')['passed'],'gates')
 for r in read(gates/'small.json')['rows']:
  z=np.load(gates/f"law_{r['case']}_{r['budget']}.npz");check(z['law'].min()>=0 and abs(z['law'].sum()-1)<2e-11 and close(abs(z['law']-z['target']).sum()/2,r['numerical_tv']),'small whole-law probabilities')
 for b in (16,64):
  for sector in (0,1):
   old=np.load(gates/f'oracle_{b}_{sector}.npz');new=np.load(gates/f'guarded_{b}_{sector}.npz');check(np.array_equal(old['clicks'],new['clicks']) and np.array_equal(old['errors'],new['errors']) and np.array_equal(old['statistics'][:,12:],new['statistics'][:,12:]),'physical matched path');check(np.all(new['bounds'][:,:,0]<=old['conditional']) and np.all(old['conditional']<=new['bounds'][:,:,1]),'numeric probabilities in saved bounds')
   for tag in ('serial','chunks'):
    z=np.load(gates/f'{tag}_{b}_{sector}.npz')
    for k in z:check(np.array_equal(z[k],new[k][:7],equal_nan=True),'thread/chunk values')
 plan=read(ROOT/'research/iteration90_quality_plan.json');check(plan['events']==sum(j['events'] for j in plan['jobs'])+1==53,'53-event allocation');delta=.01/53;datasets={};domains=[];fitstats={}
 for job in plan['jobs']:
  name=job['name'];base=ROOT/job['source'] if job['reused'] else ROOT/f'results/iteration90_quality_{name}_v1';combined=np.load(base/'combined.npz');datasets[name]=combined;N=len(combined['valid']);check(N==job['records'],'fixed job size')
  if job['reused']:continue
  check(read(base/'plan.json')['job']==job,'frozen new job');parts=[];good=[];stats_total=None;batch=256 if job['kind']=='pnr' else 64;experiment=int(job['experiment'],16)
  for sector in (0,1):
   for purpose in (range(1,5) if job['kind']=='pnr' else (5,)):
    address=(experiment,sector,purpose,0,N);check(experiment not in (0x86001001,0x88004002,0x89003001,0x89003002,0x90002001,0x90003001,0x90003002,*range(0x89002001,0x89002007)),'new domains excluded reused/pilots');check(not any((experiment,sector,purpose)==d[:3] for d in domains),'new domain overlap');domains.append(address)
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
 quality=ROOT/'results/iteration90_joint_quality_v1';report=read(quality/'summary.json');check(report['passed'] and report['confidence']['events']==53,'quality completed');real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];realN=int(real.sum());realcdf=np.r_[np.cumsum(real)/realN,1.];order=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz')['order'];profile=read(ROOT/'results/iteration83_bias_profile_v1/summary.json')['reference']
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

 from fractions import Fraction as F
 def frac(a):return F(int(a[0]),int(a[1]))
 charge=read(ROOT/'results/iteration90_charge_certificate_v1/summary.json');check(charge['passed'] and len(charge['rows'])==84,'84 conditional charge certificates')
 for row in charge['rows']:
  a,b=row['removed'];r=F.from_float(row['squeezing']);u=r*r/(1-r*r);poly=[F(1)]
  for j in [*range(a),*range(b)]:
   nxt=[F(0)]*(len(poly)+1)
   for k,x in enumerate(poly):nxt[k]-=j*x;nxt[k+1]+=x
   poly=nxt
  stirling=[[1]]
  for k in range(1,len(poly)+1):
   old=stirling[-1];stirling.append([(old[j-1] if j else 0)+(j*old[j] if j<len(old) else 0) for j in range(k+1)])
  moment=[sum(F(c*math.factorial(j))*u**j for j,c in enumerate(v)) for v in stirling];Z=sum(c*moment[k] for k,c in enumerate(poly));En=sum(c*moment[k+1] for k,c in enumerate(poly))/Z;N1,N2,M=En-a,En-b,r*(En+1)
  check(Z==frac(row['normalization']) and N1==frac(row['N1']) and N2==frac(row['N2']) and M==frac(row['M']),'independent Stirling-moment derivation')
  A=(N1+1)*(N2+1)-M*M;B=N1*(N2+1)-M*M;C=N2*(N1+1)-M*M;E=N1*N2-M*M;D=A+E;disc=D*D-4*B*C;lo,hi=frac(row['sqrt_lower']),frac(row['sqrt_upper']);check(lo*lo<=disc<hi*hi and hi-lo==F(1,1<<256),'integer square-root enclosure')
  for tag,v in (('A',A),('B',B),('C',C),('E',E),('D',D),('discriminant',disc)):check(frac(row[tag])==v,'exact Gaussian charge generating function')
  difference=b-a;side=B if difference>=0 else C
  def mass(t):return (2*side/(D+t))**abs(difference)/t
  lower,upper=mass(hi),mass(lo);check(lower==frac(row['charge_mass_lower']) and upper==frac(row['charge_mass_upper']),'charge coefficient enclosure');check(frac(row['support_tv_lower'])==max(F(0),1-upper) and frac(row['support_tv_upper'])==min(F(1),1-lower),'conditional support witness')
  def total(t):
   x,y=2*B/(D+t),2*C/(D+t);return (1+x/(1-x)+y/(1-y))/t
  check(total(hi)<=1<=total(lo),'two-sided charge law normalizes within enclosure')
  parity=1 if (a+b)%2==0 else -1;pp=(1+F(parity)/(2*D-1))/2;check(close(float(max(F(0),1-upper/pp)),row['parity_projected_support_tv_lower']),'plain parity projection support lower bound')
  name=f"closure_{str(row['squeezing']).replace('.','p')}_{a}_{b}.npz";z=np.load(ROOT/'results/iteration90_charge_certificate_v1'/name);v=z['v'];R=np.block([[v.real,-v.imag],[v.imag,v.real]]);X=R@z['closed_x']@R.T;nn=[(X[0,0]+X[2,2])/2,(X[1,1]+X[3,3])/2];mm=complex((X[0,1]-X[2,3])/2,(X[0,3]+X[2,1])/2);check(max(abs(nn[0]-float(N1)),abs(nn[1]-float(N2)),abs(abs(mm)-float(M)))<2e-10,'saved C++ closure moments versus exact target')
 paritybase=ROOT/'results/iteration90_parity_gates_v1';pg=read(paritybase/'summary.json');check(pg['passed'] and len(pg['rows'])==24 and len(pg['fock_comparisons'])==11,'parity inverse coverage');import mpmath as mp;mp.mp.dps=110
 for row in pg['rows']:
  z=np.load(paritybase/f"case_{row['n']}_{row['repetition']}_{row['parity']}.npz");d=len(z['source']);n=d//2;V=mp.matrix(z['source'].tolist())+mp.eye(d)/2;O=mp.matrix(d)
  for j in range(n):O[j,n+j]=1;O[n+j,j]=-1
  pi=1/mp.sqrt(mp.det(2*V));p=row['parity'];T=(V+p*pi*O*V**-1*O.T/4)/(1+p*pi)-mp.eye(d)/2;numeric=np.array(T.tolist(),float);check(np.max(abs(numeric-z['target']))<3e-11,'independent parity moments');check(np.linalg.norm(z['source']-z['recovered'])/(1+np.linalg.norm(z['source']))<3e-10,'general inverse recovers source')
 for row in pg['fock_comparisons']:
  k=row['photons'];j=k//2;old=1-F(k**k,(k+1)**(k+1));new=1-F(j**j,(j+1)**(j+1));check(old==frac(row['gaussian_full_pnr_tv']) and new==frac(row['parity_full_pnr_tv']) and old>new and old/new==frac(row['exact_error_ratio']),'exact full conditional Fock-model TV comparison');u=row['source_mean'];h=k-k%2;check(abs(2*u*u/(1+2*u)-h)<1e-11,'native parity inverse matches target mean')
 check(len(pg['boundary'])==4 and all(r['returncode']!=0 for r in pg['boundary']),'zero-probability, nonphysical and unresolved boundary states remain explicit')
 corrected=read(ROOT/'results/iteration90_parity_scope_v1/summary.json');check(corrected['metadata_correction']['numerical_results_unchanged'] and corrected['metadata_correction']['original_sha256']==H(paritybase/'summary.json'),'scoped correction binds original evidence');corrected.pop('metadata_correction')
 for old,new in zip(pg['fock_comparisons'],corrected['fock_comparisons']):check('specified two-mode squeezing preserves N1-N2' in new['scope'],'only the stated squeezing preserves charge');new['scope']=old['scope']
 check(corrected==pg,'scope correction changes no numerical result')
 probes=read(ROOT/'results/iteration90_optical_pilot_v1/probes.json');check(probes['passed'] and len(probes['analytic_rows'])==18 and all(r['statistics'][0]==1 for r in probes['analytic_rows'] if r['noise']>0) and probes['old_failure_statistics'][0]==0,'actual output constraint accepts noisy analytic states and rejects frozen failure')
 timing=ROOT/'results/iteration90_timing_v1';t=read(timing/'summary.json');check(t['passed'],'serial timing complete')
 for row in t['medians']:check(close(row['median_seconds'],statistics.median(r['seconds'] for r in t['rows'] if r['method']==row['method'] and r['records']==row['records'])),'serial timing median')
 for N in (4,64):
  for method in ('canonical','optical_cold','optical_warm'):
   for sector in (0,1):
    base=np.load(timing/f'raw_0_{method}_{N}_{sector}.npz')
    for rep in (1,2):
     other=np.load(timing/f'raw_{rep}_{method}_{N}_{sector}.npz')
     for k in base:check(np.array_equal(base[k],other[k],equal_nan=True),'formal raw repetitions')
  for sector in (0,1):
   cold=np.load(timing/f'raw_0_optical_cold_{N}_{sector}.npz');warm=np.load(timing/f'raw_0_optical_warm_{N}_{sector}.npz')
   for k in cold:check(np.array_equal(cold[k],warm[k],equal_nan=True),'cold/warm same optical method')
 final=dict(passed=True,checks=checks,full_suite=tests,fresh_streams=sum(d[4] for d in domains),fit_statistics_sum=fitstats,charge_certificates=84,parity_inverse_cases=24,conditional_full_tv_comparisons=11,scope=__doc__,limits=['No whole1152-pattern TV upper bound.','Exact rational charge bounds apply to exact analytic conditional models, not directly to the whole sampler or its rounded covariance.','Parity inverse is a building block, not a complete sampler.','PRNG domain separation is not an iid proof.']);save('validation.json',final);save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}));print(json.dumps({k:final[k] for k in ('passed','checks','fresh_streams','charge_certificates','parity_inverse_cases','conditional_full_tv_comparisons')}),flush=True)
if __name__=='__main__':main()
