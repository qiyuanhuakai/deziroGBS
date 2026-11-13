"""Analyze the predeclared iteration85 complete-output quality family.
Exact numerical 8-bit marginals, independent pilot-event lower bounds and
convexity/McDiarmid upper bounds. Count-CDF witnesses have separate scope.
Frugal local marginal laws are standalone 8-mode tasks, not late-prefix laws
of a Frugal full-output sampler. No inference of full TV from local upper bounds.
Failed native rows map to an explicit failure symbol with target probability zero;
never delete or refill. This analysis rule is recorded after observing B64 failures,
before computing quality bounds; fixed sample sizes and allocated event budgets stay unchanged.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,math,shutil,ctypes as C,time
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.frugal_law import full_law
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
base=ROOT/'results/iteration85_full_quality_v1';plan=read(base/'plan.json');summary=read(base/'summary.json');inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';profile=ROOT/'results/iteration83_bias_profile_v1/summary.json';cdf84=ROOT/'results/iteration84_photon_cdf_v1/reference.json';flibpath=ROOT/'results/iteration81_prefix_build_v1/prefix.dll';realpath=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz';realhist=np.load(realpath)['total_click_histogram'];realsize=int(realhist.sum());realcdf=np.cumsum(realhist)/realsize;files=[Path(__file__),base/'plan.json',base/'summary.json',inp,profile,cdf84,flibpath,realpath,*sorted(base.glob('*_combined.npz'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},confidence=plan['confidence'],supplement='One unused event allocated to shared experimental total-click CDF before quality analysis; generated full-click CDF events reused; total27 of30 events.'));delta=plan['confidence']['delta_per_event'];mp.mp.dps=80;model=np.load(inp);v,x=model['v'],model['x'];groups=plan['groups'];m,n=v.shape;runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent));lib=C.CDLL(str(flibpath));DP=C.POINTER(C.c_double);ptr=lambda x:x.ctypes.data_as(DP);ff=lib.fc81_tree;ff.argtypes=[C.c_int,C.c_int,C.c_int,C.c_int,DP,DP,DP,DP,DP,C.c_char_p,C.c_int];ff.restype=C.c_int
def projected(ids):
 u=v[ids];o=np.block([[u.real,-u.imag],[u.imag,u.real]]);return [np.eye(2*len(ids))+2*o@a@o.T for a in x]
def click_law(covs):
 k=len(covs[0])//2;vac=[]
 for mask in range(1<<k):
  ids=[j for j in range(k) if mask>>j&1];ids+= [j+k for j in ids];value=mp.mpf(1)
  for cov in covs:
   A=(cov[np.ix_(ids,ids)]+np.eye(len(ids)))/2;value*=1/mp.sqrt(mp.det(mp.matrix(A.tolist()))) if ids else 1
  vac.append(value)
 law=[]
 for mask in range(1<<k):
  zero=((1<<k)-1)^mask;sub=mask;value=mp.mpf(0)
  while True:
   value+=(-1)**sub.bit_count()*vac[zero|sub]
   if not sub:break
   sub=(sub-1)&mask
  law.append(value)
 assert min(law)>=0 and abs(sum(law)-1)<mp.mpf('1e-70')
 return law
references=[];frugal=[]
for group,ids in enumerate(groups):
 start=time.perf_counter();covs=projected(ids);q=click_law(covs);target=np.array([float(p) for p in q]);save(f'reference_{group}.json',dict(group=group,indices=ids,physical_ports=model['order'][ids].tolist(),probabilities=[str(p) for p in q],seconds=time.perf_counter()-start,scope='80-digit determinant inclusion-exclusion on float64 covariance; numerical reference, not interval certificate'));np.savez_compressed(OUT/f'reference_{group}.npz',target=target,covariances=covs);references.append(target)
 for cutoff in (2,3,5):
  laws=[]
  for s,cov in enumerate(covs):
   cov=np.ascontiguousarray(cov);q=np.zeros(255);prev=q.copy();prop=q.copy();stats=np.zeros(3);err=C.create_string_buffer(4096);rc=ff(8,cutoff,8,1,ptr(cov),ptr(q),ptr(prev),ptr(prop),ptr(stats),err,len(err));assert rc==0,err.value
   tree=[dict(prefix=''.join(str((bits>>j)&1) for j in range(length)),threshold_hex=float(q[(1<<length)-1+bits]).hex()) for length in range(8) for bits in range(1<<length)];law=np.array([float(p) for p in full_law(8,tree,model='uint64')['distribution']]);laws.append(law);save(f'frugal_{group}_{cutoff}_{s}.json',dict(tree=tree))
  combined=np.zeros(256)
  for i in range(256):
   for j in range(256):combined[i|j]+=laws[0][i]*laws[1][j]
  frugal.append(dict(group=group,cutoff=cutoff,numerical_tv=float(abs(combined-target).sum()/2)));np.savez_compressed(OUT/f'frugal_{group}_{cutoff}.npz',law=combined)
 print('reference group',group,'complete',flush=True)
def witness(values,mean,variance,width):
 hist=np.bincount(values,minlength=width+1);eps=math.sqrt(math.log(2/delta)/(2*len(values)));best=dict(lower=0.)
 for direction,h,mu in [('increasing',hist,mean),('decreasing',hist[::-1],width-mean)]:
  integral=np.r_[0,np.cumsum(1-np.cumsum(h)/len(values))]
  for U in range(1,width+1):
   L=np.arange(U);emp=(integral[U]-integral[L])/(U-L);stop=(math.hypot(math.sqrt(variance),U-mu)-(U-mu))/2;ref=(mu-L-stop)/(U-L);gap=ref-emp-eps;j=int(np.argmax(gap))
   if gap[j]>best['lower']:best=dict(lower=float(gap[j]),direction=direction,clip_lower=int(L[j]),clip_upper=U,reference_expectation_lower=float(ref[j]),empirical_expectation=float(emp[j]))
 return dict(records=len(values),mean=float(values.mean()),reference_mean=mean,reference_variance=variance,dkw_radius=eps,**best)
def regional_pmf(ids,K):
 ev=np.concatenate([np.linalg.eigvalsh((c-np.eye(len(c)))/2) for c in projected(ids)]);lam=[mp.mpf(float(a)) for a in ev];c=[a/(1+a) for a in lam];powers=[mp.mpf(1)]*len(c);traces=[];prob=[mp.fprod(1/mp.sqrt(1+a) for a in lam)]
 for j in range(1,K+1):
  powers=[a*b for a,b in zip(powers,c)];traces.append(sum(powers)/2);prob.append(sum(traces[k-1]*prob[j-k] for k in range(1,j+1))/j)
 assert min(prob)>-mp.mpf('1e-60');lo=mp.mpf(1);hi=1/max(c)
 for _ in range(150):
  radius=(lo+hi)/2;tilted=sum(a*radius/(1-a*radius) for a in c)/2
  if tilted<K+1:lo=radius
  else:hi=radius
 radius=(lo+hi)/2;tail=mp.exp(-sum(mp.log(1+a) for a in lam)/2-sum(mp.log(1-a*radius) for a in c)/2-(K+1)*mp.log(radius));assert 0<=1-sum(prob)<=tail+mp.mpf('1e-60');return prob,tail
full=read(cdf84);countrefs=[([mp.mpf(z) for z in full['probabilities']],mp.mpf(full['tail_upper']))];countrefs.append(regional_pmf(list(range(1024,1152)),256));save('regional_photon_reference.json',dict(regions=plan['regions'],probabilities=[[str(p) for p in q] for q,t in countrefs],tail_upper=[str(t) for q,t in countrefs]));moments=read(profile)['reference'];rows=[]
for panel in summary['rows']:
 name=panel['method']
 parts=[];completed=[]
 for sector in range(2):
  batches=[np.load(base/f'{name}_{sector}_{offset}.npz') for offset in range(0,panel['records'],64)]
  parts.append(np.concatenate([z['clicks'] if name.startswith('exact') else z['counts'] for z in batches]))
  completed.append(np.concatenate([(z['errors'][:,0]==0)&(z['statistics'][:,5]==1152 if name.startswith('exact') else z['statistics'][:,1]==1) for z in batches]))
 valid=completed[0]&completed[1];N=len(valid);pilot=plan['confidence']['pilot_records']
 if name.startswith('exact'):bits=np.bitwise_or(*parts);data=dict(clicks=bits)
 else:counts=parts[0].astype(np.uint64)+parts[1];bits=counts>0;data=dict(clicks=bits,counts=counts)
 row=dict(method=name,records=N,native_success=panel['success'],success=True,failed_records=int(np.count_nonzero(~valid)),empirical_failure_fraction=float(np.mean(~valid)),quality_scope='All records; failures are an explicit output symbol with target mass zero.',local_click=[],click_regions=[],photon_regions=[])
 np.savez_compressed(OUT/f'complete_{name}.npz',valid=valid,**data)
 for group,ids in enumerate(groups):
  target=np.r_[references[group],0.];labels=(bits[:,ids].astype(np.uint64)@np.array([1<<j for j in range(8)],np.uint64)).astype(int);labels[~valid]=256;hist=np.bincount(labels,minlength=257);ph=np.bincount(labels[:pilot],minlength=257);event=ph/pilot>target;validation=np.bincount(labels[pilot:],minlength=257);tv=float(abs(hist/N-target).sum()/2);upper_radius=math.sqrt(math.log(1/delta)/(2*N));lower_radius=math.sqrt(math.log(1/delta)/(2*(N-pilot)));gap=float(validation[event].sum()/(N-pilot)-target[event].sum());local=dict(group=group,empirical_tv=tv,nominal_joint99_marginal_tv_lower=max(0.,gap-lower_radius),nominal_joint99_marginal_tv_upper=min(1.,tv+upper_radius),upper_radius=upper_radius,lower_radius=lower_radius,pilot_records=pilot,validation_event_gap=gap);row['local_click'].append(local);np.savez_compressed(OUT/f'hist_{name}_{group}.npz',histogram=hist,pilot_histogram=ph,validation_histogram=validation,event=event)
 for first,end in plan['regions']:
  ref=next(a for a in moments if (a['first'],a['end'])==(first,end));values=bits[:,first:end].sum(1).astype(int);values[~valid]=end-first+1;row['click_regions'].append(dict(first=first,end=end,failure_label=end-first+1,successful_record_mean=float(values[valid].mean()) if valid.any() else None,**witness(values,ref['mean'],ref['variance'],end-first+1)))
 total_values=bits.sum(1).astype(int);total_values[~valid]=1153;hist=np.bincount(total_values,minlength=1154);ecdf=np.cumsum(hist)/N;reference=np.r_[realcdf,1.];distance=float(abs(ecdf-reference).max());radius=math.sqrt(math.log(2/delta)/(2*N))+math.sqrt(math.log(2/delta)/(2*realsize));row['experimental_click_count']=dict(empirical_cdf_distance=distance,nominal_joint99_cdf_lower=max(0.,distance-radius),nominal_joint99_cdf_upper=min(1.,distance+radius),experimental_records=realsize,scope='Total-click count CDF versus experimental population, including failure; not full bit-pattern TV upper bound.')
 if 'counts' in data:
  for (first,end),(pmf,tail) in zip(plan['regions'],countrefs):
   values=data['counts'][:,first:end].sum(1).astype(int);K=len(pmf)-1;values[~valid]=K+1;h=np.bincount(np.minimum(values,K+1),minlength=K+2);emp=np.cumsum(h[:-1])/N;ref=np.cumsum([float(p) for p in pmf]);gap=max(float(abs(emp-ref).max()),float(tail));eps=math.sqrt(math.log(2/delta)/(2*N));row['photon_regions'].append(dict(first=first,end=end,empirical_count_cdf_distance=gap,nominal_joint99_count_cdf_lower=max(0.,gap-eps-float(tail)),nominal_joint99_count_cdf_upper=min(1.,gap+eps+float(tail)),dkw_radius=eps,reference_tail_upper=float(tail)))
 rows.append(row);save(f'row_{name}.json',row);print(name,'local upper',[z['nominal_joint99_marginal_tv_upper'] for z in row['local_click']],'full click lower',max(z['lower'] for z in row['click_regions']),flush=True)
save('summary.json',dict(rows=rows,frugal_standalone_marginals=frugal,confidence=plan['confidence'],scope=__doc__,limitations=['Nominal bounds assume iid records and fixed numerical target.','Local TV upper bounds are not full-output TV upper bounds.','Standalone Frugal marginals do not give full-trajectory Frugal marginals.']));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}))
