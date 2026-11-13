"""Fixed 15-event nominal joint99 family for fresh 32768 PNR records.
Three eight-click marginals, two two-optical-mode PNR marginals, two photon CDFs,
two click-CDF witnesses, one experimental click CDF. Each marginal uses separate
upper and pilot-selected lower events. Domain separation does not prove iid.
Failed records are an extra target-zero outcome, never deleted or refilled.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def witness(values,mean,variance,width,delta):
 hist=np.bincount(values,minlength=width+1);eps=math.sqrt(math.log(2/delta)/(2*len(values)));best=dict(lower=0.)
 for direction,h,mu in [('increasing',hist,mean),('decreasing',hist[::-1],width-mean)]:
  integral=np.r_[0,np.cumsum(1-np.cumsum(h)/len(values))]
  for U in range(1,width+1):
   L=np.arange(U);emp=(integral[U]-integral[L])/(U-L);stop=(math.hypot(math.sqrt(variance),U-mu)-(U-mu))/2;ref=(mu-L-stop)/(U-L);gap=ref-emp-eps;j=int(np.argmax(gap))
   if gap[j]>best['lower']:best=dict(lower=float(gap[j]),direction=direction,clip_lower=int(L[j]),clip_upper=U,reference_expectation_lower=float(ref[j]),empirical_expectation=float(emp[j]))
 return dict(records=len(values),mean=float(values.mean()),reference_mean=mean,reference_variance=variance,dkw_radius=eps,**best)

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 base=ROOT/'results/iteration86_pnr_quality_v1';prior=ROOT/'results/iteration85_joint_quality_v1';optical=ROOT/'results/iteration86_optical_reference_v2';profile=ROOT/'results/iteration83_bias_profile_v1/summary.json';realpath=ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz';inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';plan=read(base/'plan.json');sample=read(base/'summary.json');N=plan['records'];pilot=plan['confidence']['pilot_records'];delta=plan['confidence']['delta_per_event'];assert sample['completed'] and N==32768 and pilot==2048
 files=[Path(__file__),base/'plan.json',base/'summary.json',base/'combined.npz',profile,realpath,inp,prior/'summary.json',prior/'plan.json',prior/'regional_photon_reference.json',*[prior/f'reference_{j}.npz' for j in range(3)],*[optical/f'pair_{j}.{suffix}' for j in range(2) for suffix in ('json','npz')]]
 hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files};assert read(prior/'plan.json')['hashes'][inp.relative_to(ROOT).as_posix()]==hashes[inp.relative_to(ROOT).as_posix()];save('plan.json',dict(scope=__doc__,hashes=hashes,confidence=plan['confidence'],reference_reuse='Frozen iteration85 references on identical Gaussian input; optical pair references newly computed.'))
 data=np.load(base/'combined.npz');counts=data['counts'];bits=data['clicks'];valid=data['valid'];assert len(valid)==N and np.array_equal(bits,counts>0);order=np.load(inp)['order'];row=dict(records=N,failed_records=int((~valid).sum()),local_click=[],optical_pnr=[],photon_regions=[],click_regions=[]);upper_radius=math.sqrt(math.log(1/delta)/(2*N));lower_radius=math.sqrt(math.log(1/delta)/(2*(N-pilot)))
 def marginal(labels,target,name,tail=0.):
  # Final label is failure; penultimate may be the declared PNR tail bin.
  labels=labels.copy();labels[~valid]=len(target);target=np.r_[target,0.];hist=np.bincount(labels,minlength=len(target));ph=np.bincount(labels[:pilot],minlength=len(target));validation=np.bincount(labels[pilot:],minlength=len(target));event=ph/pilot>target;tv=float(abs(hist/N-target).sum()/2);gap=float(validation[event].sum()/(N-pilot)-target[event].sum());np.savez_compressed(out/f'hist_{name}.npz',histogram=hist,pilot_histogram=ph,validation_histogram=validation,event=event,target=target)
  return dict(empirical_coarsened_tv=tv,nominal_joint99_marginal_tv_lower=max(0.,gap-lower_radius),nominal_joint99_marginal_tv_upper=min(1.,tv+upper_radius+tail),upper_radius=upper_radius,lower_radius=lower_radius,target_tail_upper=tail,validation_event_gap=gap,pilot_records=pilot)
 for group,ids in enumerate(plan['groups']):
  target=np.load(prior/f'reference_{group}.npz')['target'];labels=(bits[:,ids].astype(np.uint64)@np.array([1<<j for j in range(8)],np.uint64)).astype(int);row['local_click'].append(dict(group=group,**marginal(labels,target,f'click_{group}')))
 for pairindex,pair in enumerate(plan['optical_pairs']):
  reference=read(optical/f'pair_{pairindex}.json');assert reference['pair']==pair;target=np.load(optical/f'pair_{pairindex}.npz')['target'];K=len(target)-1;values=np.stack([counts[:,order//8==mode].sum(1) for mode in pair],axis=1).astype(int);inside=(values<=K).all(1);labels=np.full(N,(K+1)**2,int);labels[inside]=values[inside,0]*(K+1)+values[inside,1];tail=max(0.,float(reference['tail_mass']));assert abs(target.sum()+tail-1)<1e-12;target=np.r_[target.ravel(),tail];row['optical_pnr'].append(dict(pair=pair,maximum_photons_per_mode=K,**marginal(labels,target,f'pnr_{pairindex}',float(reference['chernoff_tail_upper']))))
 countrefs=read(prior/'regional_photon_reference.json');moments=read(profile)['reference'];eps=math.sqrt(math.log(2/delta)/(2*N))
 for j,(first,end) in enumerate(plan['regions']):
  target=np.array([float(p) for p in countrefs['probabilities'][j]]);tail=float(countrefs['tail_upper'][j]);K=len(target)-1;values=counts[:,first:end].sum(1).astype(int);values[~valid]=K+1;hist=np.bincount(np.minimum(values,K+1),minlength=K+2);distance=float(abs(np.cumsum(hist[:-1])/N-np.cumsum(target)).max());row['photon_regions'].append(dict(first=first,end=end,empirical_count_cdf_distance=distance,nominal_joint99_count_cdf_lower=max(0.,distance-eps),nominal_joint99_count_cdf_upper=min(1.,distance+eps+tail),dkw_radius=eps,reference_tail_upper=tail))
  ref=next(r for r in moments if (r['first'],r['end'])==(first,end));values=bits[:,first:end].sum(1).astype(int);values[~valid]=end-first+1;row['click_regions'].append(dict(first=first,end=end,**witness(values,ref['mean'],ref['variance'],end-first+1,delta)))
 realhist=np.load(realpath)['total_click_histogram'];realN=int(realhist.sum());realcdf=np.cumsum(realhist)/realN;values=bits.sum(1).astype(int);values[~valid]=1153;hist=np.bincount(values,minlength=1154);distance=float(abs(np.cumsum(hist)/N-np.r_[realcdf,1.]).max());radius=eps+math.sqrt(math.log(2/delta)/(2*realN));row['experimental_click_count']=dict(empirical_cdf_distance=distance,nominal_joint99_cdf_lower=max(0.,distance-radius),nominal_joint99_cdf_upper=min(1.,distance+radius),experimental_records=realN,scope='Total-click count CDF, not full-output pattern TV.')
 frugal=read(prior/'summary.json')['frugal_standalone_marginals'];comparisons=[]
 for f in frugal:
  ours=row['local_click'][f['group']];comparisons.append(dict(group=f['group'],cutoff=f['cutoff'],frugal_standalone_tv=f['numerical_tv'],our_full_trajectory_marginal_tv_upper=ours['nominal_joint99_marginal_tv_upper'],certified_error_ratio_lower=f['numerical_tv']/ours['nominal_joint99_marginal_tv_upper'],scope='Frugal standalone eight-mode task versus corresponding marginal of our full-output task; not full-output Frugal quality.'))
 save('summary.json',dict(passed=True,row=row,frugal_comparisons=comparisons,confidence=plan['confidence'],scope=__doc__,limitations=['Nominal concentration bounds assume iid; disjoint PRNG domains are not an iid theorem.','Numerical Gaussian references are not interval certificates.','No upper bound on the full1152-port photon/click pattern TV.']));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}));print(json.dumps(row,indent=2),flush=True)

if __name__=='__main__':main()
