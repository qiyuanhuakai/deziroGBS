"""Read-only independent final audit of raw86 records, domains, probabilities,
fixed confidence allocations, replay outputs and timing. No sampler DLL calls.
"""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,math,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);checks=0
 def check(value,detail):
  nonlocal checks
  checks+=1
  if not value:raise AssertionError(detail)
 def close(a,b,tol=2e-12):return bool(np.allclose(a,b,rtol=tol,atol=tol))
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 panels=['pnr_quality_v1','optical_reference_v2','joint_quality_v1','shared_gates_v4','shared_replay_diagnostic_v1','environment_probe_v1','quadmath_probe_v1','shared_replay_v2','shared_quality_v1','shared_timing_v1','timing_replay_audit_v1','prefix_law_v1','provenance_v1','tests_v1']
 for name in panels:
  folder=ROOT/'results'/f'iteration86_{name}'
  if (folder/'manifest.json').is_file():
   for path,expected in read(folder/'manifest.json')['artifacts'].items():check(H(folder/path)==expected,f'artifact {name}/{path}')
  if (folder/'plan.json').is_file():
   for path,expected in read(folder/'plan.json').get('hashes',{}).items():check(H(ROOT/path)==expected,f'input/source {path}')
 tests=read(ROOT/'results/iteration86_tests_v1/validation.json');check(tests['passed'] and tests['failures']==0 and tests['errors']==0 and tests['unchanged_sources'],'full suite');check(tests['shared86_passed']==6 and tests['optical86_passed']==4,'new tests enabled')
 source_hashes=read(ROOT/'results/iteration86_tests_v1/source_hashes.json')
 for path,expected in source_hashes.items():check(H(ROOT/path)==expected,f'tested source frozen {path}')
 base=ROOT/'results/iteration86_pnr_quality_v1';plan=read(base/'plan.json');N=plan['records'];check(N==32768 and plan['experiment']==0x86001001,'PNR fixed sample/domain');combined=np.load(base/'combined.npz');sum_counts=np.zeros(combined['counts'].shape,np.uint64);valid=np.ones(N,bool);domains=set();calls=0;fallbacks=0;consumed_domains=np.zeros(4,np.int64)
 for sector in range(2):
  for offset in range(0,N,256):
   z=np.load(base/f'raw_{sector}_{offset}.npz');r=read(base/f'row_{sector}_{offset}.json');size=len(z['counts']);check(size==256 and r['offset']==offset and r['sector']==sector,'PNR chunk identity');check(z['counts'].dtype==np.uint32 and z['statistics'].shape==(size,18),'PNR raw shapes');done=(z['errors'][:,0]==0)&(z['statistics'][:,1]==1);check(int(done.sum())==r['completed'],'PNR completion');valid[offset:offset+size]&=done;sum_counts[offset:offset+size]+=z['counts'];calls+=int(z['statistics'][:,16].sum());fallbacks+=int(z['statistics'][:,17].sum());consumed=z['statistics'][:,12:16];check(np.isfinite(consumed).all() and (consumed>=0).all() and (consumed==np.floor(consumed)).all(),'domain consumption integer');consumed_domains+=np.count_nonzero(consumed,axis=0)
   for record in range(offset,offset+size):
    for purpose in range(1,5):
     # Injective key/counter packing used by the independently KAT-tested84 code.
     address=(plan['experiment'],record,(sector<<16)|purpose);check(address not in domains,'PNR repeated random domain');domains.add(address)
 check(len(domains)==262144,'domain count');check(np.array_equal(sum_counts,combined['counts']) and np.array_equal(sum_counts>0,combined['clicks']) and np.array_equal(valid,combined['valid']),'combine both sectors before threshold');check(valid.all(),'all PNR records completed')
 qdir=ROOT/'results/iteration86_joint_quality_v1';quality=read(qdir/'summary.json');check(quality['passed'],'PNR quality analysis');row=quality['row'];delta=plan['confidence']['delta_per_event'];pilot=plan['confidence']['pilot_records'];check(plan['confidence']['events']==15 and delta==.01/15 and pilot==2048,'fixed PNR confidence family');order=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz')['order']
 for kind,items in [('click',row['local_click']),('pnr',row['optical_pnr'])]:
  for j,result in enumerate(items):
   h=np.load(qdir/f'hist_{kind}_{j}.npz');target=h['target'];hist=h['histogram'];ph=h['pilot_histogram'];vh=h['validation_histogram'];check(hist.sum()==N and ph.sum()==pilot and vh.sum()==N-pilot,'histogram sample sizes');check(np.array_equal(hist,ph+vh),'pilot/validation partition');check(close(target.sum(),1),'target mass');check(np.array_equal(h['event'],ph/pilot>target),'fixed pilot event');tv=abs(hist/N-target).sum()/2;gap=vh[h['event']].sum()/(N-pilot)-target[h['event']].sum();upper=min(1,tv+math.sqrt(math.log(1/delta)/(2*N))+result['target_tail_upper']);lower=max(0,gap-math.sqrt(math.log(1/delta)/(2*(N-pilot))));check(close(upper,result['nominal_joint99_marginal_tv_upper']) and close(lower,result['nominal_joint99_marginal_tv_lower']),'independent TV bounds')
   if kind=='click':
    ids=plan['groups'][j];labels=np.sum(combined['clicks'][:,ids].astype(np.int64)*(2**np.arange(8)),axis=1);labels[~valid]=256
   else:
    pair=plan['optical_pairs'][j];values=np.column_stack([sum_counts[:,order//8==mode].sum(1) for mode in pair]);K=64;inside=(values<=K).all(1);labels=np.full(N,65**2,int);labels[inside]=(65*values[inside,0]+values[inside,1]).astype(int);labels[~valid]=65**2+1
   check(np.array_equal(np.bincount(labels,minlength=len(hist)),hist),'raw labels independently reconstructed')
 refs=read(ROOT/'results/iteration85_joint_quality_v1/regional_photon_reference.json')
 for j,result in enumerate(row['photon_regions']):
  target=np.array([float(p) for p in refs['probabilities'][j]]);K=len(target)-1;first,end=plan['regions'][j];values=sum_counts[:,first:end].sum(1).astype(int);h=np.bincount(np.minimum(values,K+1),minlength=K+2);distance=float(abs(np.cumsum(h[:-1])/N-np.cumsum(target)).max());check(close(distance,result['empirical_count_cdf_distance']),'photon CDF from raw');check(close(distance+math.sqrt(math.log(2/delta)/(2*N))+float(refs['tail_upper'][j]),result['nominal_joint99_count_cdf_upper']),'photon confidence upper')
 optical=ROOT/'results/iteration86_optical_reference_v2'
 for j in range(2):
  z=np.load(optical/f'pair_{j}.npz');r=read(optical/f'pair_{j}.json');check(z['covariances'].shape==(2,32,32),'no optical rank truncation');check((z['target']>=0).all() and close(z['target'].sum()+float(r['tail_mass']),1),'PNR reference mass');check(abs(z['fourier128'][:65,:65]-z['fourier256'][:65,:65]).max()<2e-12,'independent Fourier grids');check(abs(z['target'][0,0]-float(r['vacuum_80digit']))<2e-12,'80-digit vacuum reference');check(float(r['chernoff_tail_upper'])<1e-12,'bounded optical PNR tail')
 gates=ROOT/'results/iteration86_shared_gates_v4';check(read(gates/'summary.json')['passed'],'new kernel gates')
 for name in ('thermal','lossy_squeezed','physical_first8'):
  target=np.load(gates/f'input_{name}.npz')['target']
  for budget in (1,8,32,64,128):
   old=np.load(gates/f'law_{name}_old_{budget}.npz')['law'];new=np.load(gates/f'law_{name}_shared_{budget}.npz')['law'];check((new>=0).all() and abs(new.sum()-1)<1e-11,'whole eight-bit mass');check(abs(new-old).sum()/2<2e-10,'same ideal finite-budget small law')
   if budget==128:check(abs(new-target).sum()/2<2e-11,'no-closure exact small law')
 replay=ROOT/'results/iteration86_shared_replay_v2';summary=read(replay/'summary.json');check(summary['passed'] and summary['thread_chunk_invariance'],'corrected full replay gates');check(len(summary['fixed_prefixes'])==15,'all old failures tested');natural=np.load(replay/'trajectories.npz');check(not natural['errors'].any() and (natural['statistics'][:,5]==1152).all(),'31 full paths complete');check(np.array_equal(natural['clicks'],natural['draws']<natural['conditional']),'stored probability decisions')
 for prefix in summary['fixed_prefixes']:
  z=np.load(replay/f"prefix_{prefix['record']}.npz");check(prefix['returncode']==0 and np.isfinite(z['conditional']).all() and ((z['conditional']>=0)&(z['conditional']<=1)).all(),'same failed prefix now has valid probabilities')
 for lo,hi in [(0,3),(3,7)]:
  z=np.load(replay/f'chunk_{lo}_{hi}.npz')
  for field in ('clicks','conditional','errors','statistics'):check(np.array_equal(z[field],natural[field][lo:hi],equal_nan=True),'thread/chunk bitwise equality')
 q=ROOT/'results/iteration86_shared_quality_v1';plan2=read(q/'plan.json');direct=read(q/'summary.json');check(plan2['records']==512 and plan2['experiment']==0x86004001 and plan2['confidence']['events']==9,'fixed direct family');parts=[];valids=[];domain2=set()
 for sector in range(2):
  arrays=[];done=[]
  for offset in range(0,512,32):
   z=np.load(q/f'raw_{sector}_{offset}.npz');arrays.append(z['clicks']);good=(z['errors'][:,0]==0)&(z['statistics'][:,5]==1152);done.append(good);check(np.isfinite(z['conditional'][good]).all() and ((z['conditional'][good]>=0)&(z['conditional'][good]<=1)).all(),'direct successful probabilities')
   for record in range(offset,offset+32):
    address=(plan2['experiment'],record,(sector<<16)|5);check(address not in domains and address not in domain2,'fresh direct domain');domain2.add(address)
  parts.append(np.concatenate(arrays));valids.append(np.concatenate(done))
 z=np.load(q/'combined.npz');check(np.array_equal(z['clicks'],np.bitwise_or(*parts)) and np.array_equal(z['valid'],valids[0]&valids[1]),'complete direct OR/failure combine');check(int((~z['valid']).sum())==direct['quality']['failed_records'],'failure inclusive direct quality');check(len(domain2)==1024,'direct domain count')
 for j,result in enumerate(direct['quality']['local_click']):
  h=np.load(q/f'hist_{j}.npz');target=np.r_[np.load(ROOT/f'results/iteration85_joint_quality_v1/reference_{j}.npz')['target'],0.];delta2=.01/9;tv=abs(h['histogram']/512-target).sum()/2;check(close(result['nominal_joint99_marginal_tv_upper'],min(1,tv+math.sqrt(math.log(1/delta2)/1024))),'direct TV upper from histogram')
 timing=read(ROOT/'results/iteration86_shared_timing_v1/summary.json');check(timing['passed'],'formal timing completed')
 for median in timing['medians']:check(close(statistics.median(r['seconds'] for r in timing['rows'] if r['method']==median['method'] and r['records']==median['records']),median['median_seconds']),'formal median')
 # Both experimental-CDF intervals independently reconstructed, with distinct families.
 real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];realN=int(real.sum());realcdf=np.r_[np.cumsum(real)/realN,1.]
 for dataset,result,delt in [(combined,row,delta),(z,direct['quality'],.01/9)]:
  size=len(dataset['valid']);values=dataset['clicks'].sum(1).astype(int);values[~dataset['valid']]=1153;emp=np.cumsum(np.bincount(values,minlength=1154))/size;dist=float(abs(emp-realcdf).max());radius=math.sqrt(math.log(2/delt)/(2*size))+math.sqrt(math.log(2/delt)/(2*realN));check(close(dist,result['experimental_click_count']['empirical_cdf_distance']),'experimental CDF difference');check(close(min(1,dist+radius),result['experimental_click_count']['nominal_joint99_cdf_upper']),'experimental CDF bound')
 for r in quality['frugal_comparisons']:check(close(r['certified_error_ratio_lower'],r['frugal_standalone_tv']/r['our_full_trajectory_marginal_tv_upper']),'scoped Frugal ratio')
 # Validate the reported clipped witness directly from full raw records.
 profile=read(ROOT/'results/iteration83_bias_profile_v1/summary.json')['reference'];direct_raw=np.load(q/'combined.npz')
 for r in direct['quality']['click_regions']:
  if r['lower']<=0:continue
  first,end=r['first'],r['end'];width=end-first+1;values=direct_raw['clicks'][:,first:end].sum(1).astype(int);values[~direct_raw['valid']]=width;ref=next(a for a in profile if (a['first'],a['end'])==(first,end));mu=ref['mean']
  if r['direction']=='decreasing':values=width-values;mu=width-mu
  L,U=r['clip_lower'],r['clip_upper'];emp=float(np.clip((values-L)/(U-L),0,1).mean());stop=(math.sqrt(ref['variance']+(U-mu)**2)-(U-mu))/2;expected=(mu-L-stop)/(U-L);lower=expected-emp-math.sqrt(math.log(2/(.01/9))/(2*len(values)));check(close(lower,r['lower']) and close(emp,r['empirical_expectation']),'independent clipped full-law witness')
 # OR convolution by subset zeta/Mobius transforms, independently of the driver.
 prefix=ROOT/'results/iteration86_prefix_law_v1';ps=read(prefix/'summary.json');check(ps['passed'],'actual full-model prefix enumeration')
 def zeta(values,inverse=False):
  values=values.copy()
  for j in range(8):
   for mask in range(256):
    if mask&(1<<j):values[mask]+=(-1 if inverse else 1)*values[mask^(1<<j)]
  return values
 for method in ('old','shared'):
  left=np.load(prefix/f'{method}_0.npz')['law'];right=np.load(prefix/f'{method}_1.npz')['law'];full=np.load(prefix/f'{method}_combined.npz');expected=zeta(zeta(left)*zeta(right),True);check(abs(expected-full['law']).max()<2e-14,'independent two-spectrum OR convolution');tv=float(abs(full['law']-full['target']).sum()/2);reported=next(r['numerical_tv'] for r in ps['rows'] if r['method']==method);check(abs(tv-reported)<1e-18,'same first-prefix precision metric')
 for r in timing['repetitions']:
  if r['method']=='shared':check(r['identical_repetitions'],'corrected kernel repeat determinism')
 report=dict(passed=True,checks=checks,full_suite=tests,pnr_domains=len(domains),direct_domains=len(domain2),pnr_consumed_domains_per_purpose=consumed_domains.tolist(),pnr_tree_calls=calls,pnr_tree_fallbacks=fallbacks,scope=__doc__,limitations=['Many checks are per-domain address assertions, not independent experiments.','Nominal99 families are separate; combined guarantee at least98%.','Numerical references are not interval proofs and marginal TV is not full-law TV.']);save('validation.json',report);save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}));print(json.dumps({k:report[k] for k in ['passed','checks','pnr_domains','direct_domains','pnr_tree_fallbacks']}),flush=True)

if __name__=='__main__':main()
