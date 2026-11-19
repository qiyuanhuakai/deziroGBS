"""Independent verification of lossy full-output experiments and global witnesses.
No native library loaded. Reconstruct optics, full pair moments, clipped-count
bounds and PNR moments independently; keep incomplete/failed build evidence.
"""
from pathlib import Path
import os,sys,json,hashlib,argparse,math,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.frugal_law import full_law
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
checks=[]
def check(ok,message):
 assert ok,message
 checks.append(message)
for folder in sorted((ROOT/'results').glob('iteration83*')):
 if not folder.is_dir() or folder==OUT:continue
 if (folder/'manifest.json').exists():
  for name,digest in read(folder/'manifest.json').get('artifacts',{}).items():check(H(folder/name)==digest,'frozen artifact '+folder.name+'/'+name)
 if (folder/'plan.json').exists():
  for name,digest in read(folder/'plan.json').get('hashes',{}).items():check(H(ROOT/name)==digest,'executed source/input '+name)
tests=read(ROOT/'results/iteration83_tests_v1/validation.json');check(tests['passed'] and tests['loss83_passed']==8 and tests['core83_passed']==6,'new loss/core tests and complete suite')
for name,digest in read(ROOT/'results/iteration83_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
check(read(ROOT/'results/iteration83_frugal_build_v1/build.json')['returncode']!=0,'constexpr failure retained')
check(read(ROOT/'results/iteration83_frugal_build_v2/build.json')['returncode']==0 and (ROOT/'results/iteration83_frugal_build_v2/aborted_audit.json').exists(),'successful full build and explicitly stopped redundant audit')
data=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz');T,V,x0,x=data['T'],data['v'],data['input_x'],data['x'];m,n=V.shape;check((m,n)==(1152,50),'full physical dimensions');check(np.max(abs(V.conj().T@V-np.eye(n)))<1e-12,'isometry')
rr=data['squeezing'];covs=[]
for s in range(2):
 # Direct complex optical N/M moments, independent of QR and realification.
 nn=(T*np.repeat(np.sinh(rr[:,s])**2,2))@T.conj().T;mm=(T*np.repeat(np.sinh(rr[:,s])*np.cosh(rr[:,s]),2))@T[:,np.arange(n)^1].T
 cov=np.block([[np.eye(m)+2*(nn+mm).real,2*(mm-nn).imag],[2*(mm+nn).imag,np.eye(m)+2*(nn-mm).real]]);cov=(cov+cov.T)/2;stored=np.load(ROOT/f'results/iteration83_full_quality_v1/covariance_{s}.npz')['cov'];check(np.max(abs(cov-stored))<5e-14,'full direct optical covariance '+str(s));covs.append(cov)
ids=np.arange(m);vac=[]
for c in covs:vac.append(2/np.sqrt((c[ids,ids]+1)*(c[m+ids,m+ids]+1)-c[ids,m+ids]**2))
v0=vac[0]*vac[1];mean=float((1-v0).sum());variance=float(np.dot(v0,1-v0));ii,jj=np.triu_indices(m,1);cross=np.longdouble(0);profile=read(ROOT/'results/iteration83_bias_profile_v1/summary.json');regions=[(r['first'],r['end']) for r in profile['reference']];regional_cross=np.zeros(len(regions),np.longdouble)
for first in range(0,len(ii),8192):
 i,j=ii[first:first+8192],jj[first:first+8192];ix=np.stack([i,j,m+i,m+j],axis=1);both=np.ones(len(i),np.longdouble)
 for c in covs:
  q=(c[ix[:,:,None],ix[:,None,:]]+np.eye(4))/2;sign,logdet=np.linalg.slogdet(q);check(np.all(sign>0),'independent two-port determinant sign');both*=np.exp(-logdet.astype(np.longdouble)/2)
 pair_cross=2*(both-v0[i].astype(np.longdouble)*v0[j]);cross+=np.sum(pair_cross)
 for k,(first,end) in enumerate(regions):regional_cross[k]+=pair_cross[(i>=first)&(j<end)].sum()
variance+=float(cross);ref=read(ROOT/'results/iteration83_global_quality_v1/reference_moments.json');check(abs(mean-ref['mean'])<1e-10 and abs(variance-ref['variance'])<1e-8,'all 662976 pair moments independent reconstruction');save('independent_global_moments.json',dict(mean=mean,variance=variance,native_mean=ref['mean'],native_variance=ref['variance']))
mp.mp.dps=80;vac80=[]
for s,c in enumerate(covs):
 values=[]
 for mask in range(256):
  ix=[j for j in range(8) if mask>>j&1];ix+=[m+j for j in ix]
  if not ix:values.append(mp.mpf(1));continue
  q=mp.matrix([[(mp.mpf(float(c[i,j]))+int(i==j))/2 for j in ix] for i in ix]);l=mp.cholesky(q);values.append(1/mp.fprod(l[j,j] for j in range(len(ix))))
 vac80.append(values)
target=[vac80[0][i^255]*vac80[1][i^255] for i in range(256)]
for bit in range(8):
 for mask in range(256):
  if mask>>bit&1:target[mask]-=target[mask^(1<<bit)]
target=np.array([float(p) for p in target]);quality=read(ROOT/'results/iteration83_full_quality_v1/summary.json');quality80=[]
for row in quality['rows']:
 if not row['success']:continue
 law=np.load(ROOT/f"results/iteration83_full_quality_v1/{row['method']}_{row['budget']}.npz")['law'];tv=float(abs(law-target).sum()/2);check(abs(tv-row['numerical_tv'])<1e-12,'80-digit prefix TV '+str((row['method'],row['budget'])));quality80.append(dict(method=row['method'],budget=row['budget'],tv=tv))
for s in range(2):
 tree=read(ROOT/f'results/iteration83_frugal_prefix_v1/tree_{s}.json')['tree'];law=np.array([float(p) for p in full_law(8,tree,model='uint64')['distribution']]);old=np.load(ROOT/f'results/iteration83_full_quality_v1/frugal_2_{s}.npz')['law'];check(np.max(abs(law-old))<1e-15,'full target/subsystem Frugal law '+str(s))
save('prefix_quality_80digit.json',quality80)
globalq=read(ROOT/'results/iteration83_global_quality_v1/summary.json');bounds=read(ROOT/'results/iteration83_global_witness_v2/summary.json')['rows'];boundresults=[]
for row in globalq['rows']:
 b=row['budget'];check(row['success'],'4096 complete successful records B'+str(b));parts=[]
 for s in range(2):
  p=np.load(ROOT/f'results/iteration83_global_quality_v1/sector_{b}_{s}.npz');check(not p['errors'][:,0].any(),'all GPU paths succeeded');parts.append(p['clicks'])
 d=np.load(ROOT/f'results/iteration83_global_quality_v1/combined_{b}.npz');check(np.array_equal(np.bitwise_or(*parts),d['clicks']),'two full sectors OR');totals=d['clicks'].sum(1);check(len(totals)==4096 and d['clicks'].shape[1]==1152,'full output shape');check(np.array_equal(np.bincount(totals.astype(int),minlength=1153),d['total_histogram']),'global click histogram');check(abs(totals.mean()-row['mean_clicks'])<1e-12,'mean count')
 bound=next(q for q in bounds if q['budget']==b);L,U=bound['clip_lower'],bound['clip_upper'];emp=float(np.clip((totals.astype(float)-L)/(U-L),0,1).mean());stop=(math.hypot(math.sqrt(variance),U-mean)-(U-mean))/2;lower=(mean-L-stop)/(U-L)-emp-math.sqrt(math.log(400)/(2*len(totals)));check(abs(lower-bound['nominal_joint99_full_tv_lower_bound'])<1e-9,'independent clipped-mean witness');check(lower>.1,'full-law error lower bound exceeds ten percent');boundresults.append(dict(budget=b,independently_recomputed_lower=lower))
save('global_witness_checks.json',boundresults)
regionalchecks=[]
for k,(first,end) in enumerate(regions):
 singles=v0[first:end];mu=float((1-singles).sum());var=float(np.dot(singles,1-singles)+regional_cross[k]);native=profile['reference'][k];check(abs(mu-native['mean'])<1e-10 and abs(var-native['variance'])<1e-8,'independent region moments '+str((first,end)))
 for b in (4,8):
  row=next(r for r in profile['rows'] if (r['budget'],r['first'],r['end'])==(b,first,end));bits=np.load(ROOT/f'results/iteration83_global_quality_v1/combined_{b}.npz')['clicks'];counts=bits[:,first:end].sum(1).astype(int);hist=np.load(ROOT/f'results/iteration83_bias_profile_v1/hist_{b}_{first}_{end}.npz')['histogram'];check(np.array_equal(hist,np.bincount(counts,minlength=end-first+1)),'regional count histogram '+str((b,first,end)));check(abs(float(counts.mean())-row['empirical_mean'])<1e-12,'regional empirical mean');w=row['witness'];eps=math.sqrt(math.log(4*len(regions)/.01)/(2*len(bits)));check(abs(eps-row['dkw_radius'])<1e-15,'joint region confidence radius')
  if w['direction'] is not None:
   samples=counts if w['direction']=='increasing' else end-first-counts;reference_mean=mu if w['direction']=='increasing' else end-first-mu;L,U=w['clip_lower'],w['clip_upper'];emp=float(np.clip((samples-L)/(U-L),0,1).mean());stop=(math.hypot(math.sqrt(var),U-reference_mean)-(U-reference_mean))/2;lower=(reference_mean-L-stop)/(U-L)-emp-eps;check(abs(lower-row['nominal_joint99_full_bit_tv_lower_bound'])<1e-9,'independent regional TV witness '+str((b,first,end)));regionalchecks.append(dict(budget=b,first=first,end=end,nominal_joint99_tv_lower=lower))
  else:check(row['nominal_joint99_full_bit_tv_lower_bound']==0,'zero regional witness retained')
save('regional_witness_checks.json',regionalchecks)
# Direct experimental comparison uses no Gaussian reference model.
realh=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];N=int(realh.sum());check(N==1492642,'experimental record count');realcdf=np.cumsum(realh,dtype=np.int64)/N;experimental=[]
for row in read(ROOT/'results/iteration83_experiment_gap_v1/summary.json')['rows']:
 b=row['budget'];h=np.load(ROOT/f'results/iteration83_global_quality_v1/combined_{b}.npz')['total_histogram'];count=int(h.sum());cdf=np.cumsum(h,dtype=np.int64)/count;delta=abs(cdf-realcdf);threshold=int(np.argmax(delta));bound=float(delta[threshold])-math.sqrt(math.log(600)/(2*count))-math.sqrt(math.log(600)/(2*N));check(threshold==row['threshold'] and abs(bound-row['nominal_joint99_full_bit_tv_lower_bound'])<1e-14,'independent empirical CDF bound '+str(b));check(abs(.5*np.abs(h/count-realh/N).sum()-row['empirical_count_tv'])<1e-14,'empirical count TV '+str(b));experimental.append(dict(budget=b,threshold=threshold,nominal_joint99_lower=bound))
save('experimental_gap_checks.json',experimental)
# Passive pure-core decomposition, primal feasibility and independently evaluated
# PSD dual bound. This certifies neither a global passive-basis optimum nor any
# implemented sampler. All checks here are floating-point numerical checks.
corechecks=[];omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]])
for folder in ('iteration83_core_energy_probe_v2','iteration83_core_energy_native_v1'):
 summary=read(ROOT/f'results/{folder}/summary.json')
 for row in summary['rows']:
  s=row['sector'];d=np.load(ROOT/f'results/{folder}/split_{s}.npz');R,t,P,noise,Y=d['rotation'],d['t'],d['pure'],d['noise'],d['dual'];state=np.eye(2*n)+2*x[s];W=R.T@state@R;diag=np.r_[(1-t)/(1+t),(1+t)/(1-t)];check(np.max(abs(R.T@R-np.eye(2*n)))<1e-12 and np.max(abs(R@omega@R.T-omega))<1e-12,'passive core basis '+str((folder,s)));check(np.max(abs(P@omega@P-omega))<1e-11,'pure core symplectic condition '+str((folder,s)));check(np.max(abs(P+2*noise-state))<1e-11,'core covariance preserved '+str((folder,s)));check(np.linalg.eigvalsh(noise).min()>0 and np.linalg.eigvalsh(Y).min()>0,'primal/dual feasibility '+str((folder,s)));check(np.max(abs((R*diag)@R.T-P))<1e-12,'squeezing reconstructs core '+str((folder,s)));energy=(diag.sum()-2*n)/4;lower=(2*np.sqrt((1+Y.diagonal()[:n])*(1+Y.diagonal()[n:])).sum()-np.trace(Y@W)-2*n)/4;check(abs(energy-row['optimized_core_mean'])<1e-11 and -1e-9<energy-lower<1e-5,'independent fixed-basis dual gap '+str((folder,s)));corechecks.append(dict(source=folder,sector=s,core_energy=energy,dual_lower=lower,gap=energy-lower))
save('core_primal_dual_checks.json',corechecks)
# Independent single-mode pair-number convolutions, using positive terms only.
# This avoids the cancellation that invalidated the retained v1 budget run.
def positive_core_tail(t):
 z=[mp.mpf(float(q))**2 for q in t];coeff=[mp.mpf(1)]+[mp.mpf(0)]*256
 for q in z:
  mode=[mp.sqrt(1-q)]
  for j in range(1,257):mode.append(mode[-1]*q*(2*j-1)/(2*j))
  coeff=[mp.fsum(coeff[k]*mode[j-k] for k in range(j+1)) for j in range(257)]
 radius=1/mp.sqrt(max(z));remainder=mp.fprod(mp.sqrt((1-q)/(1-q*radius)) for q in z)/radius**257
 return coeff,remainder
tailchecks=[]
for folder in ('iteration83_core_budget_v2','iteration83_core_energy_probe_v2','iteration83_core_global_probe_v2'):
 for row in read(ROOT/f'results/{folder}/summary.json')['rows']:
  s=row['sector'];t=np.load(ROOT/f'results/{folder}/split_{s}.npz')['t'];coeff,remainder=positive_core_tail(t)
  for b in row['budgets']:
   K=b.get('budget',b.get('core_photon_budget'));value=mp.sqrt(min(mp.mpf(1),mp.fsum(coeff[K//2+1:])+remainder));stored=mp.mpf(b.get('ideal_tv_upper',b.get('ideal_click_and_pnr_tv_upper',b.get('conditional_ideal_tv_upper'))));check(abs(value/stored-1)<mp.mpf('1e-65'),'positive convolution core budget '+str((folder,s,K)))
   if K==32:tailchecks.append(dict(source=folder,sector=s,budget=K,conditional_ideal_tv_upper=str(value)))
save('core_tail_checks.json',tailchecks)
resources=read(ROOT/'results/iteration83_core_resources_v1/summary.json');optimized=read(ROOT/'results/iteration83_core_energy_probe_v2/summary.json')
for row in resources['rows']:
 s=row['sector'];t=np.load(ROOT/f'results/iteration83_core_energy_probe_v2/split_{s}.npz')['t'];z=sorted([mp.mpf(float(a))**2 for a in t],reverse=True)
 for rank in row['rank_projection']:
  k=rank['retained_squeezed_modes'];prob=mp.exp(mp.fsum(mp.log1p(-a)/2 for a in z[k:]));distance=mp.sqrt(-mp.expm1(mp.log(prob)));check(abs(distance-mp.mpf(rank['pure_core_projection_trace_distance']))<mp.mpf('1e-70'),'independent rank projection '+str((s,k)))
 for b in row['dense_resources']:
  K=b['photon_cutoff'];check(b['all_number_states']==math.comb(n+K,K) and b['one_complex128_dense_vector_bytes']==16*math.comb(n+K,K),'dense core resource '+str((s,K)))
for row in resources['joint_budgets']:
 errors=[mp.mpf(next(b['ideal_tv_upper'] for b in optimized['rows'][s]['budgets'] if b['budget']==row['core_cutoffs'][s])) for s in range(2)];check(abs(mp.sqrt(1-(1-errors[0]**2)*(1-errors[1]**2))-mp.mpf(row['conditional_ideal_joint_pnr_and_click_tv_upper']))<mp.mpf('1e-65'),'joint pure-core fidelity budget '+str(row['core_cutoffs']))
# Independent invariant-subspace spectrum, without the Gram square root or
# Cholesky used by the experiment. Omega B = B Omega for a complex-linear map.
from gbslab.mps_probe import geometric_top_weights
schmidt=read(ROOT/'results/iteration83_core_schmidt_v1/summary.json')
for row in schmidt['rows']:
 s=row['sector'];P=np.load(ROOT/f'results/iteration83_core_energy_probe_v2/split_{s}.npz')['pure'];g=V[:576].conj().T@V[:576];G=np.block([[g.real,-g.imag],[g.imag,g.real]]);ev=np.linalg.eigvals(omega@(np.eye(2*n)+(P-np.eye(2*n))@G));check(np.max(abs(ev.real))<1e-11,'Schmidt invariant-subspace eigenvalues imaginary '+str(s));nu=np.sort(ev.imag)[n:];d=np.load(ROOT/f'results/iteration83_core_schmidt_v1/middle_{s}.npz');check(np.max(abs(nu-d['nu']))<1e-11,'independent low-rank Schmidt spectrum '+str(s));weights=geometric_top_weights(d['q'],4096);check(np.max(abs(weights-d['top_weights']))<1e-14,'independent reversed-tree Schmidt weights '+str(s));middle=next(q for q in row['cuts'] if q['cut']==576)
 for r in middle['rank_tradeoffs']:
  distance=math.sqrt(1-math.fsum(weights[:r['rank']]));check(abs(distance-r['necessary_pure_state_trace_distance'])<1e-12,'pure-core rank tradeoff '+str((s,r['rank'])))
# All-basis dual reference uses an independent nonsymmetric symplectic spectrum.
globalcore=[]
for s in range(2):
 failed=read(ROOT/f'results/iteration83_core_global_probe_v1/row_{s}.json');check(not failed['candidate_numerically_feasible'] and failed['diagnostics']['minimum_remainder_eigenvalue']<0,'unguarded global candidate rejected '+str(s));d=np.load(ROOT/f'results/iteration83_core_global_probe_v2/split_{s}.npz');P,S,L=d['pure'],d['S'],d['L'];state=np.eye(2*n)+2*x[s];row=read(ROOT/f'results/iteration83_core_global_probe_v2/row_{s}.json');check(np.max(abs(S-L@L.T))<1e-12,'PSD multiplier factor '+str(s));ev=np.linalg.eigvals(omega@(np.eye(2*n)+S));check(np.max(abs(ev.real))<1e-10,'global weighted symplectic spectrum imaginary');phi=float(abs(ev.imag).sum());lower=(phi-np.trace(S@state)-2*n)/4;upper=(np.trace(P)-2*n)/4;check(abs(lower-row['diagnostics']['dual_photon_lower_estimate'])<1e-10 and abs(upper-row['diagnostics']['pure_mean_photons'])<1e-12,'independent full-cone dual '+str(s));check(np.linalg.eigvalsh(state-P).min()>0 and np.max(abs(P@omega@P-omega))<1e-11,'global primal feasibility '+str(s));check(np.max(abs(P+2*d['noise_factor']@d['noise_factor'].T-state))<1e-12,'global final-state reconstruction '+str(s));check(np.max(abs(np.linalg.svd(d['pairing'],compute_uv=False)-d['t']))<1e-12 and d['t'].max()<1,'global pairing singular values '+str(s));g=V[:576].conj().T@V[:576];G=np.block([[g.real,-g.imag],[g.imag,g.real]]);nu=np.sort(np.linalg.eigvals(omega@(np.eye(2*n)+(P-np.eye(2*n))@G)).imag)[n:];check(np.max(abs(nu-d['middle_nu']))<1e-11,'global core independent Schmidt spectrum');distance=math.sqrt(1-math.fsum(d['middle_weights']));check(abs(distance-row['middle_rank4096_pure_state_distance_lower'])<1e-12,'global core rank4096 limitation');globalcore.append(dict(sector=s,numerical_all_basis_lower=lower,feasible_upper=upper,gap=upper-lower))
save('global_core_checks.json',globalcore)
# New weight arithmetic and actual complete failed-seed replays.
weightbase=ROOT/'results/iteration83_weights_pilot_v1';arithmetic=read(weightbase/'arithmetic_checks.json')['checks'];check(len(arithmetic)==172,'all scalar weight checks retained')
with mp.workdps(100):
 for r in arithmetic:
  ah,al,bh,bl=map(mp.mpf,r['inputs']);a,b=ah+al,bh+bl;expected=(lambda:a+b,lambda:a-b,lambda:a*b,lambda:a/b)[r['operation']]();actual=mp.mpf(r['high'])+mp.mpf(r['low']);check(abs(actual-expected)/max(abs(expected),mp.mpf('1e-100'))<mp.mpf('1e-62'),'independent Weight83 arithmetic '+str((r['case'],r['operation'])))
weightquality=[]
for r in read(weightbase/'prefix_summary.json')['rows']:
 check(r['success'],'weight prefix complete B'+str(r['budget']));law=np.load(weightbase/f"prefix_combined_{r['budget']}.npz")['law'];tv=float(abs(law-target).sum()/2);check(abs(tv-r['first_eight_tv'])<1e-12,'independent 80-digit weight prefix TV');weightquality.append(dict(budget=r['budget'],tv=tv,scope='first eight outputs only'))
save('weight_prefix_80digit.json',weightquality)
for row in read(weightbase/'summary.json')['rows']:
 check(row['success'],'known-seed weight batch succeeds '+str((row['rep'],row['budget'])));parts=[]
 for s in range(2):
  d=np.load(weightbase/f"sector_{row['rep']}_{row['budget']}_{s}.npz");check(not d['errors'][:,0].any() and np.all(d['statistics'][:,5]==1152),'weight full rows completed');parts.append(d['clicks'])
 combined=np.load(weightbase/f"combined_{row['rep']}_{row['budget']}.npz")['clicks'];check(np.array_equal(combined,np.bitwise_or(*parts)) and abs(combined.sum(1).mean()-row['mean_clicks'])<1e-12,'weight complete record OR')
replaybase=ROOT/'results/iteration83_full_replay_v1'
for row in read(replaybase/'summary.json')['rows']:
 d=np.load(replaybase/f"replay_{row['budget']}_{row['sector']}.npz");check(not d['errors'][:,0].any() and np.array_equal(d['cpu'],d['gpu']),'same-random complete CPU/GPU replay');difference=float(np.max(abs(d['cpu_conditional']-d['gpu_conditional'])));check(abs(difference-row['maximum_conditional_difference_through_first_divergence'])<1e-15,'replay conditional difference recomputed')
pnrbase=ROOT/'results/iteration83_loss_pnr_pilot_v1'
for row in read(pnrbase/'summary.json')['rows']:
 check(row['success'],'mixed PNR pilot completed');r=row['retained'];parts=[np.load(pnrbase/f'sector_{r}_{s}.npz')['counts'] for s in range(2)];d=np.load(pnrbase/f'combined_{r}.npz');check(np.array_equal(d['counts'],parts[0].astype(np.uint64)+parts[1]) and np.array_equal(d['clicks'],d['counts']>0),'integer photon addition before threshold');check(abs(d['counts'].sum(1).mean()-row['mean_total_photons'])<1e-12,'PNR pilot mean')
orderbase=ROOT/'results/iteration83_order_pilot_v1';order=np.load(orderbase/'order.npz')['order'];check(np.array_equal(np.sort(order),np.arange(1152)) and np.array_equal(order[:8],np.arange(8)),'input-only order permutation and pinned prefix')
for row in read(orderbase/'summary.json')['rows']:
 check(row['success'],'reordered full batch succeeded');b=row['budget'];parts=[]
 for s in range(2):
  d=np.load(orderbase/f'sector_{b}_{s}.npz');check(not d['errors'][:,0].any(),'reordered failed rows absent');parts.append(d['clicks'])
 d=np.load(orderbase/f'combined_{b}.npz');check(np.array_equal(d['clicks'][:,order],np.bitwise_or(*parts)),'reordered outputs restored to base physical labels');check(np.array_equal(np.bincount(d['clicks'].sum(1).astype(int),minlength=1153),d['total_histogram']),'reordered count histogram')
# Fresh independent quality batches: no failed-row removal or refills.
followbase=ROOT/'results/iteration83_full_followup_quality_v1'
for row in read(followbase/'summary.json')['rows']:
 method,count=row['method'],row['records'];check(row['success'],'fresh quality batch completed '+method);parts=[]
 for s in range(2):
  d=np.load(followbase/f'{method}_{s}.npz');meta=read(followbase/f'{method}_{s}.json');check(meta['returncode']==0 and meta['completed_rows']==count and meta['success'],'fresh native completion '+str((method,s)))
  if method=='weights32':
   check(not d['errors'][:,0].any() and np.all(d['statistics'][:,5]==1152),'fresh weight complete paths');parts.append(d['clicks'])
  else:
   check(d['counts'].dtype==np.uint32 and d['statistics'][1]==count and d['counts'].sum()==d['statistics'][0],'fresh integer PNR completion and total');parts.append(d['counts'])
 combined=np.load(followbase/f'{method}_combined.npz');bits=combined['clicks'];check(bits.shape==(count,1152),'fresh full bit shape')
 if method=='weights32':check(np.array_equal(bits,np.bitwise_or(*parts)),'fresh weight sector OR')
 else:
  counts=parts[0].astype(np.uint64)+parts[1];check(np.array_equal(counts,combined['counts']) and np.array_equal(bits,counts>0),'fresh PNR integer sum then threshold');check(np.array_equal(np.bincount(counts.sum(1).astype(int)),combined['photon_histogram']),'fresh photon histogram');check(abs(counts.sum(1).mean()-row['mean_total_photons'])<1e-12,'fresh photon mean')
 check(np.array_equal(np.bincount(bits.sum(1).astype(int),minlength=1153),combined['total_histogram']) and abs(bits.sum(1).mean()-row['mean_clicks'])<1e-12,'fresh click mean and histogram')
# Independent total photon moments from real Gaussian covariance invariants.
photonmean=0.;photonvariance=0.
for s in range(2):
 state=np.eye(2*n)+2*x[s];photonmean+=(np.trace(state)-2*n)/4;photonvariance+=(np.sum(state*state.T)-2*n)/8
photonref=read(ROOT/'results/iteration83_followup_witness_v1/reference_photons.json');check(abs(photonmean-photonref['mean'])<1e-10 and abs(photonvariance-photonref['variance'])<1e-9,'independent total photon invariants')
def check_new_witness(samples,mu,var,row,maximum=None):
 eps=math.sqrt(math.log(2000)/(2*len(samples)));check(abs(eps-row['dkw_radius'])<1e-15,'ten-event family radius');check(abs(mu-row['reference_mean'])<1e-9 and abs(var-row['reference_variance'])<1e-8,'fresh reference moments');check(abs(samples.mean()-row['empirical_mean'])<1e-12 and abs(samples.var(ddof=1)-row['empirical_variance'])<1e-9,'fresh empirical moments')
 if row['direction'] is None:check(row['nominal_joint99_full_tv_lower']==0,'zero witness is inconclusive');return 0.
 if row['direction']=='decreasing':check(maximum is not None,'decreasing witness has bounded count');samples=maximum-samples;mu=maximum-mu
 L,U=row['clip_lower'],row['clip_upper'];emp=float(np.clip((samples.astype(float)-L)/(U-L),0,1).mean());stop=(math.hypot(math.sqrt(var),U-mu)-(U-mu))/2;ref_lower=(mu-L-stop)/(U-L);lower=max(0.,ref_lower-emp-eps);check(abs(emp-row['empirical_expectation'])<1e-13 and abs(ref_lower-row['reference_expectation_lower'])<1e-9 and abs(lower-row['nominal_joint99_full_tv_lower'])<1e-9,'independent fresh clipped witness');return lower
followchecks=[]
for row in read(ROOT/'results/iteration83_followup_witness_v1/summary.json')['rows']:
 method=row['method'];check(row['success'],'fresh witness success');d=np.load(orderbase/f'combined_{method[-1]}.npz' if method.startswith('order') else followbase/f'{method}_combined.npz');bits=d['clicks'];lowers=[]
 for r in row['click_regions']:
  first,end=r['first'],r['end'];k=regions.index((first,end));single=v0[first:end];mu=float((1-single).sum());var=float(np.dot(single,1-single)+regional_cross[k]);lowers.append(check_new_witness(bits[:,first:end].sum(1).astype(int),mu,var,r,end-first))
 h=d['total_histogram'];cdf=np.cumsum(h,dtype=np.int64)/len(bits);ks=float(np.max(abs(cdf-realcdf)));bound=max(0.,ks-math.sqrt(math.log(2000)/(2*len(bits)))-math.sqrt(math.log(2000)/(2*N)));r=row['experimental_comparison'];check(abs(ks-r['empirical_ks'])<1e-14 and abs(bound-r['nominal_joint99_full_bit_tv_lower'])<1e-14 and int(np.argmax(abs(cdf-realcdf)))==r['threshold'],'fresh direct experimental CDF bound');out=dict(method=method,records=len(bits),nominal_joint99_full_click_tv_lower=max(lowers),nominal_joint99_experimental_bit_tv_lower=bound)
 if method=='pnr3':out['nominal_joint99_full_pnr_tv_lower']=check_new_witness(d['counts'].sum(1).astype(int),photonmean,photonvariance,row['photons'])
 followchecks.append(out)
save('followup_quality_checks.json',dict(reference_photon_mean=photonmean,reference_photon_variance=photonvariance,rows=followchecks,scope='Separate ten-event joint99 family; iid datasets and numerical Gaussian references.'))
# Independent formal-power-series recurrence for the ideal PNR pair moments.
pnr=read(ROOT/'results/iteration83_pnr_moments_v1/summary.json');base=ROOT/'results/iteration81_factor_pilot_v2'
for label in ('strong4','wide2','large128'):
 inp=np.load(base/f'input_{label}.npz');w,k,v=inp['weights'],inp['K'],inp['v'];M,N=v.shape;L=len(w);depth=min(8,M);parts=[np.load(base/f'factors_{label}_{i}.npz') for i in range(0,L,256)];aa=np.concatenate([x['left'] for x in parts]);bb=np.concatenate([x['right'] for x in parts]);forward=aa*(bb@k.T);backward=bb*(aa@k);normal=forward.sum(1);f=(forward[:,:depth]+backward[:,:depth])/normal[:,None];g=aa[:,:depth]*bb[:,:depth]*np.diag(k)[:depth]/normal[:,None]
 for r in (.3,.7,1.1):
  z=float(np.tanh(r))**2;c=z/(1-z);h=N/2;a=c*f;b=c*g;coef=[np.ones_like(a),h*a]
  for q in range(1,8):coef.append(((h+q)*a*coef[q]+(2*h+q-1)*b*coef[q-1])/(q+1))
  expected=np.array([np.dot(w/w.sum(),coef[q])*math.factorial(q) for q in range(1,9)]).T;stored=np.load(ROOT/f'results/iteration83_pnr_moments_v1/moments_{label}_{r}.npz');check(np.max(abs(expected/stored['factor']-1))<1e-11,'independent PNR generating recurrence '+str((label,r)))
 print('independent PNR moments',label,flush=True)
timing=read(ROOT/'results/iteration83_full_timing_v1/summary.json');check(timing['completed'],'full timing completed');paired=[]
for key in sorted(set((r['method'],r['budget'],r['records']) for r in timing['rows'])):
 rows=[r for r in timing['rows'] if (r['method'],r['budget'],r['records'])==key];check(len(rows)==(1 if key[0]=='frugal' else 3),'declared timing replication');q=next(q for q in quality['rows'] if (q['method'],q['budget'])==key[:2]);failed=[r for r in rows if not r['success']]
 if failed:
  check(key==('stable',32,4) and len(failed)==2 and all('normalization lost' in r['error'] for r in failed),'full B32 failures retained');paired.append(dict(method=key[0],budget=key[1],records=key[2],observations=len(rows),failed_panels=len(failed),seconds=None,prefix_tv=q['numerical_tv'],scope='prefix law only; no valid complete-batch speed assigned'))
 else:paired.append(dict(method=key[0],budget=key[1],records=key[2],observations=len(rows),failed_panels=0,seconds=statistics.median(r['seconds'] for r in rows),prefix_tv=q['numerical_tv'],scope='same full output/input; Frugal one CPU worker, stable limit sixteen with at most four active for four records; GPU separate device'))
 for row in rows:
  if row['success']:
   bits=np.load(ROOT/f"results/iteration83_full_timing_v1/samples_{row['records']}_{row['rep']}_{row['method']}_{row['budget']}.npz")['clicks'];check(bits.shape==(row['records'],1152),'timed complete output shape')
save('speed_accuracy.json',paired);save('validation.json',dict(passed=True,checks=len(checks),full_suite=tests,global_mean=mean,global_variance=variance,global_tv_lower_bounds=boundresults,regional_tv_lower_bounds=[dict(budget=b,lower=max(r['nominal_joint99_tv_lower'] for r in regionalchecks if r['budget']==b)) for b in (4,8)],followup_quality=followchecks,global_core=globalcore,timing_panels=len(timing['rows']),limitations=['First-eight accuracy does not certify full distribution.','Global witnesses assume iid generated records and fixed numerical Gaussian reference moments; not directed rounding.','PNR ideal factor moments are not actual integer-plan TV.','Frugal full timing is one observation, one CPU worker; our CPU limit is sixteen, with at most four active for four records.','Full Frugal numerical TV has not been bounded by the four benchmark records.','The two-sector Gaussian reference is an explicit model, not an exact model of all experimental noise.','Different witness confidence families are separately joint99, not jointly99 when combined.','Pure-core truncation budgets require coherent exact downstream sampling; no such sampler is implemented.']))
save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Independent verification passed',len(checks),flush=True)
