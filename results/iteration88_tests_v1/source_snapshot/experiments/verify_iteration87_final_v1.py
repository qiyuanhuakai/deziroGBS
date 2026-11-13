"""Independent raw-data/domain/decision/confidence/timing audit, no sampler calls."""
from pathlib import Path
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,hashlib,math,statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def close(a,b):return bool(np.allclose(a,b,rtol=2e-12,atol=2e-12))
def words(experiment,sector,offset,count,m):
 # Independent vectorized Philox4x32-10; no native RNG wrapper is imported.
 shape=(count,(m+1)//2);mask=np.uint64(0xffffffff);c0=np.broadcast_to(np.arange(shape[1],dtype=np.uint64),shape).copy();c1=np.zeros(shape,np.uint64);c2=np.broadcast_to(np.arange(offset,offset+count,dtype=np.uint64)[:,None],shape).copy();c3=np.full(shape,(sector<<16)|5,np.uint64);k0=np.uint64(experiment&0xffffffff);k1=np.uint64(experiment>>32)
 for _ in range(10):
  p0=np.uint64(0xD2511F53)*c0;p1=np.uint64(0xCD9E8D57)*c2;c0,c1,c2,c3=(p1>>32)^c1^k0,p1&mask,(p0>>32)^c3^k1,p0&mask;k0=(k0+np.uint64(0x9E3779B9))&mask;k1=(k1+np.uint64(0xBB67AE85))&mask
 result=np.empty((count,2*shape[1]),np.uint64);result[:,0::2]=c0|(c1<<32);result[:,1::2]=c2|(c3<<32);return (result[:,:m]>>11).astype(float)*2**-53
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);checks=0
 def check(value,detail):
  nonlocal checks
  checks+=1
  if not value:raise AssertionError(detail)
 def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
 panels=['blocktree_pilot_v1','blocktree_replay_v1','blocktree_timing_v1','guarded_gates_v1','guarded_gates_v2','guarded_replay_v1','guarded_quality64_v1','guarded_quality128_v1','guarded_timing_v1','joint_quality_v1','tests_v1']
 for panel in panels:
  folder=ROOT/'results'/f'iteration87_{panel}';check(folder.is_dir(),f'panel {panel}')
  if (folder/'manifest.json').is_file():
   for name,expected in read(folder/'manifest.json')['artifacts'].items():check(H(folder/name)==expected,f'artifact {panel}/{name}')
  if (folder/'plan.json').is_file():
   for name,expected in read(folder/'plan.json').get('hashes',{}).items():check(H(ROOT/name)==expected,f'input/source {name}')
 tests=read(ROOT/'results/iteration87_tests_v1/validation.json');check(tests['passed'] and tests['failures']==0 and tests['errors']==0 and tests['unchanged_sources'],'full tests');check(tests['guarded87_passed']==4 and tests['blocktree87_passed']==4,'eight new native tests')
 for name,expected in read(ROOT/'results/iteration87_tests_v1/source_hashes.json').items():check(H(ROOT/name)==expected,f'tested source {name}')
 replay=ROOT/'results/iteration87_blocktree_replay_v1';old=ROOT/'results/iteration86_pnr_quality_v1';check(read(replay/'summary.json')['passed'],'PNR full replay');combined=np.load(old/'combined.npz');sums=np.zeros_like(combined['counts']);valid=np.ones(32768,bool);pnr_domains=set();newleaf=0
 for sector in range(2):
  for offset in range(0,32768,256):
   a=np.load(replay/f'raw_{sector}_{offset}.npz');b=np.load(old/f'raw_{sector}_{offset}.npz');check(a['counts'].shape==(256,1152),'PNR shape');check(np.array_equal(a['counts'],b['counts']) and np.array_equal(a['statistics'][:,:18],b['statistics']) and np.array_equal(a['errors'],b['errors']),'all original PNR fields unchanged');sums[offset:offset+256]+=a['counts'];valid[offset:offset+256]&=(a['errors'][:,0]==0)&(a['statistics'][:,1]==1);newleaf+=int(a['statistics'][:,19].sum())
   for record in range(offset,offset+256):
    for purpose in range(1,5):
     address=(0x86001001,record,(sector<<16)|purpose);check(address not in pnr_domains,'PNR stream collision');pnr_domains.add(address)
 check(np.array_equal(sums,combined['counts']) and np.array_equal(sums>0,combined['clicks']) and np.array_equal(valid,combined['valid']),'PNR combine-before-threshold');check(len(pnr_domains)==262144,'PNR domain count remains old count')
 for sector in range(2):
  meta=read(replay/f'tree_{sector}.json')['metadata'];check(meta[1]==5171200 and 93081600/meta[1]==18,'stored Gram eighteenfold reduction')
 gates=ROOT/'results/iteration87_guarded_gates_v2';check(read(gates/'summary.json')['passed'],'corrected gates');check(not read(ROOT/'results/iteration87_guarded_gates_v1/failure.json')['passed'],'original gate failure retained')
 for r in read(gates/'gram.json')['rows']:check(r['returncode']!=0 or r['enclosed'],'same-Gram enclosure or explicit fallback')
 # Every complete record must honor interval decisions, independent random words
 # and the accounting identities. This does not equate midpoints with probabilities.
 def decision_check(raw,experiment,sector,offset):
  bits,p,b,k,s,e=(raw[key] for key in ('clicks','conditional','bounds','kind','statistics','errors'));n,m=bits.shape;u=words(experiment,sector,offset,n,m);reached=k!=255;ball=k==1;numeric=reached&~ball
  check(np.isfinite(p[reached]).all() and ((p[reached]>=0)&(p[reached]<=1)).all(),'diagnostic midpoint range');check(np.all(b[:,:,0][reached]<=b[:,:,1][reached]),'ordered bounds');check(np.all((u[ball]<b[:,:,0][ball])|(u[ball]>b[:,:,1][ball])),'resolved interval decisions');check(np.array_equal(bits[ball].astype(bool),u[ball]<b[:,:,0][ball]),'interval-selected bit');check(np.array_equal(bits[numeric].astype(bool),u[numeric]<p[numeric]),'numeric threshold decision on these discrete draws');check(np.all(s[:,4]==s[:,8]+s[:,9]+s[:,10]) and np.all(s[:,6]==s[:,7]+s[:,8]+s[:,10]),'precision-path accounting');check(np.all((k==1).sum(1)==s[:,7]-s[:,9]) and np.all((k==2).sum(1)==s[:,4]),'stored kind accounting');return (e[:,0]==0)&(s[:,5]==m)
 replay=ROOT/'results/iteration87_guarded_replay_v1';rs=read(replay/'summary.json');check(rs['passed'],'complete click replay');wide_old=wide_new=0
 for row in rs['rows']:
  budget,sector,offset=row['budget'],row['sector'],row['offset'];new=np.load(replay/f'raw_{budget}_{sector}_{offset}.npz');base=ROOT/'results/iteration86_shared_quality_v1' if budget==64 else replay;name=f'raw_{sector}_{offset}.npz' if budget==64 else f'oracle_{budget}_{sector}_{offset}.npz';old=np.load(base/name);check(np.array_equal(new['clicks'],old['clicks']) and np.array_equal(new['errors'],old['errors']) and np.array_equal(new['statistics'][:,[0,1,2,5]],old['statistics'][:,[0,1,2,5]]),'full replay decision/state identity');p=old['conditional'];b=new['bounds'];check(np.all((b[:,:,0]<=p)&(p<=b[:,:,1])),'frozen numeric oracle inside saved bounds');check(decision_check(new,0x86004001 if budget==64 else 0x87003001,sector,offset).all(),'replay completed');wide_old+=int(old['statistics'][:,4].sum()) if budget==64 else 0;wide_new+=int(new['statistics'][:,4].sum()) if budget==64 else 0
 for budget in (64,128):
  for sector in range(2):
   mainraw=np.load(replay/f'raw_{budget}_{sector}_0.npz')
   for tag in ('serial','chunks'):
    raw=np.load(replay/f'{tag}_{budget}_{sector}.npz')
    for key in raw.files:check(np.array_equal(raw[key],mainraw[key][:7],equal_nan=True),'thread/chunk invariance')
 datasets={'pnr':combined};fresh_domains=set();plan=read(ROOT/'research/iteration87_validation_plan.json');check(plan['events']==31 and plan['joint_failure_budget']==.01,'fixed joint quality family')
 for job in plan['fresh_direct']:
  budget=job['budget'];experiment=int(job['experiment'],16);base=ROOT/f'results/iteration87_guarded_quality{budget}_v1';meta=read(base/'plan.json');check(meta['job']==job,'preregistered fresh job');parts=[];valids=[]
  for sector in range(2):
   rows=[];good=[]
   for offset in range(0,512,32):
    raw=np.load(base/f'raw_{sector}_{offset}.npz');rows.append(raw['clicks']);good.append(decision_check(raw,experiment,sector,offset))
    for record in range(offset,offset+32):
     address=(experiment,record,(sector<<16)|5);check(address not in pnr_domains and address not in fresh_domains,'fresh domain collision');fresh_domains.add(address)
   parts.append(np.concatenate(rows));valids.append(np.concatenate(good))
  z=np.load(base/'combined.npz');check(np.array_equal(z['clicks'],np.bitwise_or(*parts)) and np.array_equal(z['valid'],valids[0]&valids[1]),'inclusive fresh combination');check(int((~z['valid']).sum())==read(base/'summary.json')['failed_records'],'fresh failures retained');datasets[f'direct{budget}']=z
 check(len(fresh_domains)==2048,'two new methods domain count')
 quality=ROOT/'results/iteration87_joint_quality_v1';report=read(quality/'summary.json');check(report['passed'],'quality family completed');delta=.01/31;real=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram'];realN=int(real.sum());realcdf=np.r_[np.cumsum(real)/realN,1.];order=np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz')['order'];profile=read(ROOT/'results/iteration83_bias_profile_v1/summary.json')['reference']
 for name,z in datasets.items():
  result=report['rows'][name];N=len(z['valid']);pilot=2048 if name=='pnr' else 128;check(result['records']==N and result['pilot_records']==pilot,'fixed sample/pilot sizes');eps=math.sqrt(math.log(2/delta)/(2*N));good=z['valid'];bits=z['clicks'];check(result['failed_records']==int((~good).sum()),'quality failure count')
  for kind,rows in (('click',result['local_click']),('pnr',result['optical_pnr'])):
   for j,row in enumerate(rows):
    h=np.load(quality/f'hist_{name}_{kind}_{j}.npz');target=h['target'];hist=h['histogram'];ph=h['pilot_histogram'];vh=h['validation_histogram'];check(hist.sum()==N and ph.sum()==pilot and vh.sum()==N-pilot and np.array_equal(hist,ph+vh),'pilot validation partition');event=ph/pilot>target;check(np.array_equal(event,h['event']),'pilot-defined event');tv=float(abs(hist/N-target).sum()/2);gap=float(vh[event].sum()/(N-pilot)-target[event].sum());check(close(min(1,tv+math.sqrt(math.log(1/delta)/(2*N))+row['target_tail_upper']),row['nominal_joint99_tv_upper']),'TV upper recomputed');check(close(max(0,gap-math.sqrt(math.log(1/delta)/(2*(N-pilot)))),row['nominal_joint99_tv_lower']),'TV lower recomputed')
    if kind=='click':labels=np.sum(bits[:,plan['click_groups'][j]].astype(np.int64)*(2**np.arange(8)),axis=1)
    else:
     pair=plan['optical_pnr_pairs'][j];values=np.column_stack([z['counts'][:,order//8==mode].sum(1) for mode in pair]);inside=(values<=64).all(1);labels=np.full(N,65**2,int);labels[inside]=(65*values[inside,0]+values[inside,1]).astype(int)
    labels[~good]=len(target)-1;check(np.array_equal(np.bincount(labels,minlength=len(target)),hist),'marginal histograms from raw')
  values=bits.sum(1).astype(int);values[~good]=1153;distance=float(abs(np.cumsum(np.bincount(values,minlength=1154))/N-realcdf).max());radius=eps+math.sqrt(math.log(2/delta)/(2*realN));r=result['experimental_click_count'];check(close(distance,r['empirical_cdf_distance']) and close(max(0,distance-radius),r['nominal_joint99_cdf_lower']) and close(min(1,distance+radius),r['nominal_joint99_cdf_upper']),'experimental CDF interval from raw')
  for r in result['click_regions']:
   check(close(r['dkw_radius'],eps),'shared click-CDF event budget')
   if r['lower']<=0:continue
   first,end=r['first'],r['end'];width=end-first+1;values=bits[:,first:end].sum(1).astype(int);values[~good]=width;ref=next(x for x in profile if (x['first'],x['end'])==(first,end));mu=ref['mean']
   if r['direction']=='decreasing':values=width-values;mu=width-mu
   L,U=r['clip_lower'],r['clip_upper'];emp=float(np.clip((values-L)/(U-L),0,1).mean());stop=(math.sqrt(ref['variance']+(U-mu)**2)-(U-mu))/2;lower=(mu-L-stop)/(U-L)-emp-eps;check(close(lower,r['lower']),'full-law clipped witness independently evaluated')
  if name=='pnr':
   refs=read(ROOT/'results/iteration85_joint_quality_v1/regional_photon_reference.json')
   for j,r in enumerate(result['photon_regions']):
    target=np.array([float(v) for v in refs['probabilities'][j]]);K=len(target)-1;first,end=plan['count_regions'][j];values=z['counts'][:,first:end].sum(1).astype(int);values[~good]=K+1;hist=np.bincount(np.minimum(values,K+1),minlength=K+2);distance=float(abs(np.cumsum(hist[:-1])/N-np.cumsum(target)).max());check(close(distance,r['empirical_cdf_distance']) and close(min(1,distance+eps+float(refs['tail_upper'][j])),r['nominal_joint99_cdf_upper']),'PNR CDF confidence recomputed')
 for r in report['frugal_comparisons']:check(close(r['error_ratio_lower'],r['frugal_standalone_tv']/r['pnr_full_trajectory_marginal_tv_upper']),'scoped Frugal precision ratio')
 for name,ratio in report['direct_to_pnr_experimental_error_ratio_lower'].items():check(close(ratio,report['rows'][name]['experimental_click_count']['nominal_joint99_cdf_lower']/report['rows']['pnr']['experimental_click_count']['nominal_joint99_cdf_upper']),'direct versus PNR experimental precision ratio')
 timing=read(ROOT/'results/iteration87_blocktree_timing_v1/summary.json');check(timing['passed'],'PNR formal timing completed')
 for block,median in timing['medians'].items():check(close(median,statistics.median(r['seconds'] for r in timing['rows'] if r['block']==int(block))),'PNR serial median')
 timing=read(ROOT/'results/iteration87_guarded_timing_v1/summary.json');check(timing['passed'],'direct formal timing completed')
 for r in timing['medians']:check(close(r['median_seconds'],statistics.median(x['seconds'] for x in timing['rows'] if x['method']==r['method'] and x['records']==r['records'])),'direct serial median')
 final=dict(passed=True,checks=checks,full_suite=tests,pnr_reused_domains=len(pnr_domains),fresh_direct_domains=len(fresh_domains),pnr_on_demand_primitive_grams=newleaf,b64_replay_wide_calls_old=wide_old,b64_replay_wide_calls_new=wide_new,scope=__doc__,limits=['Many checks are address assertions, not independent experiments.','Joint99 remains nominal under iid and numerical-reference assumptions.','Decision intervals do not certify rounded input states or the complete GBS law.']);save('validation.json',final);save('manifest.json',dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}));print(json.dumps({k:final[k] for k in ('passed','checks','pnr_reused_domains','fresh_direct_domains','b64_replay_wide_calls_old','b64_replay_wide_calls_new')}),flush=True)
if __name__=='__main__':main()
