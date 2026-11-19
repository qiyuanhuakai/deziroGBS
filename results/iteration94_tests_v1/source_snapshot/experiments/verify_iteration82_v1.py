"""Independent iteration82 evidence audit, no sampler DLL loaded.
Exact integer audits, 80-digit Cholesky targets, factor generating functions,
failed-prefix mass checks, GPU replay and complete timing/quality joins.
"""
from pathlib import Path
import os,argparse,json,hashlib,math,statistics,sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.frugal_law import full_law
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def save(name,obj):
 with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
checks=[]
def check(ok,msg):
 assert ok,msg
 checks.append(msg)
for base in sorted((ROOT/'results').glob('iteration82*')):
 if not base.is_dir() or base==OUT:continue
 if (base/'manifest.json').exists():
  for name,digest in read(base/'manifest.json').get('artifacts',{}).items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
 if (base/'plan.json').exists():
  plan=read(base/'plan.json')
  for name,digest in plan.get('hashes',{}).items():check(H(ROOT/name)==digest,'executed '+name)
tests=read(ROOT/'results/iteration82_tests_v1/validation.json');check(tests['passed'] and tests['sampling82_passed']==16,'sixteen new tests and complete suite')
for name,digest in read(ROOT/'results/iteration82_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
integer=read(ROOT/'results/iteration82_integer_audit_v1/summary.json');check(integer['passed'],'exact integer audit completed')
for row in integer['rows']:
 check(row['fixed_total34_tv_upper']['approximation']<1e-10,'integer TV bound '+row['label'])
 check(row['zero_attempt_failure_probability'] or row['log10_max_attempt_failure_bound_N34']<-100,'rejection work-bound failure probability')
pilot=read(ROOT/'results/iteration82_factor_pilot_v1/summary.json');check(pilot['all_passed'] and pilot['strict_low_branch_replay'],'factor pilot and exact low replay')
check(np.array_equal(np.load(ROOT/'results/iteration82_factor_pilot_v1/strict_replay.npz')['counts'],np.load(ROOT/'results/iteration77_number_pilot_v2/counts_1.npz')['counts']),'all 10000 historical exact-branch records')
for row in pilot['rows']:
 counts=np.load(ROOT/f"results/iteration82_factor_pilot_v1/counts_{row['label']}.npz")['counts'];check(counts.shape[0]==row['fixed_records'] and np.all(counts.sum(1)==34),'fixed-total conservation '+row['label']);s=row['fixed_statistics'];check(s['pair_proposals']-s['pair_rejections']==17*len(counts),'proposal accounting '+row['label'])
large=np.load(ROOT/'results/iteration82_factor_pilot_v1/unconditional_large128.npz');check(np.array_equal(large['counts'].sum(1),large['totals']) and np.all(large['totals']%2==0),'large unconditional counts and parity')
z=float(np.tanh(1.1))**2;mean=128*z/(1-z);variance=256*z/(1-z)**2;check(abs(large['totals'].mean()-mean)<8*math.sqrt(variance/len(large['totals'])),'large negative binomial mean eight-sigma check')
mp.mp.dps=80
def inverse(v):
 law=[v[i^(len(v)-1)] for i in range(len(v))]
 for bit in range((len(v)-1).bit_length()):
  for mask in range(len(v)):
   if mask>>bit&1:law[mask]-=law[mask^(1<<bit)]
 return law
def vacuum_mp(c,depth):
 n=len(c)//2;out=[]
 for mask in range(1<<depth):
  ids=[j for j in range(depth) if mask>>j&1];ids+=[n+j for j in ids]
  if not ids:out.append(mp.mpf(1));continue
  q=mp.matrix([[(mp.mpf(float(c[i,j]))+int(i==j))/2 for j in ids] for i in ids]);l=mp.cholesky(q);out.append(1/mp.fprod(l[i,i] for i in range(len(ids))))
 return out
base=ROOT/'results/iteration82_factor_accuracy_v1';synthetic=[];targets={}
for label in ('strong4','wide2','large128'):
 inp=np.load(base/f'input_{label}.npz');depth=min(8,len(inp['v']));target=np.array([float(x) for x in inverse(vacuum_mp(inp['cov'],depth))]);targets[label]=target;difference=float(abs(target-inp['target']).sum()/2);check(difference<1e-12,'80 digit Cholesky target '+label)
 f=np.load(base/f'factor_{label}.npz');tv=float(abs(f['law']-target).sum()/2);saved=read(base/f'row_{label}_factor.json');check(abs(tv-saved['numerical_tv'])<1e-12,'PNR factor numerical TV '+label);synthetic.append(dict(label=label,target_roundoff_tv=difference,pnr_tv_against_80digit=tv));print('80-digit synthetic',label,tv,flush=True)
 for cutoff in (2,3):
  tree=read(base/f'tree_{label}_{cutoff}.json')['tree'];law=np.array([float(p) for p in full_law(depth,tree,model='uint64')['distribution']]);check(np.max(abs(law-np.load(base/f'frugal_{label}_{cutoff}.npz')['law']))<1e-15,'Frugal exact finite-random-word tree '+label+str(cutoff))
 # A separate double-precision matrix factor calculation checks no-click events.
 oldbase=ROOT/'results/iteration81_factor_pilot_v2';old=np.load(oldbase/f'input_{label}.npz');w,k=old['weights'],old['K'];L=len(w);v=old['v'];m,n=v.shape;parts=[np.load(oldbase/f'factors_{label}_{i}.npz') for i in range(0,L,256)];aa=np.concatenate([x['left'] for x in parts]);bb=np.concatenate([x['right'] for x in parts]);forward=aa*(bb@k.T);backward=bb*(aa@k);normal=forward.sum(1);pairs=aa[:,:depth,None]*k[None,:depth,:depth]*bb[:,None,:depth];zz=float(inp['t'])**2;shape=n/2;independent=np.empty(1<<depth)
 for mask in range(1<<depth):
  ids=[j for j in range(depth) if mask>>j&1];s=(normal-forward[:,ids].sum(1)-backward[:,ids].sum(1)+pairs[:,ids][:,:,ids].sum((1,2)))/normal
  # For all physical ports no pair can avoid detection; use that exact identity.
  if depth==m and mask==(1<<depth)-1:s=np.zeros_like(s)
  check(s.min()>-1e-13 and s.max()<1+1e-13,'factor no-click range '+label+str(mask));independent[mask]=np.dot(w,np.exp(shape*(np.log1p(-zz)-np.log1p(-zz*s))))
 check(np.max(abs(independent-f['full_vacuum']))<1e-12,'independent ideal factors versus actual integer law '+label)
 # Exact low coefficients independently from singular values, rather than H^T H traces.
 low=np.ones(1<<depth,np.longdouble)
 for mask in range(1<<depth):
  remain=v[[j for j in range(m) if not (j<depth and mask>>j&1)]]
  if len(remain):
   _,singular,vh=np.linalg.svd(remain,full_matrices=False);basis=singular[:,None]*vh;ev=np.linalg.svd(basis@basis.T,compute_uv=False)**2
  else:ev=np.zeros(0)
  coeff=np.ones(5,np.longdouble)
  for j in range(1,5):coeff[j]=sum(np.longdouble(.5)*np.sum(ev**q)*coeff[j-q] for q in range(1,j+1))/j
  low[mask]=np.longdouble(1-zz)**(np.longdouble(n)/2)*sum(np.longdouble(zz)**j*coeff[j] for j in range(5))
 check(np.max(abs(low-inp['exactlow']))<1e-12,'independent low-total singular-value coefficients '+label)
save('synthetic_high_precision.json',synthetic)
stablebase=ROOT/'results/iteration82_stable_pilot_v1';stable=read(stablebase/'summary.json');jz=[]
for g in range(3):
 covs=np.load(ROOT/f'results/iteration81_jiuzhang_accuracy_v3/input_{g}.npz')['covariances'];vv=[vacuum_mp(c,8) for c in covs];refs=[np.array([float(p) for p in inverse(v)]) for v in vv];target=np.array([float(p) for p in inverse([a*b for a,b in zip(*vv)])]);stored=np.load(stablebase/f'references_{g}.npz');check(np.max(abs(np.array(refs)-stored['sector_laws']))<1e-14,'independent Cholesky sector reference '+str(g))
 for row in stable['rows']:
  if row['group']!=g:continue
  budget=row['budget']
  if row['method']=='stable_two_sectors_or':
   law=np.load(stablebase/f'combined_{g}_{budget}.npz')['law'];tv=float(abs(law-target).sum()/2);check(abs(tv-row['numerical_tv'])<1e-12 and abs(law.sum()-1)<1e-12,'stable combined law '+str((g,budget)));jz.append(dict(group=g,budget=budget,numerical_tv_80digit=tv));continue
  method,sector=row['method'],row['sector'];law=np.load(stablebase/f'law_{g}_{sector}_{budget}_{method}.npz')['law'];errors=read(stablebase/f'errors_{g}_{sector}_{budget}_{method}.json');check(np.isnan(law).sum()==len(errors)==row['failed_patterns'],'uncensored failed leaves')
  if not errors:check(abs(float(abs(law-refs[sector]).sum()/2)-row['numerical_tv'])<1e-12,'stable/old same sector target')
  prefixes=row['failure_prefixes'];check(not any(a!=b and a.startswith(b) for a in prefixes for b in prefixes),'failure cylinders disjoint');mass=0
  for prefix,detail in prefixes.items():
   mask=sum(int(bit)<<i for i,bit in enumerate(prefix));matching=(np.arange(256)&((1<<len(prefix))-1))==mask;mass+=float(refs[sector][matching].sum())
  check(abs(mass-row['target_mass_of_failed_patterns'])<1e-12,'union of first-failure cylinders equals failed target mass');check(abs(sum(d['reachable_prefix_probability'] for d in prefixes.values())-row['abort_probability_under_continuous_draws'])<1e-20,'algorithmic abort probability sum')
 check(all(any(r['group']==g and r['budget']==b and r['method']=='stable_two_sectors_or' for r in stable['rows']) for b in (8,16,32)),'all repaired budgets complete group '+str(g));check(any(r['group']==g and r['budget']==128 and r.get('failed_patterns',0)>0 and r['method']=='stable' for r in stable['rows']),'rare B128 failure retained group '+str(g))
save('jiuzhang_high_precision.json',jz)
gpu=read(ROOT/'results/iteration82_gpu_accuracy_v1/summary.json')
for row in gpu['rows']:
 check(row['passed'],'GPU actual law audit '+str((row['label'],row['budget'])));d=np.load(ROOT/f"results/iteration82_gpu_accuracy_v1/law_{row['label']}_{row['budget']}.npz");bits=d['clicks'][:,:8];law=np.prod(np.where(bits,d['conditional'],1-d['conditional']),axis=1);check(np.max(abs(law-d['law']))<1e-15,'GPU joint law from actual conditionals');check(abs(float(abs(law-targets[row['label']]).sum()/2)-row['numerical_tv'])<1e-12,'GPU high precision target TV')
timing=read(ROOT/'results/iteration82_timing_v1/summary.json');check(timing['completed'] and all(r['success'] for r in timing['rows']),'all timing panels completed');quality=read(ROOT/'results/iteration82_factor_accuracy_v1/summary.json')['rows'];paired=[]
for label,method,budget in sorted(set((r['label'],r['method'],r['budget']) for r in timing['rows'])):
 rr=[r for r in timing['rows'] if (r['label'],r['method'],r['budget'])==(label,method,budget)];check(len(rr)==3 and {r['rep'] for r in rr}=={0,1,2},'three separate timing rotations')
 if method=='gpu':q=next(r for r in gpu['rows'] if (r['label'],r['budget'])==(label,budget))['numerical_tv']
 elif method=='frugal':q=next(r for r in quality if r['label']==label and r['method']=='frugal_cpu' and r['budget']==budget)['numerical_tv']
 elif method=='factor':
  q=next(r for r in quality if r['label']==label and r['method']=='factor_pnr_b8')['numerical_tv'];check(max(max(r['factor_max_relative_differences']) for r in rr)<1e-12,'timed factors same to numerical precision');check(all(r['maximum_weight_difference']==0 for r in rr),'same latent weights')
 else:q=next(r for r in read(ROOT/'results/iteration81_accuracy_comparison_v1/summary.json')['rows'] if r['case']==2 and r['method']=='transport')['numerical_tv']
 paired.append(dict(label=label,method=method,budget=budget,records=rr[0]['records'],seconds=statistics.median(r['seconds'] for r in rr),numerical_tv=q,scope='same first eight physical ports'))
 for row in rr:
  d=np.load(ROOT/f"results/iteration82_timing_v1/{label}_{row['rep']}_{method}_{budget}.npz");check(d['clicks'].shape==(row['records'],144 if label=='wide2' else 384),'complete bit output shape')
  if 'counts' in d:check(np.array_equal(d['counts']>0,d['clicks']) and np.all(d['counts'].sum(1)%2==0),'timed integer counts then threshold conversion')
for method,budget in sorted(set((r['method'],r['budget']) for r in timing['jiuzhang'])):
 rr=[r for r in timing['jiuzhang'] if (r['method'],r['budget'])==(method,budget)];check(len(rr)==3,'three Jiuzhang timing rotations')
 if method=='stable':q=next(r for r in stable['rows'] if r['group']==0 and r['method']=='stable_two_sectors_or' and r['budget']==budget)['numerical_tv']
 else:q=next(r for r in read(ROOT/'results/iteration81_jiuzhang_accuracy_v3/summary.json')['accuracy'] if r['group']==0 and r['method']==('frugal_two_sectors_or' if method=='frugal' else 'direct_two_sectors_or') and r['budget']==budget)['numerical_tv']
 paired.append(dict(label='jiuzhang_group0',method=method,budget=budget,records=8192,seconds=statistics.median(r['seconds'] for r in rr),numerical_tv=q,scope='two-sector OR, fixed eight detectors only'))
for row in timing['reuse']:
 d=np.load(ROOT/f"results/iteration82_timing_v1/reuse_{row['label']}_{row['rep']}.npz")['counts'];check(len(d)==100000 and np.all(d.sum(1)%2==0),'reuse record count and parity');check(d.sum()==row['statistics']['photons'],'reuse photon statistics')
save('speed_accuracy.json',paired)
save('validation.json',dict(passed=True,checks=len(checks),full_suite=tests,integer_cases=len(integer['rows']),synthetic_80digit=synthetic,jiuzhang_80digit=jz,timing_panels=len(timing['rows'])+len(timing['jiuzhang']),reuse_panels=len(timing['reuse']),limitations=['Marginal TV is a full-TV lower bound, never a full-TV upper bound.','Integer rational bounds apply to exported small plans, not quantum approximation or PRNG independence.','Large native probabilities and 80-digit references are numerical, not directed-rounding certificates.','Timings exclude file I/O, DLL load and initial CUDA context; factor design/calibration/plan/destruction charged.','Jiuzhang speed samples only an eight-bin subsystem, not all 1152 detectors.','Stable B128 rare normalization failures remain; no clipping or threshold relaxation.']))
save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Independent verification passed',len(checks),flush=True)
