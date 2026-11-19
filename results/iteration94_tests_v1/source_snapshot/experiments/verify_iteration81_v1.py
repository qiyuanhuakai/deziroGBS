"""Independent evidence audit. No sampling DLL loaded; 80-digit determinant
targets, reconstructed factor constraints, exact Frugal threshold arithmetic,
independent complete experimental histograms, and explicit scope/CI checks.
"""
from pathlib import Path
import os,sys,argparse,json,hashlib,math,statistics,zipfile
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
for base in sorted((ROOT/'results').glob('iteration81*')):
 if not base.is_dir() or base==OUT:continue
 if (base/'manifest.json').exists():
  for name,digest in read(base/'manifest.json').get('artifacts',{}).items():check(H(base/name)==digest,'frozen '+base.name+'/'+name)
 if (base/'plan.json').exists():
  plan=read(base/'plan.json')
  for key in ('hashes','source_hashes'):
   for name,digest in plan.get(key,{}).items():check(H(ROOT/name)==digest,'executed source '+name)
tests=read(ROOT/'results/iteration81_tests_v1/validation.json');check(tests['passed'] and tests['factor81_passed']==12,'full suite and twelve new tests')
for name,digest in read(ROOT/'results/iteration81_tests_v1/source_hashes.json').items():check(H(ROOT/name)==digest,'tested source '+name)
archive=ROOT/'results/iteration81_large_input_archive_v1';am=read(archive/'manifest.json');h=hashlib.sha256();size=0
for row in am['chunks']:
 b=(archive/row['name']).read_bytes();h.update(b);size+=len(b)
check(h.hexdigest()==am['original_sha256'] and size==am['original_bytes'],'oversized original lossless reconstruction stream')
check(H(ROOT/am['original'])==am['original_sha256'],'original untouched locally')
partial={}
for name,reason in [('iteration81_factor_pilot_v1','Stopped while writing large factor chunks; no complete large result.'),('iteration81_jiuzhang_accuracy_v1','Exact symmetry input guard rejected covariance roundoff.'),('iteration81_jiuzhang_accuracy_v2','Direct B8 signed normalization failure was not caught by driver.')]:
 base=ROOT/'results'/name;check(not (base/'summary.json').exists(),'retained incomplete '+name);files={p.name:dict(bytes=p.stat().st_size,sha256=H(p)) for p in base.iterdir() if p.is_file()};bad=[]
 for p in base.glob('*.npz'):
  try:
   with zipfile.ZipFile(p) as z:
    error=z.testzip()
    if error:bad.append(dict(name=p.name,error=error))
  except Exception as e:bad.append(dict(name=p.name,error=str(e)))
 partial[name]=dict(reason=reason,files=files,unreadable_archives=bad)
save('partial_runs.json',partial)
base=ROOT/'results/iteration81_factor_pilot_v2';factorrows=[]
for row in read(base/'summary.json')['rows']:
 label=row['label']
 if not row['success']:
  check('converge' in row['error'],'finite design infeasibility retained');continue
 d=np.load(base/f'input_{label}.npz');v,w,q,k=d['v'],d['weights'],d['Q'],d['K'];L=len(w);m,n=v.shape;avg=np.zeros((m,m));cross=np.zeros((m,m));maxrow=0;maxp=0;denseerr=0
 check(k.min()>=0 and np.isfinite(k).all(),'nonnegative finite K '+label)
 for first in range(0,L,256):
  f=np.load(base/f'factors_{label}_{first}.npz');aa,bb=f['left'],f['right'];ww=w[first:first+len(aa)];check(min(aa.min(),bb.min())>=0 and np.isfinite(aa).all() and np.isfinite(bb).all(),'nonnegative factors '+label+str(first))
  if L>8192:
   inp=np.load(base/f'inputchunk_{label}_{first}.npz');pp=inp['marginals'];maxp=max(maxp,float(abs(abs(inp['directions']@v.T)**2-pp).max()))
  else:pp=d['marginals'][first:first+len(aa)]
  mm=(aa*(bb@k.T)+bb*(aa@k))/2;avg+=k*(aa.T@(bb*ww[:,None]));cross+=mm.T@(mm*ww[:,None]);maxrow=max(maxrow,float(abs(mm-pp).max()))
  if L<=8192:
   P=k[None]*aa[:,:,None]*bb[:,None,:];P=(P+P.transpose(0,2,1))/2
   if label.startswith('wide'):
    old=np.concatenate([np.load(ROOT/f'results/iteration80_wide_pilot_v1/table_{label[4:]}_{f}.npz')['couplings'] for f in range(first,min(L,first+256),64)])
   else:old=np.load(ROOT/f'results/iteration80_ipf_pilot_v1/{label}.npz')['couplings'][first:first+len(aa)]
   denseerr=max(denseerr,float(abs(P-old).max()))
 ell=(abs(v)**2).sum(1);R=(np.outer(ell,ell)+abs(v@v.T)**2+abs(v@v.conj().T)**2)/(n*(n+2));err=max(maxrow,float(abs((avg+avg.T)/2-q).max()),float(abs(cross-R).max()),maxp)
 check(err<1e-8 and denseerr<1e-8,'independent factor constraints and dense equivalence '+label);check(row['factor_payload_bytes']==8*(m*m+2*L*m),'factor payload formula '+label);factorrows.append(dict(label=label,maximum_constraint_error=err,dense_difference=denseerr if L<=8192 else None))
save('factor_audit.json',factorrows);print('Factor constraints independently verified',flush=True)
mp.mp.dps=80
def mpvac(c,K):
 n=len(c)//2;v=[]
 for mask in range(1<<K):
  ids=[j for j in range(K) if mask>>j&1];ids+=[n+j for j in ids]
  if not ids:v.append(mp.mpf(1));continue
  matrix=mp.matrix([[(mp.mpf(float(c[i,j]))+int(i==j))/2 for j in ids] for i in ids]);v.append(1/mp.sqrt(mp.det(matrix)))
 return v
def invert(v):
 n=len(v);law=[v[s^(n-1)] for s in range(n)]
 for bit in range((n-1).bit_length()):
  for mask in range(n):
   if mask>>bit&1:law[mask]-=law[mask^(1<<bit)]
 return law
target_errors=[];base=ROOT/'results/iteration81_accuracy_comparison_v1';accuracy=read(base/'summary.json')['rows'];targets={}
for case in (0,1,2):
 d=np.load(base/f'input_{case}.npz');K=min(8,len(d['v']));target=np.array([float(v) for v in invert(mpvac(d['cov'],K))]);err=float(abs(target-d['target']).sum()/2);check(err<1e-12,'80 digit synthetic target '+str(case));targets[case]=target;target_errors.append(dict(case=case,tv_from_fp64_target=err));print('80-digit target',case,err,flush=True)
for row in accuracy:
 case,budget,method=row['case'],row['budget'],row['method']
 if method=='frugal_cpu':
  d=np.load(base/f'frugal_{case}_{budget}.npz');tree=read(base/f'frugal_tree_{case}_{budget}.json')['tree'];law=np.array([float(p) for p in full_law(min(8,4 if case==0 else 8),tree,model='uint64')['distribution']]);check(np.max(abs(law-d['law']))<1e-15,'Frugal exact integer-uniform tree law')
 elif method=='direct_cpu_law':law=np.load(base/f'direct_{case}_{budget}.npz')['law']
 elif method=='direct_gpu_law':
  d=np.load(base/f'gpu_{case}_{budget}.npz');K=min(8,d['clicks'].shape[1]);law=np.prod(np.where(d['clicks'][:,:K],d['conditional'],1-d['conditional']),axis=1);check(np.max(abs(law-d['law']))<1e-15,'GPU conditional product')
 else:law=np.load(base/f'pnr_{case}_{method}.npz')['law']
 check(law.min()>=-1e-14 and abs(law.sum()-1)<1e-10 and abs(float(abs(law-targets[case]).sum()/2)-row['numerical_tv'])<1e-12,'TV same target '+str((case,method,budget)))
save('synthetic_high_precision.json',target_errors)
timing=read(ROOT/'results/iteration80_timing_v3/summary.json');joint=[]
for row in accuracy:
 kind={'frugal_cpu':'frugal','direct_gpu_law':'gpu','transport':'cached','sphere':'sphere'}.get(row['method']);key=f"{row['case']}_{kind}_{row['budget']}"
 if key in timing['medians']:
  times=[r for r in timing['rows'] if r['case']==row['case'] and r['kind']==kind and r['budget']==row['budget']];check(len(times)==3,'three complete timing repetitions');joint.append(dict(**row,seconds=statistics.median(r['wall_seconds'] for r in times),records=times[0]['records'],timing_source='iteration80_timing_v3',accuracy_source='iteration81_accuracy_comparison_v1'))
save('speed_accuracy.json',joint)
base=ROOT/'results/iteration81_jiuzhang_accuracy_v3';summary=read(base/'summary.json');histdata=np.load(base/'histograms.npz');ports=histdata['ports'];raw=ROOT.parent/'external_sources/jiuzhang3_iteration81_sample_v1/extracted_3/jiuzhang3/data/high power/high power (1).bin';N=raw.stat().st_size//144;data=np.memmap(raw,dtype=np.uint8,shape=(N,144),mode='r');hist=np.zeros((3,2,256),np.uint64);tot=np.zeros(1153,np.uint64);lookup=np.array([i.bit_count() for i in range(256)],np.uint16)
for start in range(0,N,16384):
 end=min(start+16384,N);d=np.asarray(data[start:end]);tot+=np.bincount(lookup[d].sum(1).astype(int),minlength=1153).astype(np.uint64)
 for g,ids in enumerate(ports):
  bits=(d[:,ids//8]>>(7-ids%8))&1;patterns=(bits*(1<<np.arange(8))).sum(1);split=max(0,min(len(d),100000-start));hist[g,0]+=np.bincount(patterns[:split],minlength=256).astype(np.uint64);hist[g,1]+=np.bincount(patterns[split:],minlength=256).astype(np.uint64)
check(np.array_equal(hist,histdata['histograms']) and np.array_equal(tot,histdata['total_click_histogram']),'all 1492642 raw records independently decoded')
jz=[]
for g in range(3):
 inp=np.load(base/f'input_{g}.npz');t=inp['T'];r=inp['squeezing'];covs=inp['covariances'];vacs=[]
 for s in range(2):
  source=np.diag(np.tile(np.repeat(np.cosh(2*r[:,s]),2),2))
  for i in range(25):source[2*i,2*i+1]=source[2*i+1,2*i]=np.sinh(2*r[i,s]);source[50+2*i,51+2*i]=source[51+2*i,50+2*i]=-np.sinh(2*r[i,s])
  optical=np.block([[t.real,-t.imag],[t.imag,t.real]]);cov=np.eye(16)+optical@(source-np.eye(100))@optical.T;check(abs(cov-covs[s]).max()<1e-13,'independent quadrature propagation '+str((g,s)));vacs.append(mpvac(covs[s],8))
 target=np.array([float(p) for p in invert([a*b for a,b in zip(*vacs)])]);check(abs(target-inp['two_common_sectors']).sum()/2<1e-12,'80 digit two-sector target '+str(g));emp=hist[g,1]/(N-100000)
 for row in [r for r in summary['accuracy'] if r['group']==g]:
  if row['method']=='experiment':
   target0=inp[row['reference']];check(abs(float(abs(emp-target0).sum()/2)-row['empirical_tv'])<1e-14,'experimental TV')
   event=(hist[g,0]/100000)>target0;delta=.01/18;lower=max(0,float(emp[event].sum()-target0[event].sum())-math.sqrt(math.log(1/delta)/(2*(N-100000))));upper=min(1,row['empirical_tv']+math.sqrt((256*math.log(2)+math.log(1/delta))/(2*(N-100000))));check(abs(lower-row['nominal_joint99_lower'])<1e-14 and abs(upper-row['nominal_joint99_upper'])<1e-14,'joint99 heldout bounds')
  elif row.get('failed'):check('normalization lost' in row['error'],'uncensored direct failure')
  else:
   method=row['method'].split('_')[0];b=row['budget'];law=np.load(base/f'law_{g}_{method}_{b}.npz')['law'];check(abs(float(abs(law-target).sum()/2)-row['numerical_tv'])<1e-12,'JZ same reference TV')
   if method=='frugal':
    for s in range(2):
     tree=read(base/f'tree_{g}_{b}_{s}.json')['tree'];f=np.array([float(p) for p in full_law(8,tree,model='uint64')['distribution']]);check(abs(f-np.load(base/f'frugal_{g}_{b}_{s}.npz')['law']).max()<1e-15,'JZ Frugal independent finite RNG law')
 jz.append(dict(group=g,target_roundoff_tv=float(abs(target-inp['two_common_sectors']).sum()/2)));print('Experimental high precision group',g,flush=True)
save('jiuzhang_high_precision.json',jz)
save('validation.json',dict(passed=True,checks=len(checks),full_suite=tests,synthetic_80digit=target_errors,jiuzhang_80digit=jz,limitations=['80-digit arithmetic cross-check is not directed-rounding certification.','Frugal law assumes independent uint64 draws, not an RNG independence proof.','Experimental joint99 intervals assume iid records and a fixed parameter reference.','Eight-port marginals lower-bound full-system TV; never upper-bound it.','Jiuzhang two common spectral sectors are an explicit README-based model; two alternative targets quantify sensitivity.','Large factor payload is coefficient bytes, not measured peak RSS; large sampler not yet integrated.']))
save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Independent validation passed',len(checks),flush=True)
