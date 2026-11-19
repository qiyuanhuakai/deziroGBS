from pathlib import Path
import os,sys,hashlib,json,shutil,traceback,itertools,math,argparse
for n in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[n]='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.special import gammainc,gammaincc
from gbslab.mps_probe import dense_to_mps
from gbslab.mps_overlap import normalized_right_mps
from gbslab.displaced_sampling import DisplacedThresholdSampler
from gbslab.native_displaced_sampling import NativeDisplacedThresholdSampler as NativeSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--build',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False);BUILD=args.build.resolve()
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):
 p=OUT/n;assert not p.exists();p.write_bytes((json.dumps(v,indent=2)+'\n').encode())
sources=[Path(__file__),ROOT/'src/gbslab/native_displaced_sampling.py',ROOT/'native/displaced_threshold_cpu.cpp',ROOT/'native/displaced_threshold_cuda.cu',ROOT/'src/gbslab/displaced_sampling.py',ROOT/'src/gbslab/mps_probe.py',ROOT/'src/gbslab/mps_overlap.py']
for p in sources:
 target=OUT/'source_snapshot'/p.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
cpu=BUILD/'native_cpu.dll';gpu=BUILD/'native_cuda.dll'
blas=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent
save('plan.json',dict(sources={p.relative_to(ROOT).as_posix():H(p) for p in sources},libraries={str(p):H(p) for p in (cpu,gpu,blas)},seed=680101,scope='All positive refined paths and independently contracted dense binary POVM laws; floating point comparison, no interval certificate.'))
def model(tensors,backend):
 if backend=='python':
  r=DisplacedThresholdSampler.__new__(DisplacedThresholdSampler);r.tensors=tensors;r.modes=len(tensors);r.input_log_norm=0.;return r
 return NativeSampler(tensors,gpu if backend=='cuda' else cpu,backend='cuda' if backend=='cuda' else 'cpu',runtime_dirs=[cuda,cuda/'x64'] if backend=='cuda' else [gcc],blas_library=blas if backend=='cpu_blas' else None,canonical=True,threads=2)
def matrices(beta,d):
 t=abs(beta)**2;vector=np.array([beta**n/math.sqrt(math.factorial(n)) for n in range(d)],complex);c=vector/np.linalg.norm(vector);v=c.copy();v[0]+=1;house=np.eye(d)-np.outer(v,v.conj())/(1+c[0].real)
 w=gammaincc(d,t);tail=gammainc(d,t);rows=np.vstack((math.sqrt(w)*house[0],math.sqrt(tail)*house[0],house[1:]));vac=np.exp(-t)*np.outer(vector,vector.conj());return rows,(vac,np.eye(d)-vac)
def oracle(state,beta):
 mats=[matrices(b,d) for b,d in zip(beta,state.shape)];paths=[]
 def visit(remaining,site,labels,uniforms,prob):
  if site==state.ndim:paths.append((labels,uniforms,prob));return
  transformed=mats[site][0]@remaining.reshape(remaining.shape[0],-1);weights=np.sum(abs(transformed)**2,axis=1);total=weights.sum();order=np.argsort(weights,kind='stable');cumulative=0.
  for chosen in order:
   weight=weights[chosen]
   if weight<=0:continue
   mid=(cumulative+.5*weight)/total;assert cumulative/total<=mid<(cumulative+weight)/total
   visit((transformed[chosen]/math.sqrt(weight)).reshape(remaining.shape[1:]),site+1,labels+[int(chosen)],uniforms+[float(mid)],prob*weight/total);cumulative+=weight
 visit(state,0,[],[],1.);binary=[]
 for bits in itertools.product((0,1),repeat=state.ndim):
  operator=np.ones((1,1),complex)
  for bit,(_,effects) in zip(bits,mats):operator=np.kron(operator,effects[bit])
  binary.append(float(np.vdot(state.ravel(),operator@state.ravel()).real))
 return paths,np.array(binary)
rng=np.random.default_rng(680101);cases=[]
for dims in [(1,),(2,),(3,),(4,),(2,3),(3,2,4),(2,2,2),(4,3,2),(32,)]:
 state=rng.normal(size=dims)+1j*rng.normal(size=dims);state/=np.linalg.norm(state);beta=(rng.normal(size=len(dims))+1j*rng.normal(size=len(dims)))*.4;cases.append((f'random_{len(cases)}',state,beta))
cases += [('zero_beta_product',np.array([[1.,0.],[0.,0.]],complex),np.zeros(2,complex)),('tiny_tail_d1',np.ones((1,),complex),np.array([1e-8+0j])),('mixed_zero',np.array([[.5,.5],[.5,.5]],complex),np.array([0.,.3j]))]
rows=[]
for name,state,beta in cases:
 paths,exact=oracle(state,beta);labels=np.array([p[0] for p in paths],np.int32);u=np.array([p[1] for p in paths]);betas=np.tile(beta,(len(u),1));tensors,_=dense_to_mps(state,max_bond=64);tensors,_=normalized_right_mps(tensors)
 np.savez_compressed(OUT/(name+'_input.npz'),state=state,beta=betas,uniforms=u,oracle_labels=labels,oracle_path_probabilities=np.array([p[2] for p in paths]),oracle_binary=exact,**{f'A{i}':a for i,a in enumerate(tensors)})
 for backend in ('python','cpu_scalar','cpu_blas','cuda'):
  row=dict(case=name,backend=backend,paths=len(u));sampler=None
  try:
   sampler=model(tensors,backend);out=sampler.sample(betas,u);law=np.zeros_like(exact);indices=np.sum(out['clicks']*(2**np.arange(state.ndim-1,-1,-1)),axis=1)
   for index,p in zip(indices,np.prod(out['selected_refined_probabilities'],axis=1)):law[index]+=p
   row.update(labels_equal=bool(np.array_equal(labels,out['refined_labels'])),binary_tv=float(.5*np.sum(abs(law-exact))),path_error=float(np.max(abs(np.prod(out['selected_refined_probabilities'],axis=1)-np.array([p[2] for p in paths])))),normalization_residual=out['maximum_normalization_residual']);row['passed']=row['labels_equal'] and row['binary_tv']<1e-11 and row['path_error']<1e-11
   np.savez_compressed(OUT/(name+'_'+backend+'.npz'),**out,binary_law=law)
  except Exception:row.update(passed=False,error=traceback.format_exc())
  finally:
   if isinstance(sampler,NativeSampler):sampler.close()
  rows.append(row);save(name+'_'+backend+'.json',row);print(row,flush=True)
negative=[('beta_nan',np.array([1.,0.]),complex(float('nan')),0.5),('u_endpoint',np.array([1.,0.]),.2,1.),('intensity_underflow',np.array([1.,0.]),1e-200,.5),('poisson_underflow',np.array([1.,0.,0.,0.]),40.,.5),('norm_underflow',np.array([1.,1e-200]),0.,.5),('uncanonical_norm',np.array([2.,0.]),.2,.5),('destructive_zero',np.array([1.,-1.])/math.sqrt(2),1.,.5)]
for name,state,beta,u in negative:
 for backend in ('python','cpu_scalar','cpu_blas','cuda'):
  sampler=model([state.astype(complex).reshape(1,-1,1)],backend);row=dict(case=name,backend=backend,expected_rejection=True)
  try:sampler.sample(np.array([[beta]]),np.array([[u]]));row.update(passed=False,error='Accepted expected numerical rejection')
  except (ValueError,ArithmeticError) as e:row.update(passed=True,error=str(e))
  finally:
   if isinstance(sampler,NativeSampler):sampler.close()
  rows.append(row);save(name+'_'+backend+'.json',row);print(row,flush=True)
save('summary.json',dict(passed=all(r['passed'] for r in rows),rows=rows,positive_cases=len(cases),negative_cases=len(negative),largest_tv=max((r.get('binary_tv',0.) for r in rows))))
save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
