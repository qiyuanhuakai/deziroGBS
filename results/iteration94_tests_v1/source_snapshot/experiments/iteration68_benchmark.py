"""Fresh-input, interleaved native trajectory benchmark with explicit scope."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import argparse,ctypes as C,hashlib,json,shutil,sys,time,platform
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from gbslab.displaced_sampling import refined_amplitudes,select_positive,prior_displacements
from gbslab.mps_overlap import normalized_right_mps
from gbslab.native_displaced_sampling import NativeDisplacedThresholdSampler
from iteration68_build import digest,save

class FlattenedReference:
    """Same measurement code; one contiguous GEMM per site instead of d GEMMs."""
    def __init__(self,tensors):self.tensors=tensors
    def sample(self,beta,uniforms):
        count,modes=beta.shape;env=np.ones((count,1),complex);labels=np.empty(beta.shape,np.int32);selected=np.empty(beta.shape);vacuum=np.empty(beta.shape);residual=0.
        for site,a in enumerate(self.tensors):
            amplitudes=(env@a.reshape(a.shape[0],-1)).reshape(count,a.shape[1],a.shape[2])
            transformed,masses,weights=refined_amplitudes(amplitudes,beta[:,site]);chosen,total=select_positive(weights,uniforms[:,site]);error=float(np.max(abs(total-1)));residual=max(residual,error)
            if error>1e-9:raise ArithmeticError('Conditional MPS norm drift')
            rows=np.arange(count);physical=np.where(chosen<=1,0,chosen-1);env=transformed[rows,physical,:]/np.sqrt(masses[rows,physical])[:,None]
            labels[:,site]=chosen;selected[:,site]=weights[rows,chosen]/total;vacuum[:,site]=weights[:,0]/total
        return dict(clicks=(labels!=0).astype(np.uint8),refined_labels=labels,selected_refined_probabilities=selected,vacuum_given_hidden_history=vacuum,maximum_normalization_residual=residual)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--build',type=Path,required=True);parser.add_argument('--state',type=Path,required=True);parser.add_argument('--noise',type=Path,required=True);parser.add_argument('--samples',type=int,default=4096);parser.add_argument('--batch',type=int,default=512);parser.add_argument('--repetitions',type=int,default=3);args=parser.parse_args()
    assert args.samples>0 and args.batch>0 and args.repetitions>=3
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);build=args.build.resolve()
    names=['experiments/iteration68_benchmark.py','experiments/iteration68_build.py','src/gbslab/native_displaced_sampling.py','src/gbslab/displaced_sampling.py','src/gbslab/mps_overlap.py','native/displaced_threshold_cpu.cpp','native/displaced_threshold_cuda.cu']
    hashes={n:digest(ROOT/n) for n in names}
    for name in names:
        target=out/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
    dependencies={str(p.resolve()):digest(p) for p in [args.state,args.noise,build/'native_cpu.dll',build/'native_cuda.dll']}
    configurations=['numpy_4','cpu_4','numpy_8','cpu_8','cuda'];order=[configurations[i:]+configurations[:i] for i in (0,2,4)]
    if args.repetitions!=3:order=[configurations[(2*i)%5:]+configurations[:(2*i)%5] for i in range(args.repetitions)]
    save(out/'plan.json',dict(source_hashes=hashes,dependencies=dependencies,seed=680201,samples=args.samples,batch=args.batch,repetitions=args.repetitions,order=order,scope='Fresh prior-Gaussian displacements on fixed 144-mode finite MPS. Same-budget native CPU comparisons and distinct GPU hardware throughput. No Gaussian-target TV certificate or universal algorithm comparison.'))
    save(out/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,logical_cpus=os.cpu_count()))
    z=np.load(args.state);keys=sorted((n for n in z.files if n.startswith('A') and n[1:].isdigit()),key=lambda x:int(x[1:]));raw=[z[k] for k in keys]
    start=time.perf_counter();tensors,log_norm=normalized_right_mps(raw);tensors=[np.ascontiguousarray(a) for a in tensors];normalization=time.perf_counter()-start;del raw
    tensor_hashes=[hashlib.sha256(a.tobytes()).hexdigest() for a in tensors];modes=len(tensors);noise=np.load(args.noise)['full_noise_factor'];rng=np.random.default_rng(680201);standard=rng.normal(size=(args.samples,noise.shape[1]));uniforms=rng.random((args.samples,modes));beta=prior_displacements(noise,standard)
    np.savez_compressed(out/'inputs.npz',standard=standard,uniforms=uniforms,beta=beta)
    blas_path=next((Path(np.__file__).parent.parent/'numpy.libs').glob('*openblas*.dll'));blas=C.CDLL(str(blas_path));setter=blas.scipy_openblas_set_num_threads64_;setter.argtypes=[C.c_int];getter=blas.scipy_openblas_get_num_threads64_;getter.restype=C.c_int
    save(out/'blas.json',dict(path=str(blas_path),sha256=digest(blas_path),integer_abi='ILP64',setting='BLAS threads and native postprocessing threads equal the configuration budget; phases sequential'))
    gcc=Path(shutil.which('g++')).parent;cuda=Path(shutil.which('nvcc')).parent
    cpu=NativeDisplacedThresholdSampler(tensors,build/'native_cpu.dll',backend='cpu',runtime_dirs=[gcc],blas_library=blas_path,canonical=True,threads=4)
    gpu=NativeDisplacedThresholdSampler(tensors,build/'native_cuda.dll',backend='cuda',runtime_dirs=[cuda,cuda/'x64'],canonical=True);reference=FlattenedReference(tensors)
    save(out/'preparation.json',dict(normalization_seconds=normalization,input_log_norm=log_norm,cpu_model_seconds=cpu.model_seconds,gpu_model_seconds=gpu.model_seconds,modes=modes,max_bond=max(max(a.shape[0],a.shape[2]) for a in tensors),tensor_elements=sum(a.size for a in tensors),tensor_sha256=tensor_hashes))
    def run(configuration,limit):
        kind,_,threads=configuration.partition('_');threads=int(threads) if threads else 1;setter(threads);assert getter()==threads;sampler=reference if kind=='numpy' else cpu if kind=='cpu' else gpu
        if kind=='cpu':cpu.threads=threads
        parts=[];elapsed=0.;chunk_times=[]
        for start in range(0,limit,args.batch):
            stop=min(start+args.batch,limit);tick=time.perf_counter();part=sampler.sample(beta[start:stop],uniforms[start:stop]);dt=time.perf_counter()-tick;elapsed+=dt;chunk_times.append(dt);parts.append(part)
        arrays={key:np.concatenate([p[key] for p in parts]) for key in ('clicks','refined_labels','selected_refined_probabilities','vacuum_given_hidden_history')}
        return arrays,dict(configuration=configuration,seconds=elapsed,samples=limit,chunk_seconds=chunk_times,normalization_residual=max(p['maximum_normalization_residual'] for p in parts))
    warmups=[]
    for configuration in configurations:
        _,row=run(configuration,min(args.samples,128));warmups.append(row);print('warmup',configuration,row['seconds'],flush=True)
    save(out/'warmups.json',warmups);rows=[];expected=None
    for repetition,sequence in enumerate(order):
        for configuration in sequence:
            arrays,row=run(configuration,args.samples)
            if expected is None:expected=arrays
            row.update(repetition=repetition,labels_equal=bool(np.array_equal(arrays['refined_labels'],expected['refined_labels'])),selected_max_error=float(np.max(abs(arrays['selected_refined_probabilities']-expected['selected_refined_probabilities']))),vacuum_max_error=float(np.max(abs(arrays['vacuum_given_hidden_history']-expected['vacuum_given_hidden_history']))),mean_clicks=float(arrays['clicks'].sum(axis=1).mean()))
            row['passed']=row['labels_equal'] and max(row['selected_max_error'],row['vacuum_max_error'])<1e-11
            np.savez_compressed(out/f'{repetition}_{configuration}.npz',**arrays);save(out/f'{repetition}_{configuration}.json',row);rows.append(row);print('measured',configuration,repetition,row['seconds'],'passed',row['passed'],flush=True)
    cpu.close();gpu.close();assert hashes=={n:digest(ROOT/n) for n in names}
    statistics={configuration:dict(median_seconds=float(np.median([r['seconds'] for r in rows if r['configuration']==configuration])),minimum_seconds=min(r['seconds'] for r in rows if r['configuration']==configuration),maximum_seconds=max(r['seconds'] for r in rows if r['configuration']==configuration)) for configuration in configurations}
    save(out/'summary.json',dict(passed=all(r['passed'] for r in rows),rows=rows,statistics=statistics,all_labels_equal=all(r['labels_equal'] for r in rows),maximum_probability_error=max(max(r['selected_max_error'],r['vacuum_max_error']) for r in rows),source_hashes=hashes))
    save(out/'manifest.json',dict(artifacts={p.relative_to(out).as_posix():digest(p) for p in out.rglob('*') if p.is_file()}));print('BENCHMARK COMPLETE',statistics,flush=True)
if __name__=='__main__':main()
