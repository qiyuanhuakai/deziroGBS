"""Independently contract saved dense laws and audit benchmark provenance."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[name]='1'
from pathlib import Path
import argparse,hashlib,itertools,json,math,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
from scipy.special import gammainc,gammaincc
from iteration68_build import digest,save

def verify(out):
    out.mkdir(parents=True,exist_ok=False);shutil.copy2(__file__,out/Path(__file__).name);counts={};maximum=0.
    for folder in ('iteration68_build','iteration68_benchmark','iteration68_validation'):
        base=ROOT/'results'/folder;manifest=json.loads((base/'manifest.json').read_text())
        for name,sha in manifest['artifacts'].items():assert digest(base/name)==sha,(folder,name)
        counts[folder+'_artifacts']=len(manifest['artifacts'])
        plan=json.loads((base/'plan.json').read_text());sources=plan.get('source_hashes',plan.get('sources'))
        for name,sha in sources.items():assert digest(ROOT/name)==sha,name
    base=ROOT/'results/iteration68_validation';panels=sorted(base.glob('*_input.npz'));cases=0;paths=0
    for path in panels:
        data=np.load(path);state=data['state'];beta=data['beta'][0];factors=[];effects=[]
        for b,d in zip(beta,state.shape):
            v=np.array([b**n/math.sqrt(math.factorial(n)) for n in range(d)]);c=v/np.linalg.norm(v);h=np.eye(d,dtype=complex);q=c.copy();q[0]+=1;h-=np.outer(q,q.conj())/(1+c[0].real);t=abs(b)**2
            factors.append(np.vstack([math.sqrt(gammaincc(d,t))*h[0],math.sqrt(gammainc(d,t))*h[0],h[1:]]));vac=math.exp(-t)*np.outer(v,v.conj());effects.append((vac,np.eye(d)-vac))
        exact_paths={};law=np.zeros(2**state.ndim)
        for labels in itertools.product(*(range(d+1) for d in state.shape)):
            row=np.ones(1,complex)
            for chosen,rows in zip(labels,factors):row=np.kron(row,rows[chosen])
            probability=float(abs(row@state.ravel())**2)
            if probability>0:exact_paths[labels]=probability
        for bits in itertools.product((0,1),repeat=state.ndim):
            operator=np.ones((1,1),complex)
            for bit,es in zip(bits,effects):operator=np.kron(operator,es[bit])
            index=sum(bit*2**(state.ndim-i-1) for i,bit in enumerate(bits));law[index]=float(np.vdot(state.ravel(),operator@state.ravel()).real)
        saved_labels=[tuple(map(int,row)) for row in data['oracle_labels']];assert len(set(saved_labels))==len(saved_labels);assert set(saved_labels)==set(exact_paths)
        for backend in ('python','cpu_scalar','cpu_blas','cuda'):
            result=np.load(base/(path.name.removesuffix('_input.npz')+'_'+backend+'.npz'));labels=[tuple(map(int,row)) for row in result['refined_labels']];assert labels==saved_labels
            products=np.prod(result['selected_refined_probabilities'],axis=1);expected=np.array([exact_paths[label] for label in labels]);error=float(np.max(abs(products-expected)));assert error<1e-11
            measured=np.zeros_like(law)
            for label,p in zip(labels,products):measured[sum(int(x!=0)*2**(state.ndim-i-1) for i,x in enumerate(label))]+=p
            tv=float(.5*np.sum(abs(measured-law)));assert tv<1e-11;maximum=max(maximum,tv,error);cases+=1
        paths+=len(exact_paths)
    controls=json.loads((base/'summary.json').read_text());assert controls['passed'] and len(controls['rows'])==76 and all(row['passed'] for row in controls['rows'])
    bench=ROOT/'results/iteration68_benchmark';summary=json.loads((bench/'summary.json').read_text());plan=json.loads((bench/'plan.json').read_text());inputs=np.load(bench/'inputs.npz');standard=inputs['standard'];rng=np.random.default_rng(plan['seed']);assert np.array_equal(standard,rng.normal(size=standard.shape));assert np.array_equal(inputs['uniforms'],rng.random(size=inputs['uniforms'].shape))
    for name,sha in plan['dependencies'].items():assert digest(Path(name))==sha,name
    reference=np.load(bench/'0_numpy_4.npz');probability_error=0.
    for row in summary['rows']:
        result=np.load(bench/f"{row['repetition']}_{row['configuration']}.npz");assert np.array_equal(result['refined_labels'],reference['refined_labels']);assert np.array_equal(result['clicks'],result['refined_labels']!=0)
        selected=float(np.max(abs(result['selected_refined_probabilities']-reference['selected_refined_probabilities'])));vacuum=float(np.max(abs(result['vacuum_given_hidden_history']-reference['vacuum_given_hidden_history'])));assert selected==row['selected_max_error'] and vacuum==row['vacuum_max_error'];assert max(selected,vacuum)<1e-11;assert abs(sum(row['chunk_seconds'])-row['seconds'])<1e-12;probability_error=max(probability_error,selected,vacuum)
    ratios={}
    for configuration,statistics in summary['statistics'].items():
        seconds=[r['seconds'] for r in summary['rows'] if r['configuration']==configuration];assert len(seconds)==3;assert float(np.median(seconds))==statistics['median_seconds']
    stats=summary['statistics']
    for threads in (4,8):ratios[f'cpu_{threads}_speedup']=stats[f'numpy_{threads}']['median_seconds']/stats[f'cpu_{threads}']['median_seconds']
    ratios['gpu_vs_fastest_numpy']=min(stats['numpy_4']['median_seconds'],stats['numpy_8']['median_seconds'])/stats['cuda']['median_seconds'];ratios['gpu_samples_per_second']=4096/stats['cuda']['median_seconds'];ratios['jiuzhang_reported_mode_ratio']=8176/144;ratios['distinct_task_sample_time_ratio_to_25_6us']=stats['cuda']['median_seconds']/4096/25.6e-6
    tests=json.loads((ROOT/'results/iteration68_tests/validation.json').read_text());assert tests['executed_gate_passed'] and tests['unchanged_sources'] and tests['new_native_tests_passed']==18
    source_hashes=json.loads((ROOT/'results/iteration68_tests/source_hashes_at_start.json').read_text())
    for name,sha in source_hashes.items():assert digest(ROOT/name)==sha,name
    save(out/'summary.json',dict(passed=True,artifact_counts=counts,independent_dense_backend_cases=cases,distinct_positive_paths=paths,maximum_dense_probability_or_tv_error=maximum,fresh_independent_trajectories=4096,benchmark_measurements=len(summary['rows']),maximum_benchmark_probability_error=probability_error,ratios=ratios,tests=tests,scope='Complete floating-point laws of 12 small finite states, 4096 fresh large-state trajectories. Repeated timing outputs are not additional independent trajectories. Jiuzhang ratio compares different tasks and is not a quality-matched ranking.'))
    print(json.dumps(dict(passed=True,ratios=ratios,maximum_error=probability_error),indent=2))

def seal():
    verification=json.loads((ROOT/'results/iteration68_verification/summary.json').read_text());assert verification['passed']
    artifacts={p.relative_to(ROOT).as_posix():digest(p) for folder in ROOT.glob('results/iteration68_*') if folder.is_dir() for p in folder.rglob('*') if p.is_file()}
    dependencies={}
    for base in ('iteration68_benchmark',):
        plan=json.loads((ROOT/'results'/base/'plan.json').read_text())
        for name,sha in plan['dependencies'].items():
            path=Path(name).resolve();relative=path.relative_to(ROOT).as_posix()
            if relative not in artifacts:dependencies[relative]=sha
    save(ROOT/'results/iteration68_manifest.json',dict(artifacts=artifacts,dependencies=dependencies,count=len(artifacts),bytes=sum((ROOT/n).stat().st_size for n in artifacts),dependency_count=len(dependencies)))

def verify_staged():
    manifest=json.loads((ROOT/'results/iteration68_manifest.json').read_text());index={}
    for record in subprocess.check_output(['git','ls-files','--stage','-z'],cwd=ROOT).split(b'\0'):
        if record:
            header,path=record.split(b'\t',1);mode,oid,stage=header.split();assert stage==b'0';index[path.decode()]=oid
    artifacts=manifest['artifacts'];checked=0
    for name,sha in artifacts.items():
        assert name in index,name;blob=subprocess.check_output(['git','cat-file','blob',index[name].decode()],cwd=ROOT);assert hashlib.sha256(blob).hexdigest()==sha,name;checked+=1
    for name,sha in manifest['dependencies'].items():assert digest(ROOT/name)==sha,name
    print(json.dumps(dict(staged_artifacts_verified=checked,local_dependencies_verified=len(manifest['dependencies']))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['verify','seal','verify-staged']);parser.add_argument('--output',type=Path,default=ROOT/'results/iteration68_verification');args=parser.parse_args()
    if args.action=='verify':verify(args.output.resolve())
    elif args.action=='seal':seal()
    else:verify_staged()
