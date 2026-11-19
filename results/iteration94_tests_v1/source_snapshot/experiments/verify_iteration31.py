"""Independent law reconstruction and provenance checks for wire-native work."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import itertools,json,math,shutil,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from iteration23_support import digest,save
from iteration28_codec import parse_hex

def determinant_law(V):
    m=len(V)//2;Q=(V+np.eye(2*m))/2;vacuum={0:1.}
    for mask in range(1,1<<m):
        modes=[i for i in range(m) if mask&(1<<i)];coordinates=modes+[i+m for i in modes];vacuum[mask]=1/math.sqrt(float(np.linalg.det(Q[np.ix_(coordinates,coordinates)])))
    answer=np.zeros(1<<m)
    for mask in range(1<<m):
        zeros=((1<<m)-1)^mask;clicks=[i for i in range(m) if mask&(1<<i)];answer[mask]=math.fsum((-1)**sum(bits)*vacuum[zeros+sum(1<<i for i,b in zip(clicks,bits) if b)] for bits in itertools.product((0,1),repeat=len(clicks)))
    return answer

def reconstruct(conditionals,order,column):
    m=len(order);law=np.zeros(1<<m)
    for mask in range(1<<m):
        probability=1.;prefix=0
        for length,mode in enumerate(order):
            p=conditionals[(1<<length)-1+prefix,column];bit=(mask>>int(mode))&1;probability*=p if bit else 1-p;prefix|=bit<<length
        law[mask]=probability
    return law

def main():
    out=ROOT/'results/iteration31_validation';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);checks=[]
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    def check(name,condition,**values):
        checks.append(dict(name=name,passed=bool(condition),**values))
        if not condition:save(out/'checks.failed.json',checks);raise AssertionError(name)
    panel=ROOT/'results/iteration31_panel';plan=read(panel/'plan.json');rows=read(panel/'rows.json');summary=read(panel/'summary.json');dependencies=dict(plan['dependencies'])
    dependencies['experiments/iteration28_codec.py']=digest(ROOT/'experiments/iteration28_codec.py')
    check('formal jobs',summary['planned']==summary['completed']==summary['finished']==20 and summary['unchanged_sources'])
    for directory,filename in [('iteration31_build','build.json'),('iteration31_reference_build','build.json'),('iteration31_native_tests','plan.json'),('iteration31_transport','plan.json'),('iteration31_panel','plan.json')]:
        folder=ROOT/'results'/directory;info=read(folder/filename)
        for name,value in info['source_hashes'].items():check(directory+' snapshot '+name,digest(folder/'source_snapshot'/name)==value);check(directory+' current '+name,digest(ROOT/name)==value)
        for field,artifact in [('library_sha256','libwire_forest.dll'),('executable_sha256','native_reference.exe')]:
            if field in info and (folder/artifact).exists():check(directory+' binary '+artifact,digest(folder/artifact)==info[field])
    # The actual kernel, ABI-1 header, legacy adapter and prototype codec are unchanged.
    for name in ('experiments/batched_forest_kernel.cpp','experiments/batched_forest_kernel.h','experiments/portable_scalar_codec.cpp','src/gbslab/batched_forest.py'):
        original=subprocess.check_output(['git','show','4430379e311284efd2cdab13170ab0adf8fa9c7f:'+name],cwd=ROOT)
        # Git text files may have checkout CRLF; compare normalized logical source.
        check('unchanged prior scientific source '+name,original.replace(b'\r\n',b'\n')==(ROOT/name).read_bytes().replace(b'\r\n',b'\n'))
    transport=ROOT/'results/iteration31_transport';tv=read(transport/'validation.json');check('transport all checks',tv['count']==tv['passed']==481 and tv['jobs']==12 and tv['expected_rejections']==4)
    tplan=read(transport/'plan.json')
    for i,job in enumerate(tplan['jobs']):
        data=read(transport/f'case_{i:02d}.json');direct=data['direct'];check(f'transport {i} expected status',direct['status']==job['expected_status'])
        for j,wire in enumerate(data['wire']):
            check(f'transport {i}/{j} controls',wire['control_before']==wire['control_after'] and wire['mxcsr_controls_before']==wire['mxcsr_controls_after'])
            check(f'transport {i}/{j} state machine',all(wire[k]==direct[k] for k in ('status','longdouble_queries','quad_queries','auto_refinements')))
            if direct['status']==0:
                for key in ('values','indicators')+(('scalars',) if job['operation'] else ()):check(f'transport {i}/{j} {key}',wire[key]==[parse_hex(x).hex() for x in direct[key]])
    native_tests=read(ROOT/'results/iteration31_native_tests/validation.json');check('19 substantive tests',native_tests['run']==native_tests['passed']==19 and not native_tests['errors'] and not native_tests['failures'] and not native_tests['skipped'] and native_tests['unchanged_sources'])
    maximum_target_error=0.;maximum_reconstruction_error=0.;replayed=0
    for i,row in enumerate(rows):
        data=np.load(panel/f'case_{i:02d}.npz');samples=data['samples'];order=data['order'];check(f'{i} binary samples',samples.ndim==2 and samples.shape[1]==row['modes'] and np.all((samples==0)|(samples==1)))
        check(f'{i} native wire version',row['native_diagnostics']['formats']['abi_version']==2)
        if i<16:
            direct=determinant_law(data['covariance']);error=float(np.max(abs(direct-data['target_law'])));maximum_target_error=max(maximum_target_error,error);check(f'{i} independent Gaussian target law',error<2e-11,maximum_error=error)
            conditional=data['conditionals'];check(f'{i} conditional domain',np.all((conditional>0)&(conditional<1)))
            for col,name in enumerate(('native_extended_law','reference_extended_law')):
                reconstructed=reconstruct(conditional,order,col);error=float(np.max(abs(reconstructed-data[name][:-1])));maximum_reconstruction_error=max(maximum_reconstruction_error,error);check(f'{i} reconstruct '+name,error<2e-13 and data[name][-1]==0,maximum_error=error)
            uniforms=np.random.default_rng(311400+i).random((128,row['modes']));expected=np.zeros_like(samples)
            for sample in range(128):
                prefix=0
                for length,mode in enumerate(order):
                    p=conditional[(1<<length)-1+prefix,0];bit=int(uniforms[sample,length]<p);expected[sample,mode]=bit;prefix|=bit<<length
            check(f'{i} independent uniform replay',np.array_equal(expected,samples) and np.array_equal(samples,data['reference_samples']));replayed+=len(samples)
            actual_tv=.5*float(np.sum(abs(data['native_extended_law']-data['reference_extended_law'])));check(f'{i} proxy TV',abs(actual_tv-row['extended_law_tv'])<1e-18 and actual_tv<2e-10)
        else:check(f'{i} large outcome recorded',len(samples)==row['successful_samples'] and row['sampling_outcome'] in ('completed','numerical_failure'))
    check('verifier unchanged',digest(source)==sha)
    save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),maximum_independent_target_law_error=maximum_target_error,maximum_law_reconstruction_error=maximum_reconstruction_error,independent_replayed_samples=replayed,dependencies=dependencies,verifier_sha256=sha,checks=checks));print(dict(count=len(checks),maximum_independent_target_law_error=maximum_target_error,maximum_law_reconstruction_error=maximum_reconstruction_error,independent_replayed_samples=replayed))
if __name__=='__main__':main()
