"""Independent Gaussian quadrature and raw-tensor trajectory audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,itertools,json,math,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from gbslab.mps_overlap import scaled_overlap
from iteration23_support import digest,save,load_mps,mps_fingerprint
from iteration22_global import load_family
from iteration22_external import ARCHIVE_SHA
from verify_iteration29 import quadrature_density,dense_reduced

def independent_row(beta,d,label):
    t=abs(beta)**2; coefficients=np.array([beta**n/math.sqrt(math.factorial(n)) for n in range(d)],complex)
    norm=float(np.linalg.norm(coefficients));c=coefficients/norm
    if label<=1:
        if label==0:mass=math.exp(-t)*norm**2
        else:
            term=math.exp(-t)*t**d/math.factorial(d);terms=[term]
            for k in range(d+1,d+1000):
                term*=t/k;terms.append(term)
                if term<2e-17*math.fsum(terms):break
            else:raise ArithmeticError('Independent Poisson series did not converge')
            mass=math.fsum(terms)
        return -c.conj(),math.log(mass)
    j=label-1;row=-c[j]*c.conj()/(1+c[0].real);row[0]=-c[j];row[j]+=1
    return row,0.

def raw_trajectory(tensors,beta,labels,log_norm_squared):
    env=np.ones(1,complex);logs=0.
    for A,b,label in zip(tensors,beta,labels):
        row,log_mass=independent_row(b,A.shape[1],int(label));local=np.einsum('n,anb->ab',row,A);env=env@local;scale=float(np.max(abs(env)))
        if scale<=0:raise ArithmeticError('Independent trajectory is zero')
        env/=scale;logs+=2*math.log(scale)+log_mass
    return logs+2*math.log(abs(env[0]))-log_norm_squared

def direct_conditional(dense,beta):
    rho=np.outer(dense.ravel(),dense.ravel().conj()).reshape(dense.shape+dense.shape);answer=[]
    for bits in itertools.product((0,1),repeat=dense.ndim):
        work=rho.copy();remaining=dense.ndim
        for d,b,bit in zip(dense.shape,beta,bits):
            v=np.array([math.exp(-abs(b)**2/2)*b**n/math.sqrt(math.factorial(n)) for n in range(d)]);E=np.outer(v,v.conj());E=np.eye(d)-E if bit else E
            work=np.tensordot(work,E,axes=([0,remaining],[1,0]));remaining-=1
        answer.append(float(work.real))
    return np.array(answer)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);checks=[]
    def check(name,condition,**values):
        checks.append(dict(name=name,passed=bool(condition),**values))
        if not condition:save(out/'failed_checks.json',checks);raise AssertionError(name)
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    panel=ROOT/'results/iteration30_panel';plan=read(panel/'plan.json');rows=read(panel/'rows.json');summary=read(panel/'summary.json')
    check('all formal jobs completed',summary['finished']==summary['planned']==summary['completed']==22 and summary['unchanged_sources'])
    dependencies=dict(plan['dependencies']);dependencies['experiments/verify_iteration29.py']=digest(ROOT/'experiments/verify_iteration29.py')
    for name,value in dependencies.items():check('dependency '+name,digest(ROOT/name)==value)
    for name,value in plan['source_hashes'].items():
        check('source '+name,digest(ROOT/name)==value);check('snapshot '+name,digest(panel/'source_snapshot'/name)==value)
    pilot=ROOT/'results/iteration30_pilot';pilot_summary=read(pilot/'summary.json')
    for name,value in pilot_summary['source_hashes'].items():check('pilot snapshot '+name,digest(pilot/'source_snapshot'/name)==value)
    check('external archive',digest(args.archive)==ARCHIVE_SHA)
    F=np.load(ROOT/'results/iteration28_panel/case_14.npz')['full_noise_factor'];rng=np.random.default_rng(301301);z=rng.normal(size=(4096,F.shape[1]));u=rng.random((4096,144));mu=z@F.T;b=-(mu[:,:144]+1j*mu[:,144:])/2
    random=np.load(panel/'large_random.npz');check('large prior normals',np.array_equal(z,random['standard']));check('large uniforms',np.array_equal(u,random['uniforms']));check('large prior displacement mapping',np.array_equal(b,random['beta']))
    max_quad=0.;max_trajectory=0.;quad_count=0;trajectories=[]
    for i,row in enumerate(rows):
        data=np.load(panel/f'case_{i:02d}.npz');clicks=data['clicks'];labels=data['refined_labels'];n=row['samples'];m=clicks.shape[1]
        check(f'{i} binary dimensions',clicks.shape==(n,m) and np.all((clicks==0)|(clicks==1)))
        check(f'{i} labels are not photon counts',np.array_equal(clicks,(labels!=0).astype(np.uint8)))
        check(f'{i} hidden probability domain',np.all(data['audit_selected_refined_probabilities']>0) and np.all(data['audit_selected_refined_probabilities']<=1))
        if i<16:
            old=np.load(ROOT/f'results/iteration29_panel/case_{i:02d}.reference.npz');dense=old['dense'];noise=old['noise_factor'];r=np.random.default_rng(row['seed']);zz=r.normal(size=(n,noise.shape[1]));uu=r.random((n,m));mm=zz[:16]@noise.T
            check(f'{i} prior mapping',np.array_equal(data['audit_beta'],-(mm[:,:m]+1j*mm[:,m:])/2));check(f'{i} uniforms',np.array_equal(uu[:16],data['audit_uniforms']))
            reference=direct_conditional(dense,data['conditional_beta']);error=float(np.max(abs(reference-data['conditional_refined_law'])));check(f'{i} entire conditional law',error<2e-11,maximum_error=error)
            subset={0:1.}
            for mask in range(1,2**m):
                sites=[j for j in range(m) if mask&(1<<j)];dims=tuple(dense.shape[j] for j in sites);rho=dense_reduced(dense,sites)
                subset[mask]=quadrature_density(rho,dims,noise[sites+[j+m for j in sites]]);quad_count+=1
            mixed=[]
            for bits in itertools.product((0,1),repeat=m):
                vacuum=sum(1<<j for j,bit in enumerate(bits) if not bit);active=[j for j,bit in enumerate(bits) if bit]
                mixed.append(math.fsum((-1)**sum(take)*subset[vacuum+sum(1<<j for j,t in zip(active,take) if t)] for take in itertools.product((0,1),repeat=len(active))))
            error=float(np.max(abs(np.array(mixed)-data['mixed_reference'])));max_quad=max(max_quad,error);check(f'{i} mixed entire quadrature law',error<2e-11,maximum_error=error)
            counts=np.bincount(clicks@(2**np.arange(m-1,-1,-1)),minlength=2**m);check(f'{i} full counts',np.array_equal(counts,data['histogram']));check(f'{i} frequency law',np.array_equal(counts/n,data['empirical_law']))
            for j in range(m):check(f'{i} label range {j}',np.all(labels[:,j]<=dense.shape[j]))
        else:
            family=row['family'];check(f'{i} common beta',np.array_equal(data['audit_beta'],b[:16]));check(f'{i} common uniforms',np.array_equal(data['audit_uniforms'],u[:16]))
            if family.startswith('new_'):tensors=load_mps(ROOT/f'results/iteration25_panel_retry/run_{i}.state.npz')
            else:
                tensors,entries=load_family(family,ROOT/'results/iteration22_external',args.archive);check(f'{i} author members',entries==read(panel/(family+'_entries.json')))
            check(f'{i} tensor fingerprint',mps_fingerprint(tensors)==row['raw_mps_fingerprint']);norm=scaled_overlap(tensors,tensors);check(f'{i} positive raw norm',not norm['zero'] and abs(complex(*norm['phase'])-1)<1e-10)
            for index in (0,3,7,15):
                actual=raw_trajectory(tensors,b[index],labels[index],norm['log_abs']);stored=math.fsum(np.log(data['audit_selected_refined_probabilities'][index]));error=abs(actual-stored);max_trajectory=max(max_trajectory,error)
                check(f'{i} raw trajectory {index}',error<2e-9,log_probability_error=error);trajectories.append(dict(case=i,index=index,raw_log_probability=actual,stored_log_probability=stored,error=error))
            for j,A in enumerate(tensors):check(f'{i} label range {j}',np.all(labels[:,j]<=A.shape[1]))
            check(f'{i} all single counts',np.array_equal(1-clicks.mean(axis=0),data['empirical_single_vacuum']))
            old=np.load(ROOT/f'results/iteration29_panel/case_{i:02d}.npz')
            for j,start in enumerate(old['pair_starts']):check(f'{i} pair counts {start}',np.array_equal(np.bincount(clicks[:,start]+2*clicks[:,start+1],minlength=4)/n,data['empirical_pair_threshold'][j]))
            check(f'{i} click counts',np.array_equal(np.bincount(clicks.sum(axis=1).astype(int),minlength=145),data['click_histogram']));del tensors
        print('verified',i,flush=True)
    check('verifier unchanged',digest(source)==sha)
    save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),quadrature_subsets=quad_count,maximum_mixed_quadrature_error=max_quad,raw_trajectory_checks=trajectories,maximum_raw_trajectory_log_error=max_trajectory,dependencies=dependencies,verifier_sha256=sha,checks=checks))
    print(dict(count=len(checks),maximum_mixed_quadrature_error=max_quad,maximum_raw_trajectory_log_error=max_trajectory),flush=True)
if __name__=='__main__':main()
