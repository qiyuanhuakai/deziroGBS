import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import itertools,json,math,shutil,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from test_threshold_blocks import bargmann_effect
from verify_iteration31 import determinant_law,reconstruct
from iteration23_support import digest,save

def reduced(rho,dims,sites):
    remaining=[i for i in range(len(dims)) if i not in sites];order=sites+remaining+[i+len(dims) for i in sites+remaining];d=math.prod(dims[i] for i in sites);r=math.prod(dims[i] for i in remaining)
    work=rho.reshape(dims+dims).transpose(order).reshape(d,r,d,r);return np.einsum('arbr->ab',work)

def inclusion(vacua):
    count=len(vacua);k=count.bit_length()-1;law=[]
    for mask in range(count):
        zeros=(count-1)^mask;clicks=[i for i in range(k) if mask&(1<<i)]
        law.append(math.fsum((-1)**sum(bits)*vacua[zeros+sum(1<<j for j,b in zip(clicks,bits) if b)] for bits in itertools.product((0,1),repeat=len(clicks))))
    return np.array(law)

def main():
    out=ROOT/'results/iteration32_validation';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);checks=[]
    def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
    def check(name,condition,**values):
        checks.append(dict(name=name,passed=bool(condition),**values))
        if not condition:save(out/'checks.failed.json',checks);raise AssertionError(name)
    panel=ROOT/'results/iteration32_panel';plan=read(panel/'plan.json');rows=read(panel/'rows.json');summary=read(panel/'summary.json');check('all formal jobs',summary['planned']==summary['finished']==summary['completed']==48 and summary['unchanged_sources']);dependencies=dict(plan['dependencies'])
    for name,value in dependencies.items():check('dependency '+name,digest(ROOT/name)==value)
    for name,value in plan['source_hashes'].items():check('source '+name,digest(ROOT/name)==value);check('snapshot '+name,digest(panel/'source_snapshot'/name)==value)
    pilot=ROOT/'results/iteration32_pilot';p=read(pilot/'summary.json')
    for name,value in p['source_hashes'].items():check('pilot snapshot '+name,digest(pilot/'source_snapshot'/name)==value)
    for name in ('tests/test_threshold_blocks.py','experiments/verify_iteration31.py'):
        dependencies[name]=digest(ROOT/name);dest=out/'source_snapshot'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,dest)
    def reuse(path,round_number):
        value=digest(ROOT/path);manifest=read(ROOT/f'results/iteration{round_number}_manifest.json');check('reused '+path,value==manifest['artifacts'][path]);dependencies[path]=value;return ROOT/path
    oldrows=read(reuse('results/iteration30_panel/rows.json',30));old29={};old30={}
    for j in range(16,22):old29[j]=np.load(reuse(f'results/iteration29_panel/case_{j:02d}.npz',29));old30[j]=np.load(reuse(f'results/iteration30_panel/case_{j:02d}.npz',30))
    independent={};maximum_effect_error=0.;effect_count=0
    for start in (0,18,54,140):
        block=np.load(panel/f'block_{start:03d}.npz');noise=block['noise'];independent[start]={}
        for mask in range(1,16):
            sites=[j for j in range(4) if mask&(1<<j)];E=bargmann_effect(noise[sites+[j+4 for j in sites]],(4,)*len(sites));stored=block[f'effect_{mask:02d}'];error=float(np.max(abs(E-stored)));maximum_effect_error=max(maximum_effect_error,error);effect_count+=1;independent[start][mask]=E
            check(f'block {start} Bargmann effect {mask}',error<3e-12,maximum_error=error)
        target=determinant_law(block['covariance']);check(f'block {start} Gaussian law',np.max(abs(target-block['target']))<2e-12)
    maximum_mps_law_error=0.;maximum_density_single_error=0.;maximum_prefix_error=0.;maximum_sample_z=0.;sample_calibration=[]
    for i,row in enumerate(rows):
        data=np.load(panel/f'case_{i:02d}.npz');law=data['law'];target=data['target'];check(f'{i} probability law domain',np.all(law>=0) and np.all(target>=0) and abs(math.fsum(law)-1)<1e-11 and abs(math.fsum(target)-1)<1e-11)
        tv=.5*math.fsum(abs(law-target));check(f'{i} TV',abs(tv-row['distances']['tv'])<2e-15)
        upper=.5*math.sqrt(math.fsum(((law-target)/(np.sqrt(law)+np.sqrt(target)))**2)*math.fsum((np.sqrt(law)+np.sqrt(target))**2));check(f'{i} Cauchy bound for this block',tv<=upper+2e-15 and abs(upper-row['distances']['hellinger_cauchy_upper_for_this_block'])<2e-15)
        if i<24:
            rho=data['density'];start=row['start'];eigen=np.linalg.eigvalsh(rho);check(f'{i} density domain',abs(np.trace(rho)-1)<1e-9 and np.max(abs(rho-rho.conj().T))<1e-9 and eigen[0]>-2e-10,minimum_eigenvalue=float(eigen[0]));vacua=np.ones(16)
            for mask in range(1,16):
                sites=[j for j in range(4) if mask&(1<<j)];r=reduced(rho,(4,)*4,sites);value=np.einsum('ij,ji->',r,independent[start][mask]);check(f'{i} real integral {mask}',abs(value.imag)<1e-12);vacua[mask]=value.real
            expected=inclusion(vacua);error=float(np.max(abs(expected-law)));maximum_mps_law_error=max(maximum_mps_law_error,error);check(f'{i} independent full MPS law',error<3e-12,maximum_error=error)
            old_index=16+i//4;check(f'{i} prior sampler same state',oldrows[old_index]['raw_mps_fingerprint']==row['raw_mps_fingerprint'])
            for j in range(4):
                error=float(np.max(abs(reduced(rho,(4,)*4,[j])-old29[old_index]['single_densities'][start+j])));maximum_density_single_error=max(maximum_density_single_error,error);check(f'{i} old single density {j}',error<2e-11)
            prior=0.
            for length in range(1,5):
                p=np.array([math.fsum(law[mask] for mask in range(16) if mask% (1<<length)==j) for j in range(1<<length)]);q=np.array([math.fsum(target[mask] for mask in range(16) if mask% (1<<length)==j) for j in range(1<<length)]);distance=.5*math.fsum(abs(p-q));check(f'{i} prefix {length}',abs(distance-data['prefix_tv'][length-1])<2e-15 and distance+2e-15>=prior);prior=distance
            samples=old30[old_index]['clicks'][:,start:start+4];hist=np.bincount(samples@(2**np.arange(4)),minlength=16);n=len(samples);z=(hist/n-law)/np.sqrt(law*(1-law)/n);maximum_sample_z=max(maximum_sample_z,float(np.max(abs(z))));sample_calibration.append(dict(case=i,observed_counts=hist.tolist(),maximum_standardized_error=float(np.max(abs(z)))))
        else:
            reference=determinant_law(data['covariance']);check(f'{i} direct Gaussian law',np.max(abs(reference-target))<2e-11);reconstructed=reconstruct(data['conditionals'][:,None],np.arange(8),0);check(f'{i} independent prefix law reconstruction',np.max(abs(reconstructed-law))<2e-13)
            order=data['full_measurement_order'];check(f'{i} explicit full ordering',np.array_equal(order[:8],np.arange(row['start'],row['start']+8)) and sorted(order.tolist())==list(range(144)))
            for j,record in enumerate(row['full_prefix_checks']):
                prefix=record['prefix'];index=(1<<len(prefix))-1+sum(bit<<k for k,bit in enumerate(prefix));error=abs(record['full_probability']-data['conditionals'][index]);maximum_prefix_error=max(maximum_prefix_error,error);check(f'{i} full prefix {j}',error<2e-11 and record['small_probability']==data['conditionals'][index])
        print('verified',i,flush=True)
    check('verifier unchanged',digest(source)==sha);save(out/'sample_calibration.json',sample_calibration)
    save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),independent_effects=effect_count,maximum_effect_error=maximum_effect_error,maximum_mps_law_error=maximum_mps_law_error,maximum_density_single_error=maximum_density_single_error,maximum_full_prefix_probability_error=maximum_prefix_error,maximum_old_sample_standardized_error=maximum_sample_z,dependencies=dependencies,verifier_sha256=sha,checks=checks));print(dict(count=len(checks),maximum_effect_error=maximum_effect_error,maximum_mps_law_error=maximum_mps_law_error,maximum_full_prefix_probability_error=maximum_prefix_error,maximum_old_sample_standardized_error=maximum_sample_z))
if __name__=='__main__':main()
