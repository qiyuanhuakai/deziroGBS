"""Independent exact density law, exact physical covariance and CDF replay."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration45_independent import inverse,integer_density_law,grid_below,mobius,tv
from iteration46_displacement_independent import exp_inverse_positive,density_vacuum_coefficient
from gbslab.local_mps import threshold_step,sample_local_threshold
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration46_displacement_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration46_displacement','iteration46_displacement_independent','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration46_displacement';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('all24complete',summary['planned']==summary['finished']==summary['completed']==24)
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for index,job in enumerate(plan['jobs']):
        n=job['modes'];tag=f'case{index} ';rec=read(base+f'/case_{index:02d}.json');name=base+f'/case_{index:02d}.npz';deps[name]=digest(ROOT/name);stored=np.load(ROOT/name);alpha=stored['alpha'];tensors=[stored[f'tensor_{j}'] for j in range(n)];check(tag+'job',rec['job']==job);check(tag+'stored C layout',all(a.flags.c_contiguous and list(a.strides)==stride for a,stride in zip(tensors,rec['stored_strides'])));target=rec['target'];norm=density_vacuum_coefficient(tensors,alpha,0);check(tag+'positive represented norm',norm>0);vac=[]
        for mask,r in enumerate(target['vacuum_records']):
            coefficient=density_vacuum_coefficient(tensors,alpha,mask)/norm;exponent=sum(F(float(alpha[j].real))**2+F(float(alpha[j].imag))**2 for j in range(n) if mask>>j&1);lo,hi=exp_inverse_positive(exponent);a,b=r['exponential_interval'];check(tag+'independent exponential '+str(mask),a<=lo<=hi<=b);check(tag+'independent double layer '+str(mask),r['mask']==mask and r['coefficient']==coefficient and r['exponent']==exponent);vac.append((coefficient*a,coefficient*b))
        check(tag+'vacuum intervals',encode(vac)==encode(target['vacuum_intervals']));raw,law=mobius(vac,n);check(tag+'independent displaced law',encode(raw)==encode(target['raw_intervals']) and encode(law)==encode(target['intervals']));check(tag+'rigorous target width',max(b-a for a,b in law)<F(1,10**200))
        sampling=rec['sampling'];trace={tuple(node['path']):node for node in sampling['trace']};check(tag+'unique nodes',len(trace)==len(sampling['trace'])==sampling['nodes']);real=[F(0)]*((1<<n)+1);word=[F(0)]*((1<<n)+1);queue=[((),0,np.ones(1,complex),F(1),F(1))];seen=set();leafcount=0
        for path,mask,left,rm,wm in queue:
            if len(path)==n:real[mask]+=rm;word[mask]+=wm;leafcount+=1;continue
            node=trace[path];seen.add(path);check(tag+'node '+str(path),node['mask']==mask and node['real_mass']==rm and node['word_mass']==wm and node['left_hex']==[[float(z.real).hex(),float(z.imag).hex()] for z in left]);branches,weights,total=threshold_step(left,tensors[len(path)],alpha[len(path)]);p=weights/total;cdf=p.cumsum();cdf/=cdf[-1];check(tag+'native arithmetic '+str(path),node['status']=='completed' and node['weights_hex']==[float(x).hex() for x in weights] and node['total_hex']==total.hex() and node['probabilities_hex']==[float(x).hex() for x in p] and node['cdf_hex']==[float(x).hex() for x in cdf]);previous=F(0);previous_count=0;counts=[]
            for channel,edge in enumerate(cdf):
                e=F(float(edge));count=grid_below(e,53);width=e-previous;c=count-previous_count;counts.append(c);previous=e;previous_count=count;rchild=rm*width;wchild=wm*F(c,1<<53)
                if not rchild and not wchild:continue
                if weights[channel]<=0:real[-1]+=rchild;word[-1]+=wchild;continue
                queue.append((path+(channel,),mask|((channel!=0)<<len(path)),branches[channel]/math.sqrt(weights[channel]),rchild,wchild))
            check(tag+'independent grid counts '+str(path),counts==node['counts'] and previous==1 and previous_count==1<<53)
        check(tag+'complete sampling law',seen==set(trace) and leafcount==sampling['leaves'] and real==sampling['real_law'] and word==sampling['word_law'] and sum(real)==sum(word)==1)
        cal=rec['calibration'];rng=np.random.default_rng(cal['seed']);outputs,info=sample_local_threshold(tensors,np.tile(alpha,(cal['samples'],1)),rng);check(tag+'public sampler replay',outputs.tolist()==cal['outputs'] and len(cal['records'])==n*cal['samples']);rng=np.random.default_rng(cal['seed']);path=()
        for c in cal['records']:
            if len(path)==n:path=()
            node=trace[path];probs=np.array([float.fromhex(x) for x in node['probabilities_hex']]);channel=int(rng.choice(len(probs),p=probs));check(tag+'recorded choice '+str(len(path)),c['path']==list(path) and c['probabilities_hex']==node['probabilities_hex'] and c['channel']==channel);path+=(channel,)
        value=tv(word,law+[(F(0),F(0))]);check(tag+'measurement TV interval',list(value)==rec['TV_to_exact_measurement_interval']);rows.append(dict(case=index,job=job,TV=value,maximum_target_width=max(b-a for a,b in law),failure=word[-1]));print('verified displacement',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),cases=24,source_hashes=hashes,dependencies=deps,numpy_version=np.__version__,scope='Exact stored binary64 tensors and independent uniform53-bit word model; not a fixed PRNG law'))
if __name__=='__main__':main()
