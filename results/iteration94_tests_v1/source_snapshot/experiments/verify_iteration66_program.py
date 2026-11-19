"""Independent full fixed-displacement measurement and word-law certificates."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,math
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from iteration66_support import ROOT,snapshot,save,digest,constructions,displacement,encode
from iteration66_independent import verify_gaussian
from verify_iteration34 import decode
from iteration47_radicals import density_radicals
from iteration51_normal_independent import exp_negative
from iteration45_independent import mobius,tv
from iteration54_independent import trace_check
from gbslab.local_mps import sample_local_threshold
exp_negative=lru_cache(None)(exp_negative)

def main():
    out=ROOT/'results/iteration66_program_validation';hashes=snapshot(out);deps={};checks=[];rows=[];nodes=0
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def ck(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration66_program';p=read(base+'/plan.json');s=read(base+'/summary.json');v=read('results/iteration66_gsvd_validation/validation.json');ck('60 completed programs',s['planned']==s['completed']==s['finished']==60);ck('20 certified inputs',v['constructions']==20 and v['count']==v['passed']);expected=[dict(construction=i,displacement=z,**j) for i,j in enumerate(constructions()) for z in range(3)];ck('predeclared jobs',p['jobs']==expected)
    for n,h in p['source_hashes'].items():ck('frozen '+n,digest(ROOT/n)==h==digest(ROOT/base/'source_snapshot'/n))
    for index,job in enumerate(p['jobs']):
        tag=f'case{index} ';rec=read(base+f'/case_{index:02d}.json');ci=job['construction'];name=f'results/iteration66_gsvd/case_{ci:02d}.json';source=read(name);ck(tag+'certified input JSON',deps[name]==v['dependencies'][name]);name=f'results/iteration66_gsvd/case_{ci:02d}.npz';deps[name]=digest(ROOT/name);ck(tag+'certified input tensor bytes',deps[name]==v['dependencies'][name]);z=np.load(ROOT/name);n=job['modes'];tensors=[z[f'tensor_{j}'] for j in range(n)];alpha=displacement(job['displacement'],n);ck(tag+'job displacement',rec['job']==job and rec['alpha_hex']==[[a.real.hex(),a.imag.hex()] for a in alpha]);verify_gaussian(rec,source['bridge'],alpha,ck,tag)
        target=rec['finite_target'];ind=density_radicals(tensors,alpha);save(out/f'case_{index:02d}.radicals.json',ind);vac=[]
        for k in range(job['dimension']):
            a,b=target['factorial_root_intervals'][str(k)];f=math.factorial(k);ck(tag+'factorial root '+str(k),0<a<=b and a*a*f<=1<=b*b*f and b-a<=F(1,1<<192))
        for mask,r in enumerate(target['vacuum_records']):
            cl,ch=ind[mask]['coefficient_interval'];a,b=r['coefficient_interval'];ck(tag+'independent coefficient '+str(mask),a<=cl<=ch<=b);ex=sum(F(float(alpha[j].real))**2+F(float(alpha[j].imag))**2 for j in range(n) if mask>>j&1);el,eh=exp_negative(ex);l,h=r['exponential_interval'];ck(tag+'independent exponential '+str(mask),l<=el<=eh<=h and ex==r['exponent'] and r['mask']==mask);raw=(a*l,b*h);value=(max(F(0),raw[0]),min(F(1),raw[1]));ck(tag+'vacuum product '+str(mask),list(raw)==r['raw_vacuum_interval'] and value[0]<=cl*el<=ch*eh<=value[1]);vac.append(value)
        raw,law=mobius(vac,n);ck(tag+'complete finite law',encode(vac)==encode(target['vacuum_intervals']) and encode(raw)==encode(target['raw_intervals']) and encode(law)==encode(target['intervals']) and max(b-a for a,b in law)<F(1,10**40));word,num=trace_check(tensors,alpha,rec['sampling'],ck,tag);nodes+=num
        cal=rec['calibration'];ck(tag+'fixed seed count',cal['seed']==660000+index and cal['samples']==7);rng=np.random.default_rng(cal['seed']);outputs,_=sample_local_threshold(tensors,np.tile(alpha,(7,1)),rng);ck(tag+'public outputs replay',outputs.tolist()==cal['outputs'] and len(cal['records'])==7*n);trace={tuple(x['path']):x for x in rec['sampling']['trace']};rng=np.random.default_rng(cal['seed']);path=()
        for k,c in enumerate(cal['records']):
            if len(path)==n:path=()
            node=trace[path];probs=np.array([float.fromhex(x) for x in node['probabilities_hex']]);channel=int(rng.choice(len(probs),p=probs));ck(tag+'public choice '+str(k),c['path']==list(path) and c['probabilities_hex']==node['probabilities_hex'] and c['channel']==channel);path+=(channel,)
        arithmetic=tv(word,law+[(F(0),F(0))]);actual=tv(word,rec['gaussian_target']['intervals']+[(F(0),F(0))]);upper=min(F(1),source['bridge']['full_trace_interval'][1]+arithmetic[1]);ck(tag+'complete program budget',list(arithmetic)==rec['arithmetic_TV'] and list(actual)==rec['full_Gaussian_TV'] and upper==rec['total_TV_upper'] and actual[1]<=upper);rows.append(dict(case=index,job=job,nodes=num,arithmetic_TV=arithmetic,actual_TV=actual,total_upper=upper,failure=word[-1]));print('verified program',index,flush=True)
    ck('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',rows);shards=[]
    for i in range(0,len(checks),50000):name=f'checks_{i//50000:03d}.json';save(out/name,checks[i:i+50000]);shards.append(name)
    save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),check_shards=shards,cases=60,constructions=20,nodes=nodes,public_samples=420,source_hashes=hashes,dependencies=deps,scope='Complete2/3-mode threshold plus failure distributions,actual GSVD-produced binary64 tensors,independent uniform53-bit words,fixed displacements only.'))
if __name__=='__main__':main()
