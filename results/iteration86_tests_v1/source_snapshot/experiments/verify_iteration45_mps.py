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
from gbslab.local_mps import threshold_step,sample_local_threshold
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration45_mps_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration45_mps','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration45_mps';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('all18complete',summary['planned']==summary['finished']==summary['completed']==18)
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    targets={}
    for strength,parameters in (('moderate',[F(1,4),F(3,8),F(1,8),F(1,4)]),('strong',[F(1,2),F(3,4),F(3,8),F(5,8)])):
        name=base+'/'+strength+'.npz';deps[name]=digest(ROOT/name);stored=np.load(ROOT/name);B=stored['B'];U=stored['U'];ur=np.array([[F(float(x.real)) for x in row] for row in U],dtype=object);ui=np.array([[F(float(x.imag)) for x in row] for row in U],dtype=object);check(strength+'unitary',np.array_equal(ur@ur.T+ui@ui.T,np.eye(4,dtype=object)) and not np.any(ui@ur.T-ur@ui.T));check(strength+'strict physical singular values',list(map(lambda x:F(float(x)),stored['parameters']))==parameters and all(0<x<1 for x in parameters));D=np.diag(np.array(parameters,dtype=object));br=ur@D@ur.T-ui@D@ui.T;bi=ur@D@ui.T+ui@D@ur.T;check(strength+'exact pairing',all(F(float(B[i,j].real))==br[i,j] and F(float(B[i,j].imag))==bi[i,j] for i in range(4) for j in range(4)))
        S=np.block([[br,bi],[bi,-br]]);inv=inverse(np.eye(8,dtype=object)-S);V=[[2*inv[i][j]-int(i==j) for j in range(8)] for i in range(8)];record=read(base+'/'+strength+'.target.json');vac=[None]*16;vac[0]=(F(1),F(1))
        for enclosure in record['enclosures']:
            modes=enclosure['modes'];mask=sum(1<<j for j in modes);ix=modes+[j+4 for j in modes];Q=[[(V[a][b]+int(a==b))/2 for b in ix] for a in ix];det=exact_determinant(Q);a,b=enclosure['vacuum_interval'];check(strength+'exact determinant root '+str(mask),det>0 and 0<=a<=b<=1 and a*a*det<=1<=b*b*det);check(strength+'unique subset '+str(mask),vac[mask] is None);vac[mask]=(a,b)
        check(strength+'all subsets',all(x is not None for x in vac));raw,law=mobius(vac,4);check(strength+'physical Mobius',encode(raw)==encode(record['law']['raw_intervals']) and encode(law)==encode(record['law']['intervals']));targets[strength]=law+[(F(0),F(0))]
    for index,job in enumerate(plan['jobs']):
        tag=f'case{index} ';rec=read(base+f'/case_{index:02d}.json');name=base+f'/case_{index:02d}.npz';deps[name]=digest(ROOT/name);stored=np.load(ROOT/name);tensors=[stored[f'tensor_{j}'] for j in range(4)];masses,norm,scale,law=integer_density_law(tensors);r=rec['represented'];check(tag+'independent double layer',masses==r['integer_threshold_masses'] and norm==r['integer_norm'] and scale==r['amplitude_denominator'] and law==r['distribution'] and F(norm,scale**2)==r['represented_norm']);check(tag+'job',rec['job']==job)
        sampling=rec['sampling'];trace={tuple(node['path']):node for node in sampling['trace']};check(tag+'unique nodes',len(trace)==len(sampling['trace'])==sampling['nodes']);real=[F(0)]*17;word=[F(0)]*17;queue=[((),0,np.ones(1,complex),F(1),F(1))];seen=set();leafcount=0
        for path,mask,left,rm,wm in queue:
            if len(path)==4:real[mask]+=rm;word[mask]+=wm;leafcount+=1;continue
            node=trace[path];seen.add(path);check(tag+'node '+str(path),node['mask']==mask and node['real_mass']==rm and node['word_mass']==wm and node['left_hex']==[[float(z.real).hex(),float(z.imag).hex()] for z in left]);branches,weights,total=threshold_step(left,tensors[len(path)],0j);p=weights/total;cdf=p.cumsum();cdf/=cdf[-1];check(tag+'native arithmetic '+str(path),node['status']=='completed' and node['weights_hex']==[float(x).hex() for x in weights] and node['total_hex']==total.hex() and node['probabilities_hex']==[float(x).hex() for x in p] and node['cdf_hex']==[float(x).hex() for x in cdf]);previous=F(0);previous_count=0;counts=[]
            for channel,edge in enumerate(cdf):
                e=F(float(edge));count=grid_below(e,53);width=e-previous;c=count-previous_count;counts.append(c);previous=e;previous_count=count;rchild=rm*width;wchild=wm*F(c,1<<53)
                if not rchild and not wchild:continue
                if weights[channel]<=0:real[-1]+=rchild;word[-1]+=wchild;continue
                queue.append((path+(channel,),mask|((channel!=0)<<len(path)),branches[channel]/math.sqrt(weights[channel]),rchild,wchild))
            check(tag+'independent grid counts '+str(path),counts==node['counts'] and previous==1 and previous_count==1<<53)
        check(tag+'complete sampling law',seen==set(trace) and leafcount==sampling['leaves'] and real==sampling['real_law'] and word==sampling['word_law'] and sum(real)==sum(word)==1)
        cal=rec['calibration'];rng=np.random.default_rng(cal['seed']);outputs,info=sample_local_threshold(tensors,np.zeros((cal['samples'],4),complex),rng);check(tag+'public sampler replay',outputs.tolist()==cal['outputs'] and len(cal['records'])==4*cal['samples']);rng=np.random.default_rng(cal['seed']);path=()
        for c in cal['records']:
            if len(path)==4:path=()
            node=trace[path];probs=np.array([float.fromhex(x) for x in node['probabilities_hex']]);channel=int(rng.choice(len(probs),p=probs));check(tag+'recorded choice '+str(len(path)),c['path']==list(path) and c['probabilities_hex']==node['probabilities_hex'] and c['channel']==channel);path+=(channel,)
        state=law+[F(0)];physical=targets[job['strength']];state_tv=tv(state,physical);sampler_tv=tv(word,physical);arithmetic=sum(abs(a-b) for a,b in zip(state,word))/2;quantization=sum(abs(a-b) for a,b in zip(real,word))/2;check(tag+'three separate errors',list(state_tv)==rec['state_physical_TV_interval'] and list(sampler_tv)==rec['sampler_physical_TV_interval'] and arithmetic==rec['arithmetic_TV'] and quantization==rec['quantization_TV']);rows.append(dict(case=index,job=job,state_TV=state_tv,sampler_TV=sampler_tv,arithmetic_TV=arithmetic,quantization_TV=quantization,failure=word[-1]));print('verified MPS',index,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),cases=18,source_hashes=hashes,dependencies=deps,numpy_version=np.__version__,scope='Exact stored binary64 tensors and independent uniform53-bit word model; not a fixed PRNG law'))
if __name__=='__main__':main()
