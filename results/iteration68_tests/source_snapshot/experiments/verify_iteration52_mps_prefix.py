"""Independent exact density law, exact physical covariance and CDF replay."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,math,gzip,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration45_independent import inverse,grid_below,mobius,tv
from iteration52_prefix_density import integer_density_law
from gbslab.local_mps import threshold_step,sample_local_threshold
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration52_mps_prefix_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration52_mps_prefix','iteration45_independent','iteration52_prefix_density','iteration22_external','iteration33_run','verify_iteration34')]+[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):
        deps[n]=digest(ROOT/n);raw=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(raw) if n.endswith('.gz') else raw))
    total_checks=0;passed_checks=0;shards=[]
    def check(n,ok):
        nonlocal total_checks,passed_checks
        total_checks+=1;passed_checks+=bool(ok);checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
        if len(checks)>=20000:
            name=f'checks_{len(shards):03d}.json';save(out/name,checks);shards.append(name);checks.clear()
    base='results/iteration52_mps';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('all20complete',summary['planned']==summary['finished']==summary['completed']==20)
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    targets={};physical_records={}
    for it in (46,47):
        validation=read(f'results/iteration{it}_input_validation/validation.json');check(f'{it} validated shared targets',validation['count']==validation['passed'])
        for i in range(4,8):
            name=f'results/iteration{it}_inputs/case_{i:02d}.json';record=read(name);check('bound target'+str((it,i)),deps[name]==validation['dependencies'][name]);physical_records[(it,i)]=record;targets[(it,i)]=record['target']['target_intervals']+[(F(0),F(0))]
    for index,job in enumerate(plan['jobs']):
        n=physical_records[(job['source_round'],job['case'])]['job']['modes'];tag=f'case{index} ';rec=read(base+f'/case_{index:02d}.json');name=base+f'/case_{index:02d}.npz';deps[name]=digest(ROOT/name);stored=np.load(ROOT/name);tensors=[stored[f'tensor_{j}'] for j in range(n)];masses,norm,scale,law=integer_density_law(tensors);r=rec['represented'];check(tag+'independent double layer',masses==r['integer_threshold_masses'] and norm==r['integer_norm'] and scale==r['amplitude_denominator'] and law==r['distribution'] and F(norm,scale**2)==r['represented_norm']);check(tag+'job',rec['job']==job);check(tag+'stored C layout',all(a.flags.c_contiguous and list(a.strides)==stride for a,stride in zip(tensors,rec['stored_strides'])))
        sampling=rec['sampling']
        for storage in sampling['trace_storage']:
            packed=(ROOT/base/storage['path']).read_bytes();raw=gzip.decompress(packed);check(tag+'lossless shard '+storage['path'],len(packed)==storage['compressed_bytes'] and len(raw)==storage['raw_bytes'] and hashlib.sha256(raw).hexdigest()==storage['raw_sha256'] and packed[4:8]==b'\x00'*4)
        check(tag+'storage inventory',[x['path'] for x in sampling['trace_storage']]==sampling['trace_shards']);sampling['trace']=[node for name in sampling['trace_shards'] for node in read(base+'/'+name)];trace={tuple(node['path']):node for node in sampling['trace']};check(tag+'unique nodes',len(trace)==len(sampling['trace'])==sampling['nodes']);real=[F(0)]*((1<<n)+1);word=[F(0)]*((1<<n)+1);queue=[((),0,np.ones(1,complex),F(1),F(1))];seen=set();leafcount=0
        for path,mask,left,rm,wm in queue:
            if len(path)==n:real[mask]+=rm;word[mask]+=wm;leafcount+=1;continue
            node=trace[path];seen.add(path);check(tag+'node '+str(path),node['mask']==mask and node['real_mass']==rm and node['word_mass']==wm and node['left_hex']==[[float(z.real).hex(),float(z.imag).hex()] for z in left]);branches,weights,total=threshold_step(left,tensors[len(path)],0j);p=weights/total;cdf=p.cumsum();cdf/=cdf[-1];check(tag+'native arithmetic '+str(path),node['status']=='completed' and node['weights_hex']==[float(x).hex() for x in weights] and node['total_hex']==total.hex() and node['probabilities_hex']==[float(x).hex() for x in p] and node['cdf_hex']==[float(x).hex() for x in cdf]);previous=F(0);previous_count=0;counts=[]
            for channel,edge in enumerate(cdf):
                e=F(float(edge));count=grid_below(e,53);width=e-previous;c=count-previous_count;counts.append(c);previous=e;previous_count=count;rchild=rm*width;wchild=wm*F(c,1<<53)
                if not rchild and not wchild:continue
                if weights[channel]<=0:real[-1]+=rchild;word[-1]+=wchild;continue
                queue.append((path+(channel,),mask|((channel!=0)<<len(path)),branches[channel]/math.sqrt(weights[channel]),rchild,wchild))
            check(tag+'independent grid counts '+str(path),counts==node['counts'] and previous==1 and previous_count==1<<53)
        check(tag+'complete sampling law',seen==set(trace) and leafcount==sampling['leaves'] and real==sampling['real_law'] and word==sampling['word_law'] and sum(real)==sum(word)==1)
        cal=rec['calibration'];rng=np.random.default_rng(cal['seed']);outputs,info=sample_local_threshold(tensors,np.zeros((cal['samples'],n),complex),rng);check(tag+'public sampler replay',outputs.tolist()==cal['outputs'] and len(cal['records'])==n*cal['samples']);rng=np.random.default_rng(cal['seed']);path=()
        for c in cal['records']:
            if len(path)==n:path=()
            node=trace[path];probs=np.array([float.fromhex(x) for x in node['probabilities_hex']]);channel=int(rng.choice(len(probs),p=probs));check(tag+'recorded choice '+str(len(path)),c['path']==list(path) and c['probabilities_hex']==node['probabilities_hex'] and c['channel']==channel);path+=(channel,)
        state=law+[F(0)];physical=targets[(job['source_round'],job['case'])];state_tv=tv(state,physical);sampler_tv=tv(word,physical);arithmetic=sum(abs(a-b) for a,b in zip(state,word))/2;quantization=sum(abs(a-b) for a,b in zip(real,word))/2;check(tag+'three separate errors',list(state_tv)==rec['state_physical_TV_interval'] and list(sampler_tv)==rec['sampler_physical_TV_interval'] and arithmetic==rec['arithmetic_TV'] and quantization==rec['quantization_TV']);rows.append(dict(case=index,job=job,state_TV=state_tv,sampler_TV=sampler_tv,arithmetic_TV=arithmetic,quantization_TV=quantization,failure=word[-1]));print('verified MPS',index,flush=True)
    timing=read(base+'/rows.json')
    for row in timing:
        times=list(row['clock_milestones'].items());check('valid clock decomposition '+str(row['sequence']),times[0][0]=='start' and times[-1][0]=='stored_result' and all(b[1]>=a[1] for a,b in zip(times,times[1:])) and row['seconds']==times[-1][1]-times[0][1] and row['phase_seconds']=={b[0]:b[1]-a[1] for a,b in zip(times,times[1:])} and abs(sum(row['phase_seconds'].values())-row['seconds'])<1e-8)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(rows));save(out/'checks_final.json',checks);shards.append('checks_final.json');save(out/'validation.json',dict(count=total_checks,passed=passed_checks,check_shards=shards,cases=20,source_hashes=hashes,dependencies=deps,numpy_version=np.__version__,scope='Exact stored binary64 tensors and independent uniform53-bit word model; not a fixed PRNG law'))
if __name__=='__main__':main()
