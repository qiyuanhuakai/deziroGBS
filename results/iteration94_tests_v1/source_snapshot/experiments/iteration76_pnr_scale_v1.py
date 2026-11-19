"""Three repeated old/projected PNR timings, plus weak-energy high-input capacity.
The projected v1/v3 versions implement the same ideal approximation. Old prefix
resource rejections are recorded separately and never counted as timing wins.
"""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import mpmath as mp
mp.mp.dps=80
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_projected_pnr import NativeProjectedPNRSampler
from gbslab.native_prefix_pnr import NativePrefixPNRSampler
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
runtime=[Path(shutil.which('g++')).parent];methods={'old':(NativePrefixPNRSampler,ROOT/'results/iteration75_prefix_pilot_v1/prefix.dll'),'projected1':(NativeProjectedPNRSampler,ROOT/'results/iteration76_projected_pilot_v1/projected.dll'),'projected3':(NativeProjectedPNRSampler,ROOT/'results/iteration76_projected_pilot_v3/projected.dll')};save('plan.json',dict(scope=__doc__,source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),*[p for _,p in methods.values()]]},input_seed=760401,seed_base=0x83000000000000,seed_stride=2**40,workers=16,repetitions=3));shutil.copy2(__file__,OUT/Path(__file__).name);rows=[];targets=[];rejects=[];replays=[]
for case,(m,n,k,count) in enumerate([(144,2,31,4096),(144,32,7,512),(192,64,7,512),(384,128,7,512)]):
    if case==0:
        data=np.load(ROOT/'results/iteration75_lowrank_v2/input.npz');v,t=data['v'],data['t'];kinds=list(methods)
    else:
        rng=np.random.default_rng(760400+case);v=np.ascontiguousarray(np.linalg.qr(rng.normal(size=(m,n))+1j*rng.normal(size=(m,n)))[0]);t=np.full(n,np.sqrt(1/(n+1)));kinds=['projected1','projected3']
    shape=mp.mpf(n)/2;z=mp.mpf(float(t[0]))**2;prob=(1-z)**shape;mass=prob
    for pairs in range(1,k//2+1):prob*=z*(shape+pairs-1)/pairs;mass+=prob
    target=dict(case=case,inputs=n,outputs=m,retained=k,ideal_prefix_tv_upper=str(1-mass),target_mean=str(n*z/(1-z)),scope='Equal pure squeezing, exact isometry, real arithmetic and ideal draws; no complete floating/PRNG certificate.');targets.append(target);save(f'target_{case}.json',target);np.savez_compressed(OUT/f'input_{case}.npz',v=v,t=t)
    if case:
        try:
            with NativePrefixPNRSampler(v,t,methods['old'][1],retained=k,workers=16,runtime_dirs=runtime) as model:model.sample(1,0)
            rejected=False;error=''
        except ArithmeticError as e:rejected=True;error=str(e)
        row=dict(case=case,rejected=rejected,error=error);rejects.append(row);save(f'old_resource_{case}.json',row);assert rejected and 'resource estimate' in error,row
    for kind in kinds:
        cls,binary=methods[kind]
        with cls(v,t,binary,retained=k,workers=16,runtime_dirs=runtime) as model:model.sample(1,0)
    for rep in range(3):
        seed=0x83000000000000+(16*case+rep)*2**40;values={}
        for kind in kinds[rep%len(kinds):]+kinds[:rep%len(kinds)]:
            label=f'{case}_{rep}_{kind}';save(f'started_{label}.json',dict(case=case,rep=rep,kind=kind,seed=seed));cls,binary=methods[kind];start=time.perf_counter()
            with cls(v,t,binary,retained=k,workers=16,runtime_dirs=runtime) as model:r=model.sample(count,seed)
            counts=r['counts'];row=dict(case=case,rep=rep,kind=kind,seed=seed,records=count,inputs=n,outputs=m,retained=k,wall_seconds=time.perf_counter()-start,statistics=r['statistics'],odd_records=int((counts.sum(axis=1)%2!=0).sum()));values[kind]=counts;rows.append(row);np.savez_compressed(OUT/f'counts_{label}.npz',counts=counts);save(f'row_{label}.json',row);print(label,row['wall_seconds'],flush=True)
        reference=values['projected3']
        for kind,counts in values.items():
            different=int(np.any(counts!=reference,axis=1).sum());replays.append(dict(case=case,rep=rep,kind=kind,different_records=different));assert different==0,(case,rep,kind,different)
medians={str(case):{kind:statistics.median(r['wall_seconds'] for r in rows if r['case']==case and r['kind']==kind) for kind in sorted(set(r['kind'] for r in rows if r['case']==case))} for case in range(4)}
save('summary.json',dict(passed=True,rows=rows,medians=medians,targets=targets,old_resource_rejections=rejects,replays=replays));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
