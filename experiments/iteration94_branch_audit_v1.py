"""Complete small click trees, independent Gaussian references and compression audit.
No empirical-iid confidence claim is made: complete trees include failure mass.
Paired sampling controls test worker/batch replay, not statistical independence.
"""
import argparse,itertools
import mpmath as mp
from iteration94_branch_common_v1 import *

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
lib,runtime=load_branch94();sources=[ROOT/f'results/iteration93_reference_v2/input_{j}.npz' for j in range(4)]
files=[Path(__file__),ROOT/'experiments/iteration94_branch_common_v1.py',ROOT/'native/gaussian_click_branch94_v1.cpp',ROOT/'native/gaussian_click_gram94_v1.cpp',ROOT/'native/gaussian_mps_core94_v2.cpp',ROOT/'results/iteration94_branch_build_v1/branch.dll',*sources]
save(out,'plan.json',dict(scope=__doc__,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},limits=[1,2,4,64],budgets=[.01,1.],compression_seed=940061,control_experiment_base=0x94006000,control_records=256))
cases=[]
for j,source in enumerate(sources):
    q,mean=gaussian94(np.load(source));target=np.load(ROOT/f'results/iteration93_reference_v2/law_{j}.npz')['click'];cases.append((['pure4','displaced4','pure8','lossy_tmsv'][j],q,mean,target))
q=np.eye(6);mean=np.zeros(6);target=np.zeros(8);target[0]=1;cases.append(('vacuum3',q,mean,target))
photons=np.array([.2,.7,1.3]);q=np.diag(np.r_[1+photons,1+photons]);p0=1/(1+photons);target=np.array([np.prod([p0[j] if not (mask>>j)&1 else 1-p0[j] for j in range(3)]) for mask in range(8)]);cases.append(('thermal3',q,np.zeros(6),target))
rows=[];issues=[]
for j,(name,q,mean,target) in enumerate(cases):
    np.savez_compressed(out/f'input_{j}.npz',q=q,mean=mean,target=target)
    for limit,budget in itertools.product((1,2,4,64),(.01,1.)):
        law,stats,seconds=branch_enumerate94(lib,q,mean,limit,budget);distance=float(np.sum(abs(law-np.r_[target,0]))/2);bound=min(1.,stats[0]+stats[1]);normerr=float(abs(law.sum()-1));negative=float(law.min())
        if normerr>3e-12 or negative<0 or distance>bound+3e-11:issues.append(dict(case=name,limit=limit,budget=budget,reason='normalization or total-variation budget',distance=distance,bound=bound))
        if limit==64 and len(mean)<=8 and (distance>3e-12 or stats[1]>0):issues.append(dict(case=name,reason='uncompressed exact small-law gate',distance=distance,failure=float(stats[1])))
        row=dict(case=name,limit=limit,budget=budget,full_click_tv=distance,weighted_charge=float(stats[0]),failure_mass=float(stats[1]),numeric_failure_mass=float(stats[2]),budget_failure_mass=float(stats[3]),ideal_arithmetic_tv_bound=bound,normalization_error=normerr,seconds=seconds)
        rows.append(row);save(out,f'row_{j}_{limit}_{budget}.json',row);np.savez_compressed(out/f'law_{j}_{limit}_{budget}.npz',law=law,statistics=stats);print(row,flush=True)
    raw,meta=branch_sample94(lib,q,mean,256,0x94006000+j,limit=4,budget=.1)
    np.savez_compressed(out/f'raw_{j}.npz',**raw);save(out,f'raw_{j}.json',meta)
    parts=[branch_sample94(lib,q,mean,count,0x94006000+j,limit=4,budget=.1,offset=offset,threads=4)[0] for offset,count in ((0,73),(73,183))]
    for key in raw:
        if not np.array_equal(raw[key],np.concatenate([part[key] for part in parts])):issues.append(dict(case=name,reason='worker/batch replay',key=key))

mp.mp.dps=80;rng=np.random.default_rng(940061);compression=[]
for case in (0,1,2):
    name,q,mean,_=cases[case];m=len(mean)//2;cache={}
    def vacuum(mask):
        if mask not in cache:
            modes=[j for j in range(m) if (mask>>j)&1];ids=modes+[j+m for j in modes]
            if not ids:v=mp.mpf(1)
            else:
                Q=mp.matrix([[float(q[a,b]) for b in ids] for a in ids]);mu=mp.matrix([float(mean[a]) for a in ids]);v=mp.exp(-(mu.T*mp.lu_solve(Q,mu))[0]/2)/mp.sqrt(mp.det(Q))
            cache[mask]=v
        return cache[mask]
    def overlap(a,b):return vacuum(a|b)/mp.sqrt(vacuum(a)*vacuum(b))
    for trial in range(3):
        masks=np.asarray(rng.choice(1<<m,size=12,replace=False),np.uint64).reshape(-1,1);coefficients=rng.normal(size=12);original=[int(v) for v in masks[:,0]];G=mp.matrix([[overlap(a,b) for b in original] for a in original]);x=mp.matrix([float(v) for v in coefficients]);norm=(x.T*G*x)[0]
        for limit in (1,2,4):
            result=branch_compress94(lib,q,mean,masks,coefficients,limit)
            if result['returncode']!=0:issues.append(dict(case=name,reason='compression probe failure',message=result['error']));continue
            rank=int(result['statistics'][12]);selected=[int(v) for v in result['masks'][:rank,0]];y=mp.matrix([float(v) for v in result['coefficients'][:rank]]);H=mp.matrix([[overlap(a,b) for b in selected] for a in selected]);cross=mp.matrix([[overlap(a,b) for b in selected] for a in original]);ynorm=(y.T*H*y)[0];inner=(x.T*cross*y)[0];square=1-inner**2/(norm*ynorm);actual=float(mp.sqrt(max(mp.mpf(0),square)));charged=float(result['statistics'][4]);difference=abs(actual-charged)
            if difference>5e-11:issues.append(dict(case=name,reason='80-digit projection trace distance',actual=actual,charged=charged))
            compression.append(dict(case=name,trial=trial,limit=limit,actual_trace_distance=actual,charged=charged,error=difference))
            np.savez_compressed(out/f'compression_{case}_{trial}_{limit}.npz',masks=masks,coefficients=coefficients,new_masks=result['masks'],new_coefficients=result['coefficients'],statistics=result['statistics'])
report=dict(passed=not issues,complete_trees=len(rows),compression_checks=len(compression),control_records=256*len(cases),maximum_compression_identity_error=max((r['error'] for r in compression),default=None),issues=issues,rows=rows,compression=compression)
save(out,'validation.json',report);manifest(out);print(json.dumps({k:v for k,v in report.items() if k not in ('rows','compression')}),flush=True);assert report['passed']
