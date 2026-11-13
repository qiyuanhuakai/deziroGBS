"""Common-Philox-input fused vs materialized CUDA timing and full replay."""
from pathlib import Path
import os,json,hashlib,shutil,argparse,sys,time,statistics
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.native_philox_click import NativePhiloxClickSampler
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
libraries={'old':ROOT/'results/iteration73_philox_pilot_v1/philox.dll','fused':ROOT/'results/iteration74_fused_pilot_v1/philox.dll'};cpu=ROOT/'results/iteration72_cuda_pilot_v1/cpu.dll';runtime=[Path(shutil.which('g++')).parent,Path(shutil.which('nvcc')).parent]
panels=[(case,512,budget) for case in (3,4,5) for budget in (1,3,4,8,32,128)]+[('large',count,budget) for count in (512,4096) for budget in (1,4,8)]
save('plan.json',dict(driver_sha256=H(Path(__file__)),libraries={k:H(v) for k,v in libraries.items()},panels=panels,repetitions=3,seed_base=741001,scope=__doc__,acceptance='All clicks identical; conditional probability difference < 1e-8. Every output, including failures, retained. Wall time includes model setup but excludes DLL loading/common input reads.'))
rows=[];pairs=[]
for panel,(case,count,budget) in enumerate(panels):
    data=np.load(ROOT/('results/iteration70_archive/iteration70_frugal_large_v1_20261110/input.npz' if case=='large' else f'results/iteration71_branch_pilot_v1/input_{case}.npz'))
    x=data['x0'] if case=='large' else data['x'];mean=data['mean'];variance=data['variance']
    for rep in range(3):
        records={}
        for kind in (('old','fused') if rep%2==0 else ('fused','old')):
            label=f'{panel}_{rep}_{kind}';save(f'started_{label}.json',dict(case=case,count=count,budget=budget));start=time.perf_counter();row=dict(panel=panel,case=case,count=count,budget=budget,rep=rep,kind=kind,seed=741001+100*panel+rep)
            try:
                with NativePhiloxClickSampler(data['v'],x,cpu,gpu_library=libraries[kind],mean=mean,noise_variance=variance,branch_budget=budget,runtime_dirs=runtime) as model:r=model.sample(count,row['seed'])
                row.update(success=True,wall_seconds=time.perf_counter()-start,cuda_timings=r['cuda_timings'],statistics=r['statistics']);records[kind]=r;np.savez_compressed(OUT/f'records_{label}.npz',clicks=r['clicks'],probabilities=r['conditional_click_probabilities'],diagnostics=r['trajectory_diagnostics'])
            except Exception as e:row.update(success=False,wall_seconds=time.perf_counter()-start,error=str(e))
            rows.append(row);save(f'row_{label}.json',row)
        pair=dict(panel=panel,rep=rep,case=case,count=count,budget=budget,complete=len(records)==2)
        if len(records)==2:pair.update(mismatched_bits=int(np.count_nonzero(records['old']['clicks']!=records['fused']['clicks'])),maximum_probability_difference=float(abs(records['old']['conditional_click_probabilities']-records['fused']['conditional_click_probabilities']).max()),maximum_diagnostic_difference=float(abs(records['old']['trajectory_diagnostics']-records['fused']['trajectory_diagnostics']).max()))
        pairs.append(pair);save(f'pair_{panel}_{rep}.json',pair);print(pair,flush=True)
medians=[]
for panel,(case,count,budget) in enumerate(panels):
    r=dict(case=case,count=count,budget=budget)
    for kind in libraries:
        subset=[v for v in rows if v['panel']==panel and v['kind']==kind];r[kind]=statistics.median(v['wall_seconds'] for v in subset) if all(v['success'] for v in subset) else None
    medians.append(r)
passed=all(p['complete'] and p['mismatched_bits']==0 and p['maximum_probability_difference']<1e-8 for p in pairs)
save('summary.json',dict(completed=True,passed=passed,rows=rows,pairs=pairs,medians=medians));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}));print(json.dumps(dict(passed=passed,medians=medians)),flush=True)
