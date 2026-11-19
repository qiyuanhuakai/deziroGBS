"""Small capacity controls on both actual Gaussian spectra; all failures retained.
Identical addresses across branch limits are deliberate paired resource probes.
This pilot does not provide independent-quality confidence intervals.
"""
import argparse
from iteration94_branch_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);lib,runtime=load_branch94()
sources=[ROOT/f'results/iteration93_actual_quality_v1/input_{j}.npz' for j in (0,1)]
files=[Path(__file__),ROOT/'experiments/iteration94_branch_common_v1.py',ROOT/'native/gaussian_click_branch94_v1.cpp',ROOT/'results/iteration94_branch_build_v1/branch.dll',*sources]
save(out,'plan.json',dict(scope=__doc__,records=4,limits=[4,8],budgets=[.1,.01],threads=4,experiment=0x94007000,hashes={f.relative_to(ROOT).as_posix():H(f) for f in files}));rows=[]
for sector,source in enumerate(sources):
    start=time.perf_counter();q,mean=gaussian94(np.load(source));prep=time.perf_counter()-start
    for limit in (4,8):
        print('starting actual',sector,limit,flush=True)
        raw,meta=branch_sample94(lib,q,mean,4,0x94007000,limit=limit,budget=(.1,.01)[sector],sector=sector,threads=4)
        np.savez_compressed(out/f'raw_{sector}_{limit}.npz',**raw)
        row=dict(sector=sector,limit=limit,preparation_seconds=prep,processed_modes=raw['statistics'][:,0].tolist(),clicks_so_far=raw['clicks'].sum(1).tolist(),charged=raw['statistics'][:,4].tolist(),attempted_max_charge=raw['statistics'][:,5].tolist(),**meta);rows.append(row);save(out,f'row_{sector}_{limit}.json',row);print(json.dumps(row),flush=True)
save(out,'summary.json',dict(completed=True,scope=__doc__,rows=rows));manifest(out)
