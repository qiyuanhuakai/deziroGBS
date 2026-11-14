"""Adapter gates on the frozen inputs; controls excluded from quality datasets."""
import argparse
from iteration93_baseline_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);plan=read(ROOT/'research/iteration93_quality_plan_v1.json');save(out,'plan.json',dict(scope=__doc__,records_each=32,source_sha256=H(Path(__file__)),adapter_sha256=H(ROOT/'experiments/iteration93_baseline_common_v1.py')));rows=[]
for job in [j for j in plan['jobs'] if j['family']!='mps']:
    q=np.load(ROOT/f"results/iteration93_reference_v2/input_{job['case']}.npz");model=Baseline93(job,q,threads=1)
    try:raw,meta=model.run(32,0,0x93008000+job['id'],0x93009000+job['id'])
    finally:model.close()
    np.savez_compressed(out/f"raw_{job['id']}.npz",**raw);row=dict(job=job,setup=model.meta,**meta);rows.append(row);save(out,f"row_{job['id']}.json",row);assert meta['failures']==0,row;print(job['name'],meta,flush=True)
save(out,'validation.json',dict(passed=True,rows=rows));manifest(out)
