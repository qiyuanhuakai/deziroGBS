"""Fixed baseline feasibility checks, never reused as quality records."""
import argparse,subprocess,zipfile
from iteration92_pe_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip();assert head=='6c1ed6f8f11ffc9f3c268757f0fe0bd48a768d82';files=sorted((UPSTREAM/'photonic_engines').rglob('*.py'));save(out,'plan.json',dict(commit=head,version='0.2.2',torch=torch.__version__,device='cpu',threads=1,upstream_hashes={p.relative_to(UPSTREAM).as_posix():H(p) for p in files},driver_sha256=H(Path(__file__))));
with zipfile.ZipFile(out/'upstream_source.zip','x',zipfile.ZIP_DEFLATED) as archive:
    for path in files+[UPSTREAM/'LICENSE',UPSTREAM/'pyproject.toml',UPSTREAM/'docs/GAUSSIAN.md']:archive.write(path,path.relative_to(UPSTREAM))
rows=[]
for j,r in enumerate((.6,.85)):
    u=np.eye(2,dtype=complex);state,U=setup_pe(u,r);counts,stats,meta=run_pe(state,U,128,0x92003000+j);np.savez_compressed(out/f'raw_{j}.npz',counts=counts);save(out,f'stats_{j}.json',vars(stats));rows.append(dict(r=r,records=128,**meta));print(rows[-1],flush=True)
save(out,'summary.json',dict(passed=all(r['censored']==0 for r in rows),rows=rows));manifest(out)
