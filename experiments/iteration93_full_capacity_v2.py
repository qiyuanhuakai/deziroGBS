"""Frozen one-record capacity probes on the actual main spectrum.
Shared control record across configurations; never an independent quality batch.
Failures and incomplete photon records are retained, never called samples.
"""
import argparse
from iteration93_common_v3 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);lib,runtime=load93();save(out,'backend.json',lib._backend);z=np.load(ROOT/'results/iteration84_correlated_pnr_pilot_v1/input_0.npz');u=z['v'];t=-z['t'];alpha=np.zeros(50,complex);noise=z['noise_factor']/np.sqrt(2);configs=[(6,32,.02),(6,64,.02),(6,32,.1),(6,16,.2)];save(out,'plan.json',dict(scope=__doc__,configs=configs,records_each=1,experiment=0x93003004,sector=0,record=0,threads=1,cap=256,source_sha256=H(Path(__file__)),library_sha256=H(ROOT/'results/iteration93_build_v3/mps.dll')));rows=[]
for d,bond,budget in configs:
    name=f'{d}_{bond}_{budget}';save(out,f'started_{name}.json',dict(d=d,bond=bond,budget=budget));print('starting actual main spectrum',name,flush=True);raw,meta=sample93(lib,u,t,alpha,1,0x93003004,d=d,bond=bond,budget=budget,cap=256,noise=noise);np.savez_compressed(out/f'raw_{name}.npz',**raw);row=dict(d=d,bond=bond,budget=budget,photons=int(raw['counts'].sum()),trace_budget=float(raw['statistics'][0,5]),statistics=raw['statistics'][0].tolist(),**meta);rows.append(row);save(out,f'row_{name}.json',row);print(row,flush=True)
save(out,'summary.json',dict(completed=True,rows=rows,scope=__doc__));manifest(out)
