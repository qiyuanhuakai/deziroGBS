"""Lightweight resource/error audit of the optimized coherent-core proposal.
No sampling or tensor construction. Dropping squeezed eigenmodes means projecting
them to vacuum in this fixed basis; the pure-state trace distance is exact in real
arithmetic, but after noise/measurement it is only an upper error budget. This is
not a lower bound on the measured GBS distribution or on all possible algorithms.
"""
from pathlib import Path
import argparse,json,hashlib,math
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
base=ROOT/'results/iteration83_core_energy_probe_v2';files=[Path(__file__),base/'summary.json',base/'split_0.npz',base/'split_1.npz'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}));mp.mp.dps=80;reference=json.loads((base/'summary.json').read_text());rows=[]
for sector in range(2):
 t=np.load(base/f'split_{sector}.npz')['t'];z=sorted([mp.mpf(float(q))**2 for q in t],reverse=True);ranks=[]
 for rank in (0,1,2,4,8,16,24,32,40,48,49,50):
  vacuum=mp.fprod(mp.sqrt(1-q) for q in z[rank:]);distance=mp.sqrt(1-vacuum);ranks.append(dict(retained_squeezed_modes=rank,pure_core_projection_trace_distance=str(distance),interpretation='upper budget after exact common displacement/measurement; not measured TV lower bound'))
 dimensions=[]
 for K in (4,8,12,16,24,32):
  all_states=math.comb(len(t)+K,K);even_states=sum(math.comb(len(t)+k-1,k) for k in range(0,K+1,2));b=next(q for q in reference['rows'][sector]['budgets'] if q['budget']==K);dimensions.append(dict(photon_cutoff=K,all_number_states=all_states,even_initial_number_states=even_states,one_complex128_dense_vector_bytes=16*all_states,ideal_tv_upper=b['ideal_tv_upper']))
 row=dict(sector=sector,rank_projection=ranks,dense_resources=dimensions);rows.append(row);save(f'row_{sector}.json',row);print(dict(sector=sector,drop_weakest_mode_trace_distance=ranks[-2]['pure_core_projection_trace_distance'],K32_basis=dimensions[-1]['all_number_states']),flush=True)
joint=[]
for a,b in ((16,8),(24,12),(32,16),(48,24)):
 e0=mp.mpf(next(q['ideal_tv_upper'] for q in reference['rows'][0]['budgets'] if q['budget']==a));e1=mp.mpf(next(q['ideal_tv_upper'] for q in reference['rows'][1]['budgets'] if q['budget']==b));joint.append(dict(core_cutoffs=[a,b],conditional_ideal_joint_pnr_and_click_tv_upper=str(mp.sqrt(e0*e0+e1*e1-e0*e0*e1*e1)),requires='normalized coherent projections in both independent sectors and exact subsequent sampling; no numerical sampler error included'))
save('summary.json',dict(rows=rows,joint_budgets=joint,scope=__doc__));save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('joint',joint,flush=True)
