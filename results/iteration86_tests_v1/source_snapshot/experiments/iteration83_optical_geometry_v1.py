"""Lightweight input-only geometry audit of physical detector ordering.
Only linear-size row comparisons, no sampling or optimization on data records.
Tests the eight temporal-bin/144-spatial-port grouping hypothesis explicitly.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in [Path(__file__),inp]}));d=np.load(inp);T=np.zeros_like(d['T']);T[d['order']]=d['T'];rr=d['squeezing'];T*=np.repeat(np.sqrt(np.sum(np.sinh(rr)**2,axis=1)),2);norm=np.linalg.norm(T,axis=1);assert np.all(norm>0);unit=T/norm[:,None]
rows=[]
for shift in (1,8,144,288,576):
 overlap=abs(np.sum(unit[:-shift].conj()*unit[shift:],axis=1))**2;row=dict(shift=shift,mean=float(overlap.mean()),minimum=float(overlap.min()),maximum=float(overlap.max()),above_point999=int(np.count_nonzero(overlap>.999)),pairs=len(overlap));rows.append(row)
prefix=d['order'][:8].tolist();grouped=[i+144*j for i in range(144) for j in range(8)];grouped=prefix+[i for i in grouped if i not in prefix];grouped=np.array(grouped,np.int32);base=np.argsort(d['order'])[grouped];assert np.array_equal(base[:8],np.arange(8)) and np.array_equal(np.sort(base),np.arange(1152))
adj={}
for label,order in [('original',d['order']),('temporal_grouped',grouped)]:
 u=unit[order];overlap=abs(np.sum(u[:-1].conj()*u[1:],axis=1))**2;adj[label]=dict(mean=float(overlap.mean()),above_point999=int(np.count_nonzero(overlap>.999)))
save('summary.json',dict(shifts=rows,adjacency=adj,scope=__doc__));np.savez_compressed(OUT/'order.npz',order=base,physical_order=grouped);save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print(json.dumps(dict(shifts=rows,adjacency=adj)),flush=True)
