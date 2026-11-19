"""Independent annihilation-recurrence replay of selected MPS amplitudes.

Uses no projected-pair exponentials. Every local virtual coefficient at the
selected physical occupations is independently computed and contracted.
Small cases replay the entire finite-box tensor.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
from time import perf_counter
import numpy as np
from iteration21_worker import save
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.mps_probe import mps_to_dense


def recurrence_slices(Q,left,right,occupations):
    """f(n)=sum_j Q_ij sqrt(n_j-delta_ij) f(n-e_i-e_j)/sqrt(n_i)."""
    groups=[np.asarray(left,int),np.asarray(right,int)]; tables=[]; nonzero=[]; vacuum=[]
    for basis in groups:
        lookup={tuple(n):i for i,n in enumerate(basis)}; vacuum.append(lookup[(0,)*basis.shape[1]])
        predecessors=np.full(basis.shape,-1,int); nz=[]
        for row,n in enumerate(basis):
            occupied=list(np.flatnonzero(n)); nz.append(occupied)
            for j in occupied:
                sub=n.copy(); sub[j]-=1; predecessors[row,j]=lookup[tuple(sub)]
        tables.append(predecessors); nonzero.append(nz)
    physical=left.shape[1]; rightoffset=physical+1
    @lru_cache(None)
    def coefficient(a,p,b):
        na=groups[0][a]; nb=groups[1][b]
        if (int(na.sum())+p+int(nb.sum()))%2: return 0j
        if a==vacuum[0] and p==0 and b==vacuum[1]: return 1.+0j
        if nonzero[0][a]:
            j=nonzero[0][a][0]; pivot=j; divisor=math.sqrt(na[j]); a=int(tables[0][a,j])
        elif p:
            pivot=physical; divisor=math.sqrt(p); p-=1
        else:
            j=nonzero[1][b][0]; pivot=rightoffset+j; divisor=math.sqrt(nb[j]); b=int(tables[1][b,j])
        value=0j
        for j in nonzero[0][a]: value+=Q[pivot,j]*math.sqrt(groups[0][a,j])*coefficient(int(tables[0][a,j]),p,b)
        if p: value+=Q[pivot,physical]*math.sqrt(p)*coefficient(a,p-1,b)
        for j in nonzero[1][b]: value+=Q[pivot,rightoffset+j]*math.sqrt(groups[1][b,j])*coefficient(a,p,int(tables[1][b,j]))
        return value/divisor
    slices={}
    for p in occupations:
        matrix=np.empty((len(left),len(right)),complex)
        for a in range(len(left)):
            for b in range(len(right)): matrix[a,b]=coefficient(a,p,b)
        slices[p]=matrix
    return slices,coefficient.cache_info().currsize


def main():
    p=argparse.ArgumentParser(); p.add_argument('--panel',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False); panel=args.panel; rows=json.loads((panel/'rows.json').read_text()); summaries=[]
    plan=json.loads((panel/'plan.json').read_text()); assert len(rows)==len(plan['jobs'])
    for row in rows:
        i=row['sequence']; start=perf_counter(); stem=panel/f'run_{i:02d}'; m=row['modes']; d=row['dimension']
        if row['status'] not in ('accepted','budget_missed'):
            summaries.append(dict(sequence=i,status='not_constructed',original_status=row['status'])); continue
        with np.load(stem.with_suffix('.npz')) as data: tensors=[data[f'A{k}'] for k in range(m)]
        with np.load(stem.with_suffix('.maps.npz')) as data:
            K=[data[f'K{k}'] for k in range(m)]; basis=[data[f'basis{k}'] for k in range(m+1)]; scalar=complex(data['final_scalar'])
        patterns=[np.zeros(m,int),np.eye(1,m,m//2,dtype=int)[0]*2,np.array([1]+[0]*(m-2)+[1],int)]
        raw=[np.ones(1,complex) for _ in patterns]; actual=[np.ones(1,complex) for _ in patterns]; all_tensors=[]; cache_entries=0
        for site in range(m):
            occ=list(range(d)) if row['panel']=='quality' else sorted({int(n[site]) for n in patterns})
            slices,count=recurrence_slices(K[site],basis[site],basis[site+1],occ); cache_entries+=count
            scale=row['info']['cuts'][site]['kernel_scale'] if site<m-1 else scalar
            for k,pattern in enumerate(patterns):
                raw[k]=raw[k]@(slices[int(pattern[site])]*scale); actual[k]=actual[k]@tensors[site][:,int(pattern[site]),:]
            if row['panel']=='quality': all_tensors.append(np.stack([slices[p]*scale for p in range(d)],axis=1))
        raw=np.array([x[0] for x in raw]); actual=np.array([x[0] for x in actual])*math.sqrt(row['info']['projection_success'])
        error=float(np.max(np.abs(raw-actual))); item=dict(sequence=i,status='complete',selected_amplitude_max_error=error,selected_amplitude_check=error<1e-10,cache_entries=cache_entries)
        np.savez_compressed(out/f'run_{i:02d}.npz',patterns=np.asarray(patterns),independent_raw_amplitudes=raw,saved_raw_amplitudes=actual)
        if row['panel']=='quality':
            independent=mps_to_dense(all_tensors); saved=mps_to_dense(tensors)*math.sqrt(row['info']['projection_success'])
            item.update(full_tensor_max_error=float(np.max(np.abs(independent-saved))),full_tensor_check=bool(np.max(np.abs(independent-saved))<1e-10),independent_success=float(np.linalg.norm(independent)**2))
        item['seconds']=perf_counter()-start; summaries.append(item); save(out/f'run_{i:02d}.json',item); save(out/'rows.json',summaries)
        print('reference',i,item['selected_amplitude_max_error'],item.get('full_tensor_max_error'),item['seconds'],flush=True)
    save(out/'rows.json',summaries)
    shutil.copy2(__file__,out/'iteration21_reference.py'); save(out/'metadata.json',dict(panel=panel.as_posix(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Annihilation recurrence, distinct from production projected pair creation'))


if __name__=='__main__': main()
