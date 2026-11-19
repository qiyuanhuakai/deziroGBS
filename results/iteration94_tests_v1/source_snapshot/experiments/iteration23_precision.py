"""Controlled storage precision and multiplication-order diagnostics."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
from time import perf_counter
import traceback
import zipfile
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from gbslab.mps_overlap import compare_mps
from iteration22_external import ARCHIVE_SHA,digest,load_entry,save
from iteration22_global import load_family


def family(name,archive):
    if name in ('pco128','pco64','hafnian_promoted'):
        original='author_hafnian' if name=='hafnian_promoted' else 'author_pco'
        tensors,entries=load_family(original,ROOT/'results/iteration22_external',archive)
        if name=='pco64': tensors=[a.astype(np.complex64).astype(np.complex128) for a in tensors]
        return tensors,entries
    if name!='hafnian_single_product': raise ValueError(name)
    result=[]; entries=[]; base='PCO mapping/data/d_4_chi_200/'
    with zipfile.ZipFile(archive) as z:
        for i in range(144):
            a=load_entry(z,base+f'Gamma_{i}.npy',entries).astype(np.complex64).transpose(0,2,1)
            if i<143:
                lam=load_entry(z,base+f'Lambda_{i}.npy',entries).astype(np.float32)
                a=np.multiply(a,lam[None,None,:],dtype=np.complex64)
            a=a.astype(np.complex128)
            if i==0: a=a[:1,:,:]
            if i==143: a=a[:,:,:1]
            result.append(a)
    return result,entries


def main():
    p=argparse.ArgumentParser(); p.add_argument('--archive',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    files=['experiments/iteration23_precision.py','experiments/iteration22_external.py','experiments/iteration22_global.py','src/gbslab/mps_overlap.py']
    hashes={f:digest(ROOT/f) for f in files}; (out/'sources').mkdir()
    for f in files: shutil.copy2(ROOT/f,out/'sources'/Path(f).name)
    pairs=['pco128:pco64','hafnian_promoted:hafnian_single_product','pco128:hafnian_single_product','pco64:hafnian_single_product']
    save(out/'plan.json',dict(pairs=pairs,source_hashes=hashes,archive_sha256=ARCHIVE_SHA,scope='Fixed stored MPS and deterministic cast/product transformations. Cannot reconstruct unknown pre-storage hafnian tensors or isolate all original arithmetic errors.',supplementary_inventory='All S_, res_, num_ npy members: metadata and hashes only; semantics unassigned.'))
    assert digest(args.archive)==ARCHIVE_SHA
    rows=[]
    for pair in pairs:
        started=perf_counter(); row=dict(pair=pair)
        try:
            left,right=pair.split(':'); a,ea=family(left,args.archive); b,eb=family(right,args.archive)
            save(out/(left+'_inputs.json'),ea); save(out/(right+'_inputs.json'),eb)
            row.update(compare_mps(a,b)); row['status']='complete'; del a,b
        except Exception: row.update(status='error',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-started; rows.append(row); save(out/'rows.json',rows)
        print(pair,row.get('pure_trace_distance_from_difference'),row['status'],flush=True)
    inventory=[]
    with zipfile.ZipFile(args.archive) as z:
        names=sorted(n for n in z.namelist() if n.startswith('PCO mapping/data/d_4_chi_200/') and Path(n).name.startswith(('S_','res_','num_')) and n.endswith('.npy'))
        for name in names:
            entries=[]; a=load_entry(z,name,entries); entry=entries[0]
            entry.update(finite=bool(np.isfinite(a).all()),minimum=None if not a.size else float(np.min(a)),maximum=None if not a.size else float(np.max(a)))
            inventory.append(entry)
    save(out/'supplementary_inventory.json',inventory)
    assert hashes=={f:digest(ROOT/f) for f in files}
    save(out/'summary.json',dict(completed=sum(r['status']=='complete' for r in rows),pairs=len(rows),supplementary_arrays=len(inventory),unchanged_sources=True))


if __name__=='__main__': main()
