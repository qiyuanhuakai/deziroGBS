"""Construct actual Gaussian-SVD maps and certify their stored finite outputs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import math,time,traceback
import numpy as np
from iteration66_support import ROOT,constructions,snapshot,save,digest
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.gaussian_kernel_chain import fold_kernel_chain
from gbslab.gaussian_fock_bridge import gaussian_fidelity
from gbslab.gaussian_pair_recurrence import pair_recurrence_tensor

def main():
    out=ROOT/'results/iteration66_gsvd';hashes=snapshot(out);jobs=constructions();save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies={},numpy_version=np.__version__));rows=[]
    for index,job in enumerate(jobs):
        start=time.perf_counter()
        try:
            B=np.array(job['real'],float)+1j*np.array(job['imag'],float);tensors,info,maps=build_schmidt_core(B,job['dimension'],job['rank'],return_maps=True,coefficient_backend='recurrence');tensors=[np.ascontiguousarray(t) for t in tensors];scales=[math.log(r['kernel_scale']) for r in info['cuts']]+[np.log(maps['final_scalar'])];folded,logscale,fold=fold_kernel_chain(maps['kernels'],list(range(job['modes']))+[0],log_scales=scales);bridge=gaussian_fidelity(tensors,job['real'],job['imag']);arrays=dict(pairing=B,folded=folded,final_scalar=np.array(maps['final_scalar']),log_scales=np.array(scales),folded_log_scale=np.array(logscale));arrays.update({f'tensor_{j}':t for j,t in enumerate(tensors)});arrays.update({f'kernel_{j}':k for j,k in enumerate(maps['kernels'])});arrays.update({f'basis_{j}':b for j,b in enumerate(maps['bases'])});arrays.update({f'weights_{j}':w for j,w in enumerate(maps['weights'])})
            for j,K in enumerate(maps['kernels']):
                local,_=pair_recurrence_tensor(K,maps['bases'][j],job['dimension'],maps['bases'][j+1]);local*=info['cuts'][j]['kernel_scale'] if j<job['modes']-1 else maps['final_scalar'];arrays[f'local_{j}']=local
            np.savez_compressed(out/f'case_{index:02d}.npz',**arrays);save(out/f'case_{index:02d}.json',dict(job=job,info=info,bridge=bridge,fold=fold,stored_strides=[t.strides for t in tensors],fold_residual=float(np.max(np.abs(folded-B)))));row=dict(case=index,family=job['family'],action=job['action'],status='completed',actual_bond=info['max_bond_actual'],stored_elements=info['stored_complex_elements'],state_trace=[float(x) for x in bridge['full_trace_interval']],ideal_float_budget=info['construction_trace_budget'])
        except Exception:row=dict(case=index,job=job,status='failed',traceback=traceback.format_exc());save(out/f'case_{index:02d}.failed.json',row)
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('gsvd',index,row['status'],flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=20,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),source_hashes=hashes,dependencies={},unchanged_sources=True))
if __name__=='__main__':main()
