"""Independent PCO, global elimination, and exact stored-state overlap checks."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,math
from fractions import Fraction as F
import numpy as np
from iteration66_support import ROOT,snapshot,save,digest,constructions,encode
from iteration66_independent import verify_input
from verify_iteration34 import decode
from iteration24_reference import global_elimination
from gbslab.gaussian_schmidt_mps import projected_pair_tensor

def main():
    out=ROOT/'results/iteration66_gsvd_validation';hashes=snapshot(out);deps={};checks=[];rows=[]
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def ck(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration66_gsvd';p=read(base+'/plan.json');s=read(base+'/summary.json');ck('20 completed constructions',s['planned']==s['finished']==s['completed']==20);ck('frozen input panel',p['jobs']==constructions())
    for n,h in p['source_hashes'].items():ck('frozen '+n,digest(ROOT/n)==h==digest(ROOT/base/'source_snapshot'/n))
    for index,job in enumerate(p['jobs']):
        tag=f'case{index} ';rec=read(base+f'/case_{index:02d}.json');name=base+f'/case_{index:02d}.npz';deps[name]=digest(ROOT/name);z=np.load(ROOT/name);m=job['modes'];d=job['dimension'];rank=job['rank'];tensors=[z[f'tensor_{j}'] for j in range(m)];kernels=[z[f'kernel_{j}'] for j in range(m)];bases=[z[f'basis_{j}'] for j in range(m+1)];weights=[z[f'weights_{j}'] for j in range(m-1)];B=np.array(job['real'],float)+1j*np.array(job['imag'],float);info=rec['info'];ck(tag+'input and job identity',rec['job']==job and np.array_equal(z['pairing'],B));ck(tag+'stored layouts',all(a.flags.c_contiguous and list(a.strides)==stride for a,stride in zip(tensors,rec['stored_strides'])));ck(tag+'actual resource counts',info['max_bond_actual']==max(t.shape[2] for t in tensors) and info['stored_complex_elements']==sum(t.size for t in tensors));ck(tag+'backend identity',info['coefficient_backend']=='recurrence')
        ck(tag+'boundary bases',bases[0].shape==bases[-1].shape==(1,0));local=[]
        for j in range(m+1):
            b=bases[j];expected_modes=j if j<m else 0;ck(tag+f'basis dimensions {j}',b.ndim==2 and b.shape[1]==expected_modes and 1<=len(b)<=rank and np.all(b>=0));states={tuple(int(x) for x in row) for row in b};ck(tag+f'downward basis {j}',len(states)==len(b) and (0,)*expected_modes in states and all(tuple(x-int(k==axis) for k,x in enumerate(o)) in states for o in states for axis,value in enumerate(o) if value))
            if 0<j<m:
                q=info['cuts'][j-1]['parameters'];ck(tag+f'all virtual modes {j}',len(q)==j and all(0<=v<1 for v in q));exact_weights=[math.prod((1-F(float(v)))*F(float(v))**int(k) for v,k in zip(q,o)) for o in b];ck(tag+f'geometric weights {j}',len(weights[j-1])==len(b) and all(abs(float(e)-a)<1e-13 for e,a in zip(exact_weights,weights[j-1])));ck(tag+f'cut retained mass {j}',abs(info['cut_tail_estimates'][j-1]-max(0.,1-sum(float(e) for e in exact_weights)))<1e-13)
        for j,K in enumerate(kernels):
            count=bases[j].shape[1]+1+bases[j+1].shape[1];ck(tag+f'kernel shape symmetry {j}',K.shape==(count,count) and np.max(np.abs(K-K.T))<1e-12)
            A,pc=projected_pair_tensor(K,bases[j],d,bases[j+1]);factor=info['cuts'][j]['kernel_scale'] if j<m-1 else complex(z['final_scalar']);A*=factor;local.append(A);ck(tag+f'full local PCO versus recurrence {j}',A.shape==z[f'local_{j}'].shape and np.max(np.abs(A-z[f'local_{j}']))<1e-11);ck(tag+f'local scale {j}',abs(np.exp(z['log_scales'][j])-factor)<1e-13)
        raw=local[0][0]
        for a in local[1:]:raw=np.tensordot(raw,a,axes=(-1,0))
        raw=raw[...,0];rawnorm=float(np.vdot(raw,raw).real);ck(tag+'PCO projection success',abs(rawnorm-info['projection_success'])<1e-11)
        ind=verify_input(rec,tensors,ck,tag,out,index);actual=np.array([complex(float(a),float(b)) for a,b in ind['amplitudes']]).reshape((d,)*m)/math.sqrt(float(ind['norm']));pcostate=raw/math.sqrt(rawnorm);delta=float(np.max(np.abs(actual-pcostate)));ck(tag+'PCO to actual normalized tensor chain',delta<1e-11)
        C,elim=global_elimination(kernels,list(range(m))+[0]);err=float(np.max(np.abs(C-B)));ck(tag+'independent global input recovery',err<1e-10 and np.max(np.abs(C-z['folded']))<1e-10);logfactor=complex(z['folded_log_scale'])-sum(z['log_scales']);phase=complex(*elim['determinant_phase']);ck(tag+'global scalar squared',abs(np.exp(2*logfactor+elim['log_determinant_abs'])*phase-1)<1e-10)
        save(out/f'case_{index:02d}.maps.json',dict(elimination=elim,global_input_residual=err,local_PCO_full_state_max_error=delta,raw_norm=rawnorm));rows.append(dict(case=index,job=job,state_trace=rec['bridge']['full_trace_interval'],ideal_float_budget=info['construction_trace_budget'],actual_bond=info['max_bond_actual'],stored_elements=info['stored_complex_elements'],global_input_residual=err,PCO_state_error=delta));print('verified gsvd',index,flush=True)
    ck('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',rows);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),constructions=20,source_hashes=hashes,dependencies=deps,scope='Map recovery and PCO crosschecks are float64 diagnostics; actual stored-state versus rational Gaussian fidelity is independently interval-certified.'))
if __name__=='__main__':main()
