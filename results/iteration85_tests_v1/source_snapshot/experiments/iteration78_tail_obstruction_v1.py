"""Numerical separating witness for every shared-intensity iid tail policy.
Same NB total law, budget12. A random p followed by N iid ports has PSD pair
matrix E[p p^T]. The target tail pair matrix has a negative quadratic form.
This checks a mathematical model obstruction numerically, not interval-certifies
all floating arithmetic or the binary64 input's exact physical isometry.
"""
from pathlib import Path
import os,json,argparse,hashlib,shutil
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
base=ROOT/'results/iteration78_strong_v1';d=np.load(base/'input.npz');v=d['v'];target=json.loads((base/'target.json').read_text());ca=np.longdouble(target['coefficient_A'])-np.longdouble(target['low_A']);cl=np.longdouble(target['coefficient_L'])-np.longdouble(target['low_L']);ell=(abs(v)**2).sum(axis=1);A=abs(v@v.T)**2;L=np.outer(ell,ell)+abs(v@v.conj().T)**2;T=float(ca)*A+float(cl)*L;values,vectors=np.linalg.eigh(T);u=vectors[:,0];u=u/max(abs(u));rayleigh=float(u@T@u)
V=v.astype(np.clongdouble);U=u.astype(np.longdouble);M=V.conj().T@(U[:,None]*V);qa=np.sum(M*M).real;ql=np.trace(M).real**2+np.sum(abs(M)**2);independent=ca*qa+cl*ql;assert abs(rayleigh-float(independent))<1e-16 and independent<0;bound=-float(independent)/2;residual=np.linalg.norm(T@vectors[:,0]-values[0]*vectors[:,0]);assert residual<1e-16
result=dict(passed=True,scope=__doc__,minimum_eigenvalue=float(values[0]),negative_eigenvalues=int((values<0).sum()),eigen_residual=float(residual),rayleigh=rayleigh,independent_input_space_rayleigh=str(independent),normalized_witness_max=float(abs(u).max()),ideal_model_numerical_full_tv_lower=bound,target_remainder_upper=target['omitted_mass_upper'],budget=12,inputs=v.shape[1],outputs=len(v),isometry_residual=float(np.linalg.norm(v.conj().T@v-np.eye(v.shape[1]))),witness='1{N>12} * (1 + [(u dot counts)^2-sum_i u_i^2 counts_i]/[N(N-1)])/2',no_full_interval_certificate=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
np.savez_compressed(OUT/'witness.npz',v=v,weights=u,tail_pair_matrix=T,eigenvalues=values,input_space_matrix=M,coefficients=np.array([ca,cl]));(OUT/'validation.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());shutil.copy2(__file__,OUT/Path(__file__).name);(OUT/'manifest.json').write_bytes((json.dumps({'artifacts':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}},indent=2)+'\n').encode());print(json.dumps(result),flush=True)
