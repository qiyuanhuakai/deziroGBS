"""Re-evaluate the frozen88 failure in its actual remaining optical subspace.
The selected rows are the retained clicks and every unmeasured output. SVD
compresses this contractive map to its non-vacuum subspace; no eigenvalue of the
raw latent matrix is substituted for the remaining-output uncertainty matrix.
"""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,hashlib
import numpy as np
import mpmath as mp
ROOT=Path(__file__).resolve().parents[1]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);inp=ROOT/'results/iteration83_loss_pilot_v1/input.npz';statepath=ROOT/'results/iteration88_failure_diagnostic_v1/state_project16_1_685.npz';files=[Path(__file__),inp,statepath];(out/'plan.json').write_text(json.dumps(dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files}),indent=2)+'\n');v=np.load(inp)['v'];z=np.load(statepath);x=z['covariance'];k=int(z['statistics'][0]);mode=int(z['ports'][k]);ports=np.r_[z['ports'][:k],np.arange(mode,len(v))];W=v[ports];U,s,Vh=np.linalg.svd(W,full_matrices=False);A=s[:,None]*Vh;R=np.block([[A.real,-A.imag],[A.imag,A.real]]);effective=R@x@R.T;n=len(s);d=2*n;omega=np.block([[np.zeros((n,n)),np.eye(n)],[-np.eye(n),np.zeros((n,n))]]);Q=effective+np.eye(d)/2+1j*omega/2;values,vectors=np.linalg.eigh(Q);w=vectors[:,0];mp.mp.dps=80;value=mp.mpf(0)
 for i in range(d):
  for j in range(d):value+=mp.conj(mp.mpc(w[i]))*mp.mpc(Q[i,j])*mp.mpc(w[j])
 row=dict(passed=True,retained_ports=z['ports'][:k].tolist(),next_mode=mode,selected_outputs=len(ports),minimum_singular_value=float(s.min()),maximum_singular_value=float(s.max()),map_reconstruction_error=float(np.linalg.norm(U@A-W)),remaining_uncertainty_minimum=float(values[0]),independent80_digit_rayleigh_real=mp.nstr(value.real,70),independent80_digit_rayleigh_imag=mp.nstr(value.imag,25),latent_canonical_minimum=float(np.linalg.eigvalsh(x+np.eye(d)/2+1j*omega/2).min()),scope=__doc__,limits='SVD/eigenvectors and state are floating-point; the80-digit Rayleigh check is on the exported effective matrix. It is a numerical negative witness, not an interval around the exact target state.');np.savez_compressed(out/'state.npz',selected_ports=ports,optical=W,compressed_optical=A,effective_covariance=effective,witness=w);(out/'summary.json').write_text(json.dumps(row,indent=2)+'\n');(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(json.dumps(row,indent=2),flush=True)
if __name__=='__main__':main()
