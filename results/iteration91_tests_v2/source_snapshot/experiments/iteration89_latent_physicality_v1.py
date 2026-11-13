"""A latent covariance need not satisfy the canonical uncertainty relation.
Analytic beam-splitter/no-click counterexamples distinguish a sufficient canonical
base filter from the necessary physicality test on remaining optical outputs.
No active quality sampler or its preregistered plan is changed.
"""
import argparse,mpmath as mp
from iteration89_common_v1 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);dll=ROOT/'results/iteration89_pilot_v1/physical.dll';files=[Path(__file__),dll,ROOT/'native/gaussian_click_physical89_v1.cpp'];save(out,'plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in files},digits=100));mp.mp.dps=100;runtime=os.add_dll_directory(str(runtime_dirs()[0]));lib=C.CDLL(str(dll));fn=lib.tc89_physical_probe;fn.argtypes=[C.c_int,DP,DP,C.c_char_p,C.c_int];rows=[]
def minimum(q,p):return (q+p+1-mp.sqrt((q-p)**2+1))/2
for r in map(mp.mpf,('.2','.6','.9')):
 for measured in map(mp.mpf,('.1','.5','.9')):
  for noise in map(mp.mpf,('0','.01')):
   q=-r/(1+r)+noise;p=r/(1-r)+noise;ql=q/(1+measured*q);pl=p/(1+measured*p);qr=(1-measured)*ql;pr=(1-measured)*pl;initial=minimum(q,p);latent=minimum(ql,pl);remaining=minimum(qr,pr);reference=-((1-measured)*r)/(1+(1-measured)*r);assert initial>=-mp.mpf('1e-95') and remaining>=-mp.mpf('1e-95')
   if noise==0:assert abs(qr-reference)<mp.mpf('1e-95')
   outcomes=[]
   for name,a0,b0 in (('latent',ql,pl),('remaining',qr,pr)):
    x=np.diag([float(a0),float(b0)]);stats=np.zeros(4);err=C.create_string_buffer(4096);rc=fn(2,ptr(x),ptr(stats),err,len(err));assert rc==0,err.value;outcomes.append(dict(which=name,accepted=bool(stats[0]),shift=float(stats[1]),residual=float(stats[2])))
   if noise>0:assert outcomes[1]['accepted']
   rows.append(dict(squeezing=str(r),measured_fraction=str(measured),added_classical_variance=str(noise),initial_uncertainty_minimum=mp.nstr(initial,90),latent_uncertainty_minimum=mp.nstr(latent,90),remaining_uncertainty_minimum=mp.nstr(remaining,90),native=outcomes))
save(out,'summary.json',dict(passed=True,rows=rows,scope=__doc__,conclusion='X+I/2+iOmega/2 positive is sufficient for a canonical latent Gaussian followed by a contractive remaining optical map, but is not necessary for physical remaining outputs. The89 interior filter is deliberately conservative. The88 negative conditional probability remains an independent failure witness; its latent canonical eigenvalue must not be called the remaining-output uncertainty eigenvalue.'));manifest(out);print(json.dumps(rows[9],indent=2),flush=True)
