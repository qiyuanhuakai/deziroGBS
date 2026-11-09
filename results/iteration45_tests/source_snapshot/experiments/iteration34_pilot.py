import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import shutil,sys
from pathlib import Path
from fractions import Fraction as F
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.rational_gaussian import VacuumEnclosure
from gbslab.atom_certificate import selected_atoms
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel,NumericalProbabilityError
from iteration22_external import digest,save
from iteration33_run import encode

def main():
    out=ROOT/'results/iteration34_pilot';out.mkdir(exist_ok=False);names=['experiments/iteration34_pilot.py','experiments/iteration22_external.py','experiments/iteration33_run.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    paths=['results/iteration31_panel/large_input.npz','results/iteration31_build/libwire_forest.dll','results/iteration31_build/build.json'];dependencies={n:digest(ROOT/n) for n in paths};V=np.load(ROOT/paths[0])['covariance'];start=perf_counter();cert=VacuumEnclosure(V);certificate_seconds=perf_counter()-start;print('certificate seconds',certificate_seconds,'residual',float(cert.delta),flush=True)
    atoms={str(c):cert.atom(c) for c in ((),(0,),(54,),(0,1))};np.savez_compressed(out/'certificate.npz',covariance=V,R=cert.R);save(out/'certificate.json',encode(dict(diagnostics=cert.diagnostics,atoms=atoms)))
    m=144;q=F(1,513);table=[];start=perf_counter()
    with WireForestSampler(ThresholdModel((1+2**-8)*np.eye(2*m)),4,selection='recent',forest='chain',precision='auto',library=ROOT/paths[1]) as sampler:
        def oracle(prefix):
            try:value=F(sampler.conditional_click_probability(prefix));table.append(dict(prefix=prefix,q=value));return value
            except NumericalProbabilityError as e:table.append(dict(prefix=prefix,q=None,error=str(e)));return None
        patterns=[()]+[(j,) for j in range(0,144,18)];result=selected_atoms(m,patterns,oracle,lambda c:(q**len(c)*(1-q)**(m-len(c)),)*2);diagnostics=sampler.kernel.cache_info()
    native_seconds=perf_counter()-start;save(out/'native.json',encode(dict(result=result,oracle=table,diagnostics=diagnostics)));summary=dict(source_hashes=hashes,dependencies=dependencies,certificate_seconds=certificate_seconds,native_seconds=native_seconds,native_queries=result['queries'],native_upper=float(result['tv_upper']),unchanged_sources=hashes=={n:digest(ROOT/n) for n in names});save(out/'summary.json',summary);assert summary['unchanged_sources'];print({k:v for k,v in summary.items() if k not in ('source_hashes','dependencies')})
if __name__=='__main__':main()
