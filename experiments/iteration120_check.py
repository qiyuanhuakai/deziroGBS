"""Complex-valued contraction tests independent of the compiled arithmetic graph."""
import json
from time import perf_counter
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from iteration119_records import brute
from iteration120_contractions import amplitudes
from iteration120_replay import OUT

def run():
    rng=np.random.default_rng(120141)
    u=np.linalg.qr(rng.normal(size=(4,4))+1j*rng.normal(size=(4,4)))[0]
    S=(u*np.array([.41,.32,.23,.14]))@u.T
    v=rng.normal(size=4)+1j*rng.normal(size=4)
    J=(rng.normal(size=(10,4))+1j*rng.normal(size=(10,4)))*.34
    J[7]=J[3]
    checks=[]
    for k in (5,7,9,10):
        ref=brute(S,v,J,k)
        for device in ([None,'cuda'] if k>=9 else [None]):
            got=amplitudes(S,v,J,k,device)
            error=float(max(abs(got-ref)))
            checked=amplitudes(S,v,J,k,device,compensated=True)
            checked_error=float(max(abs(checked-ref)))
            print(k,device,error,checked_error,flush=True)
            assert checked_error<2e-11 and max(abs(got-checked))<1e-7*max(abs(ref))
            checks.append(dict(pairs=k,device=device or 'cpu',max_absolute_error=error,
                               compensated_absolute_error=checked_error))
    (OUT/'contraction_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    print(checks)
if __name__=='__main__':
    with threadpool_limits(limits=1):run()
