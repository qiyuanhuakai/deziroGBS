"""Transfer the fixed cutoffs and rejection bounds across the three published calibrations."""
import json
import mpmath as mp
import numpy as np
from threadpoolctl import threadpool_limits
from iteration117_power_inputs import load_power
from iteration118_auxiliary import prepare
from iteration119_hardware import OUT

def retained(t,s):
    matrix=prepare(t,s,2)[1]
    coeff=[mp.mpf(1)]+[mp.mpf(0)]*10
    for singular in np.linalg.svd(matrix,compute_uv=False):
        x=mp.mpf(float(singular))**2
        one=[mp.sqrt(1-x)]
        for n in range(1,11):one.append(one[-1]*x*(2*n-1)/(2*n))
        coeff=[mp.fsum(coeff[j]*one[n-j] for j in range(n+1)) for n in range(11)]
    return [mp.fsum(coeff[:k+1]) for k in range(11)]

def run():
    mp.mp.dps=65
    def excess(scale,F):
        scale=mp.mpf(str(scale));d=1-F
        return 2*scale*d/(mp.sqrt((scale-1)**2+4*scale*d)+scale-1)
    out={}
    for power in ('low','median','high'):
        t,ss=load_power(power)
        a,b=[retained(t,s) for s in ss]
        rb,rm,rp=a[7]*b[5],a[9]*b[6],a[10]*b[6]
        loss=excess(4,rb/rp)+excess(1.25,rm/rp)
        extra=mp.mpf('.5')*excess(2,rb/rp)+mp.mpf('.8')*excess(1.25,rm/rp)
        intrinsic=mp.sqrt(1-rp);clamp=max(loss,extra)
        out[power]=dict(mean_photons=float(sum((abs(t)**2 @ np.sinh(s)**2).sum() for s in ss)),
                        truncation_tv=float(intrinsic),clamp_tv_upper=float(clamp),
                        full_tv_upper=float(intrinsic+clamp),output_records_measured_here=(power=='high'))
    assert abs(out['high']['full_tv_upper']-.004823426230405225)<1e-12
    (OUT/'power_transfer.json').write_text(json.dumps(out,indent=2)+'\n')
    print(out)
if __name__=='__main__':
    with threadpool_limits(limits=2):run()
