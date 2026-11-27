"""Only even total-count bins fit the mixture; odd bins and all spatial data are held out."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration115_tilt_v1 import setup
from iteration115_visible_v1 import DATA,hardware,gauss,energy_law


def run(out):
    out.mkdir(parents=True,exist_ok=True)
    t,ss=hardware();rows=[];arrays={}
    cases=[('gain035',t,.35),('gain065',t,.65),('original',t,1.),('gain120',t,1.2),
           ('banks72',t[:576],1.),('banks36',t[:288],1.)]
    for name,tt,scale in cases:
        gains=np.arcsinh(scale*np.sinh(ss))
        ev=np.concatenate([gauss(tt,s)[-1] for s in gains])
        cap=480
        truth,tail=energy_law(ev,cap)
        _,fixed,_=setup(tt,gains,truth,fit_even=True,cap=cap)
        args,info,z=setup(tt,gains,truth,fit_even=True,cap=cap,adaptive=True)
        odd=slice(1,None,2)
        conditional=lambda p:p[odd]/p[odd].sum()
        row=dict(case=name,ports=len(tt),gain_scale=scale,mean_photons=ev.sum()/2,
            fixed_grid_full_tv_upper=fixed['total_tv_upper'],weights=info['weights'],log_tilts=info['log_tilts'],tail=tail,
            base_total_tv_upper=info['base_total_tv_upper'],heldout_fit_full_tv_upper=info['total_tv_upper'],
            odd_conditional_tv=float(abs(conditional(z['truth'])-conditional(z['total'])).sum()/2),
            odd_conditional_base_tv=float(abs(conditional(z['truth'])-conditional(z['base'])).sum()/2))
        # Independently evaluate each prepared tilted covariance at complex fugacities.
        phase=np.linspace(.003,.75,28);u=np.exp(1j*phase)
        characteristic=np.zeros(len(u),complex)
        for w,mats in zip(z['weights'],z['marginal']):
            e=np.concatenate([np.linalg.eigvalsh(x) for x in mats])
            characteristic+=w*np.exp(-.5*np.log1p((1-u[:,None])*e).sum(1))
        rhs=np.exp(1j*phase[:,None]*np.arange(cap+1))@z['total']
        gap=float(abs(characteristic-rhs).max())
        assert gap<info['outside_compared_model']+2e-9
        row['characteristic_max_abs']=gap
        arrays[name+'_truth']=truth;arrays[name+'_model']=z['total'];arrays[name+'_base']=z['base']
        rows.append(row);print(name,row['base_total_tv_upper'],row['heldout_fit_full_tv_upper'],flush=True)
    np.savez_compressed(out/'totals.npz',**arrays)
    (out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    with threadpool_limits(limits=4):run(p.parse_args().output)
