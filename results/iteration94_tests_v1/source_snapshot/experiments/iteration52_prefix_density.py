"""Independent exact double-layer law with shared threshold-prefix transfers."""
from fractions import Fraction as F
from time import perf_counter
import numpy as np

def integer_density_law(tensors,*,return_stats=False):
    started=perf_counter();converted=[];scale=1
    for tensor in tensors:
        pairs=[float(x).as_integer_ratio() for z in tensor.flat for x in (z.real,z.imag)];den=1<<max(q.bit_length()-1 for p,q in pairs);scale*=den;values=[p*(den//q) for p,q in pairs];converted.append((np.array(values[::2],dtype=object).reshape(tensor.shape),np.array(values[1::2],dtype=object).reshape(tensor.shape)))
    masses=[0]*(1<<len(converted));transfers=0;leaves=0
    def visit(site,mask,er,ei):
        nonlocal transfers,leaves
        if site==len(converted):
            assert ei.shape==er.shape==(1,1) and ei[0,0]==0 and er[0,0]>=0;masses[mask]=int(er[0,0]);leaves+=1;return
        ar,ai=converted[site]
        for clicked in (False,True):
            transfers+=1;nr=np.zeros((ar.shape[2],ar.shape[2]),dtype=object);ni=nr.copy()
            for k in (range(1,ar.shape[1]) if clicked else (0,)):
                r=ar[:,k,:];i=ai[:,k,:];ur=er@r-ei@i;ui=er@i+ei@r;nr+=r.T@ur+i.T@ui;ni+=r.T@ui-i.T@ur
            visit(site+1,mask|((1<<site) if clicked else 0),nr,ni)
    visit(0,0,np.ones((1,1),dtype=object),np.zeros((1,1),dtype=object));norm=sum(masses)
    if norm<=0:raise ValueError('Zero represented state')
    result=(masses,norm,scale,[F(x,norm) for x in masses]);stats=dict(prefix_transfers=transfers,leaf_outputs=leaves,repeated_path_transfers=len(tensors)*(1<<len(tensors)),seconds=perf_counter()-started)
    return (*result,stats) if return_stats else result
