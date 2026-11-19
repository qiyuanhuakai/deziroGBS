"""Record every arithmetic sampling branch for independent CDF reconstruction."""
from fractions import Fraction as F
import math
import numpy as np
from gbslab.local_mps import threshold_step,sample_local_threshold
from gbslab.sampling_audit import choice_intervals,enumerate_floating_law

def enumerate_trace(tensors,bits=53,max_nodes=20000):
    n=len(tensors);grid=1<<bits;real=[F(0)]*(1<<n);words=[F(0)]*(1<<n);rf=wf=F(0);trace=[];stack=[((),0,np.ones(1,complex),F(1),F(1))];leaves=0
    while stack:
        path,mask,left,rweight,wweight=stack.pop();depth=len(path)
        if depth==n:real[mask]+=rweight;words[mask]+=wweight;leaves+=1;continue
        if len(trace)>=max_nodes:raise MemoryError('Trace node cap')
        node=dict(path=path,mask=mask,real_mass=rweight,word_mass=wweight,left_hex=[(float(z.real).hex(),float(z.imag).hex()) for z in left]);trace.append(node)
        try:
            branches,weights,total=threshold_step(left,tensors[depth],0j);p=weights/total;intervals,counts,cdf=choice_intervals(p,bits);node.update(status='completed',weights_hex=[float(x).hex() for x in weights],total_hex=total.hex(),probabilities_hex=[float(x).hex() for x in p],cdf_hex=[float(x).hex() for x in cdf],counts=counts)
        except (ValueError,ArithmeticError) as error:node.update(status='failed',error=str(error));rf+=rweight;wf+=wweight;continue
        for channel,(width,count) in enumerate(zip(intervals,counts)):
            r=rweight*width;w=wweight*F(count,grid)
            if not r and not w:continue
            if weights[channel]<=0:rf+=r;wf+=w;continue
            child=branches[channel]/math.sqrt(weights[channel]);stack.append((path+(channel,),mask|((channel!=0)<<depth),child,r,w))
    assert sum(real)+rf==sum(words)+wf==1
    old=enumerate_floating_law(tensors,np.zeros(n,complex),bits=bits,max_nodes=max_nodes);assert real==old['real_cdf_law'] and words==[F(x,old['grid_denominator']) for x in old['grid_numerators']] and rf==old['real_failure'] and wf==F(old['grid_failure_numerator'],old['grid_denominator'])
    return dict(real_law=real+[rf],word_law=words+[wf],trace=trace,nodes=len(trace),leaves=leaves,bits=bits,original_audit=old)

def public_calibration(tensors,trace,seed,samples=7):
    by_path={tuple(r['path']):r for r in trace};n=len(tensors)
    class RecordedChoice:
        def __init__(self):self.rng=np.random.default_rng(seed);self.path=();self.records=[]
        def choice(self,size,p):
            if len(self.path)==n:self.path=()
            expected=by_path[self.path];assert expected['status']=='completed' and [float(x).hex() for x in p]==expected['probabilities_hex'];channel=int(self.rng.choice(size,p=p));self.records.append(dict(path=self.path,channel=channel,probabilities_hex=expected['probabilities_hex']));self.path+=(channel,);return channel
    rng=RecordedChoice();outputs,info=sample_local_threshold(tensors,np.zeros((samples,n),complex),rng);return dict(seed=seed,samples=samples,outputs=outputs.tolist(),records=rng.records,diagnostics=info)
