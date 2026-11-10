"""Independent density-pair Gaussian moments and complete floating-tree replay."""
from fractions import Fraction as F
from math import factorial,prod,sqrt
import numpy as np
from iteration47_radicals import mul,conj,squarefree
from iteration48_independent import rational_amplitudes,bisect_sqrt
from iteration45_independent import grid_below,mobius
from gbslab.local_mps import threshold_step

def continuous_density(tensors,direction,sigma):
    dims=[t.shape[1] for t in tensors];n=len(dims);occ=list(np.ndindex(*dims));amps=rational_amplitudes(tensors);norm=sum(a*a+b*b for a,b in amps);powers=[]
    for (x,y),d in zip(direction,dims):
        p=[(F(1),F(0))]
        for k in range(1,d):p.append(mul(p[-1],(-F(x),-F(y))))
        powers.append(p)
    records=[];vac=[]
    for mask in range(1<<n):
        A=1+2*sigma*sigma*sum(x*x+y*y for j,(x,y) in enumerate(direction) if mask>>j&1);groups={};imaginary={}
        for oa,aa in zip(occ,amps):
            if aa==(0,0):continue
            for ob,ab in zip(occ,amps):
                if ab==(0,0) or any(oa[j]!=ob[j] for j in range(n) if not mask>>j&1):continue
                degree=sum(oa[j]+ob[j] for j in range(n) if mask>>j&1)
                if degree%2:continue
                moment=F(prod(range(1,degree,2)))*sigma**degree/A**(degree//2);value=mul(conj(aa),ab);fact=1
                for j in range(n):
                    if mask>>j&1:value=mul(value,mul(powers[j][oa[j]],conj(powers[j][ob[j]])));fact*=factorial(oa[j])*factorial(ob[j])
                sq,rad=squarefree(fact);groups[rad]=groups.get(rad,F(0))+value[0]*moment/(norm*sq);imaginary[rad]=imaginary.get(rad,F(0))+value[1]*moment/(norm*sq)
        assert all(v==0 for v in imaginary.values());lo=hi=F(0)
        for rad,c in groups.items():
            a,b=bisect_sqrt(1/(rad*A));lo+=c*(a if c>=0 else b);hi+=c*(b if c>=0 else a)
        p=(max(F(0),lo),min(F(1),hi));vac.append(p);records.append(dict(mask=mask,tilt=A,radicals=groups,vacuum_interval=p))
    raw,law=mobius(vac,n);return dict(records=records,vacuum_intervals=vac,raw_intervals=raw,intervals=law)

def trace_check(tensors,alpha,sampling,check,tag):
    n=len(tensors);trace={tuple(node['path']):node for node in sampling['trace']};check(tag+'unique nodes',len(trace)==len(sampling['trace'])==sampling['nodes']);real=[F(0)]*((1<<n)+1);word=[F(0)]*((1<<n)+1);queue=[((),0,np.ones(1,complex),F(1),F(1))];seen=set();leaves=0
    for path,mask,left,rm,wm in queue:
        if len(path)==n:real[mask]+=rm;word[mask]+=wm;leaves+=1;continue
        node=trace[path];seen.add(path);check(tag+'node '+str(path),node['mask']==mask and node['real_mass']==rm and node['word_mass']==wm and node['left_hex']==[[float(z.real).hex(),float(z.imag).hex()] for z in left]);branches,weights,total=threshold_step(left,tensors[len(path)],alpha[len(path)]);p=weights/total;cdf=p.cumsum();cdf/=cdf[-1];check(tag+'arithmetic '+str(path),node['status']=='completed' and node['weights_hex']==[float(x).hex() for x in weights] and node['total_hex']==total.hex() and node['probabilities_hex']==[float(x).hex() for x in p] and node['cdf_hex']==[float(x).hex() for x in cdf]);last=F(0);lastcount=0;counts=[]
        for channel,edge in enumerate(cdf):
            e=F(float(edge));count=grid_below(e,53);c=count-lastcount;counts.append(c);rchild=rm*(e-last);wchild=wm*F(c,1<<53);last=e;lastcount=count
            if not rchild and not wchild:continue
            if weights[channel]<=0:real[-1]+=rchild;word[-1]+=wchild;continue
            queue.append((path+(channel,),mask|((channel!=0)<<len(path)),branches[channel]/sqrt(weights[channel]),rchild,wchild))
        check(tag+'word counts '+str(path),counts==node['counts'] and last==1 and lastcount==1<<53)
    check(tag+'complete law',seen==set(trace) and leaves==sampling['leaves'] and real==sampling['real_law'] and word==sampling['word_law'] and sum(real)==sum(word)==1);return word,len(trace)
