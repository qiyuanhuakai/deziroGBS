"""Independent rational input identity and Gaussian measurement reconstruction."""
import math
from fractions import Fraction as F
import numpy as np
from iteration66_support import save,encode
from iteration48_independent import independent_overlap,bisect_sqrt
from verify_iteration34 import exact_determinant
from iteration45_independent import inverse,mobius
from iteration51_normal_independent import exp_negative
from functools import lru_cache
exp_negative=lru_cache(None)(exp_negative)

def verify_input(rec,tensors,check,tag,out,index):
    job=rec['job'];n=job['modes'];d=job['dimension'];r=job['real'];im=job['imag'];br=rec['bridge']
    check(tag+'box dimensions',br['dimensions']==[d]*n and br['occupations']==[list(x) for x in np.ndindex(*([d]*n))]);ind=independent_overlap(tensors,r,im);save(out/f'case_{index:02d}.overlap.json',ind);check(tag+'independent polynomial coefficients',encode(ind['hermite'])==encode(br['hermite']));check(tag+'projection coefficient norm',ind['projection_unnormalized_norm']==br['projection_unnormalized_norm']);scales=[max(F(float(z)).denominator for z in np.concatenate([t.real.ravel(),t.imag.ravel()])) for t in tensors];scale=math.prod(scales);check(tag+'independent rational MPS amplitudes',all([ar*scale,ai*scale]==v for (ar,ai),v in zip(ind['amplitudes'],br['integer_amplitudes'])) and ind['norm']*scale**2==br['integer_norm'])
    S=np.block([[np.array(r,dtype=object),np.array(im,dtype=object)],[np.array(im,dtype=object),-np.array(r,dtype=object)]]);A=np.eye(2*n,dtype=object)-S;det=exact_determinant(A.tolist());inv=np.array(inverse(A.tolist()),dtype=object);V=2*inv-np.eye(2*n,dtype=object);omega=np.block([[np.zeros((n,n),dtype=object),np.eye(n,dtype=object)],[-np.eye(n,dtype=object),np.zeros((n,n),dtype=object)]]);check(tag+'exact Gaussian covariance and purity',V.tolist()==br['physical']['covariance'] and det==br['physical']['determinant'] and np.array_equal(V@omega@V,omega));check(tag+'physical positive leading minors',all(exact_determinant(A[:k,:k].tolist())>0 for k in range(1,2*n+1)));normlo,normhi=bisect_sqrt(det);l,h=br['normalization_probability_interval'];check(tag+'independent Gaussian normalization',l<=normlo<=normhi<=h)
    overlap=ind['overlap_squared_over_norm_interval'];lo,hi=br['overlap_squared_interval'];check(tag+'independent density overlap',lo/br['integer_norm']<=overlap[0]<=overlap[1]<=hi/br['integer_norm']);R=ind['projection_unnormalized_norm'];pairs=[('projection_mass_interval',(normlo*R,normhi*R)),('projected_fidelity_interval',(overlap[0]/R,overlap[1]/R)),('full_fidelity_interval',(normlo*overlap[0],normhi*overlap[1]))]
    for name,(lo,hi) in pairs:
        lo=max(F(0),lo);hi=min(F(1),hi);l,h=br[name];check(tag+'independent '+name,l<=lo<=hi<=h)
    for fid,dist in [('projection_mass_interval','tail_trace_interval'),('projected_fidelity_interval','compression_trace_interval'),('full_fidelity_interval','full_trace_interval')]:
        fl,fh=br[fid];dl,dh=br[dist];check(tag+'trace square inequalities '+dist,0<=dl<=dh<=1 and dl*dl<=1-fh<=1-fl<=dh*dh)
    check(tag+'tail mass identity',br['tail_mass_interval']==[1-br['projection_mass_interval'][1],1-br['projection_mass_interval'][0]])
    if job['family']=='pair2':
        a=F(1,3);check(tag+'analytic pair tail',br['tail_mass_interval'][0]<=a**(2*d)<=br['tail_mass_interval'][1])
    return ind

def verify_gaussian(rec,bridge,alpha,check,tag):
    n=rec['job']['modes'];V=np.array(bridge['physical']['covariance'],dtype=object)
    target=rec['gaussian_target'];check(tag+'Gaussian target uses same exact covariance',target['physical']['covariance']==V.tolist());z=[F(float(x.real)) for x in alpha]+[F(float(x.imag)) for x in alpha];vac=[]
    for mask,row in enumerate(target['records']):
        modes=[j for j in range(n) if mask>>j&1];ix=modes+[j+n for j in modes]
        if ix:
            Q=[[(V[i,j]+int(i==j))/2 for j in ix] for i in ix];di=exact_determinant(Q);qi=inverse(Q);exponent=sum(z[i]*qi[j][k]*z[l] for j,i in enumerate(ix) for k,l in enumerate(ix))
        else:di=F(1);exponent=F(0)
        check(tag+'Gaussian determinant/exponent '+str(mask),row['mask']==mask and row['determinant']==di and row['exponent']==exponent);rl,rh=bisect_sqrt(1/di);el,eh=exp_negative(exponent);l,h=row['factor_interval'];check(tag+'Gaussian independent radical '+str(mask),l<=rl<=rh<=h);l,h=row['exponential_interval'];check(tag+'Gaussian independent exponential '+str(mask),l<=el<=eh<=h);v=target['vacuum_intervals'][mask];check(tag+'Gaussian independent vacuum '+str(mask),v[0]<=rl*el<=rh*eh<=v[1]);vac.append((rl*el,rh*eh))
    raw,law=mobius(vac,n)
    for k,(l,h) in enumerate(law):
        pl,ph=target['intervals'][k];check(tag+'independent full Gaussian atom '+str(k),pl<=l<=h<=ph)
    raw,law=mobius(target['vacuum_intervals'],n);check(tag+'full Gaussian interval Mobius',encode(raw)==encode(target['raw_intervals']) and encode(law)==encode(target['intervals']))
