"""Independent rational density contractions and positive-exponential tails."""
from fractions import Fraction as F
import numpy as np

def exp_inverse_positive(x,terms=192):
    x=F(x);assert 0<=x<=4;term=F(1);total=F(1)
    for k in range(1,terms+1):term*=x/k;total+=term
    first_omitted=term*x/(terms+1);tail=first_omitted/(1-x/F(terms+2))
    return 1/(total+tail),1/total

def density_vacuum_coefficient(tensors,alpha,mask):
    er=np.ones((1,1),dtype=object);ei=np.zeros((1,1),dtype=object)
    for j,a in enumerate(tensors):
        ar=np.array([F(float(z.real)) for z in a.flat],dtype=object).reshape(a.shape);ai=np.array([F(float(z.imag)) for z in a.flat],dtype=object).reshape(a.shape)
        if mask>>j&1:
            x=F(float(alpha[j].real));y=F(float(alpha[j].imag));matrices=[(ar[:,0,:]-x*ar[:,1,:]-y*ai[:,1,:],ai[:,0,:]-x*ai[:,1,:]+y*ar[:,1,:])]
        else:matrices=[(ar[:,k,:],ai[:,k,:]) for k in range(2)]
        nr=np.zeros((a.shape[2],a.shape[2]),dtype=object);ni=nr.copy()
        for r,i in matrices:
            ur=er@r-ei@i;ui=er@i+ei@r;nr+=r.T@ur+i.T@ui;ni+=r.T@ui-i.T@ur
        er,ei=nr,ni
    assert ei[0,0]==0 and er[0,0]>=0
    return F(er[0,0])
