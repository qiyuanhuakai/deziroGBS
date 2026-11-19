"""Cross-check optical triple sums with direct 6x6 determinants and LP duals without HiGHS."""
import os
for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import math
from iteration92_common_v2 import ROOT,Path,np,H,read,save,manifest

parser=argparse.ArgumentParser()
parser.add_argument('--output',type=Path,required=True)
out=parser.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
base=ROOT/'results/iteration95_third_click_v1'
order=np.argsort(np.load(ROOT/'results/iteration83_loss_pilot_v1/input.npz')['order'])
matrices=[]
for sector in (0,1):
    z=np.load(ROOT/f'results/iteration94_actual_quality_v1/input_{sector}.npz')
    u=z['u'][order];t=z['t'];l=z['noise']
    transfer=np.block([[u.real,-u.imag],[u.imag,u.real]])
    core=np.diag(np.r_[t/(1-t),-t/(1+t)])+2*l@l.T
    matrices.append(np.eye(2*len(u))+transfer@core@transfer.T)
m=len(order);ref=np.load(base/'cumulants.npz')
truth=ref['truth'];parts=ref['contributions']
def vacuum(ports):
    # Batched principal determinants in physical coordinates, no whitening or Schur update.
    idx=np.concatenate((ports,ports+m),axis=1)
    logsum=np.zeros(len(idx),np.longdouble)
    for q in matrices:
        sign,logabs=np.linalg.slogdet(q[idx[:,:,None],idx[:,None,:]])
        assert (sign==1).all()
        logsum+=logabs
    return np.exp(-logsum/2)
single=vacuum(np.arange(m)[:,None])
pair=np.zeros((m,m),np.longdouble)
for i in range(m-1):
    j=np.arange(i+1,m)
    pair[i,j]=vacuum(np.column_stack((np.full(len(j),i),j)))
selected=(0,127,511,895,1138,1149)
checks=[];checked=0
for i in selected:
    acc=np.longdouble(0)
    for j in range(i+1,m-1):
        k=np.arange(j+1,m)
        v=vacuum(np.column_stack((np.full(len(k),i),np.full(len(k),j),k)))
        centered=v-single[i]*pair[j,k]-single[j]*pair[i,k]-single[k]*pair[i,j]+2*single[i]*single[j]*single[k]
        acc-=6*centered.sum();checked+=len(k)
    error=float(abs(acc-parts[i,2]));assert error<1e-7
    checks.append(dict(first_port=i,third_contribution=float(acc),absolute_error=error))
mean=(1-single).sum();v=sum(single*(1-single))
for i in range(m-1):v+=2*(pair[i,i+1:]-single[i]*single[i+1:]).sum()
assert abs(mean-truth[0])<1e-9 and abs(v-truth[1])<1e-8
hw=np.load(ROOT/'results/iteration81_jiuzhang_accuracy_v3/histograms.npz')['total_click_histogram']
emp=hw/hw.sum();cdf=emp.cumsum()[:-1]
score=(np.arange(m+1)-truth[0])/np.sqrt(truth[1])
dual_checks=[]
for row in read(base/'floors.json'):
    d=row['degree'];data=np.load(base/f'floor_{d}.npz');p=data['law']
    y=data['inequality_multipliers'].astype(np.longdouble);z=data['equality_multipliers'].astype(np.longdouble)
    assert max(y)<=0
    functions=np.array([score**i for i in range(1,d+1)])
    jac=np.diff(functions,axis=1);scale=abs(jac).max(axis=1)
    targets=np.array([0.,1.,truth[2]/truth[1]**1.5])[:d]
    rhs=(functions@emp-targets)/scale
    mono=y[:m-1];plus=y[m-1:2*m-1];minus=y[2*m-1:]
    residual=np.r_[-(np.r_[mono,0]-np.r_[0,mono]+plus-minus+(jac/scale[:,None]).T@z),
                   1+sum(plus)+sum(minus)]
    lower=mono@np.diff(cdf)+z@rhs+np.minimum(np.r_[-cdf,0]*residual,np.r_[1-cdf,1]*residual).sum()
    upper=max(abs(p.cumsum()-emp.cumsum()))
    assert abs(lower-row['dual_lower'])<1e-9 and abs(upper-row['minimum_ks'])<1e-8
    assert np.max(abs(functions@p-targets))<1e-6 and p.min()>-1e-8
    dual_checks.append(dict(degree=d,lower=float(lower),upper=float(upper),gap=float(upper-lower)))
window=np.load(ROOT/'results/iteration95_ensemble_v2/quality_4_0.npz')['count']
quality=[]
for degree in (2,3):
    data=np.load(base/f'quality_{degree}.npz')
    probs=data['count'];support=window>0
    logs=np.log(window[support])[None,:]+data['slopes'][:,None]*np.arange(m+1)[support]
    weights=np.exp(logs-logs.max(1,keepdims=True))
    pred=data['weights']@(weights/weights.sum(1,keepdims=True))
    assert abs(pred-probs[support]).sum()<1e-10 and probs[~support].sum()<1e-12
    hardware_ks=float(max(abs(probs.cumsum()-emp.cumsum())))
    quality.append(dict(degree=degree,hardware_ks=hardware_ks,reconstructed_count_max_error=float(max(abs(pred-probs[support])))))
save(out,'validation.json',dict(checked_triples=checked,exhaustive_first_port_slices=checks,
    mean_error=float(abs(mean-truth[0])),variance_error=float(abs(v-truth[1])),duals=dual_checks,models=quality))
manifest(out)
