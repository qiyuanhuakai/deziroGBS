"""Exact passive purity, exact B/V relation, and independent target elimination."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration45_independent import inverse,mobius
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration46_input_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration46_inputs','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration46_inputs';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('8 targets completed',summary['planned']==summary['completed']==8)
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    records=[]
    for i,job in enumerate(plan['jobs']):
        rec=read(base+f'/case_{i:02d}.json');records.append(rec);n=job['modes'];p=rec['physical'];tag=f'case{i} ';check(tag+'identity',rec['job']==job and p['order']==job['order'] and sorted(job['order'])==list(range(n)));phases=([1,1j,-1,1] if n==4 else [1,1j,-1,1,1j,-1j,1,-1]);ar=np.array([[sum(F(int(complex(phases[k]).real))*(-1)**((a&k).bit_count()+(b&k).bit_count()) for k in range(n))/n for b in range(n)] for a in range(n)],dtype=object);ai=np.array([[sum(F(int(complex(phases[k]).imag))*(-1)**((a&k).bit_count()+(b&k).bit_count()) for k in range(n))/n for b in range(n)] for a in range(n)],dtype=object);check(tag+'fixed unitary',ar.tolist()==p['unitary_real'] and ai.tolist()==p['unitary_imag'] and np.array_equal(ar@ar.T+ai@ai.T,np.eye(n,dtype=object)) and not np.any(ai@ar.T-ar@ai.T));S=np.block([[ar,-ai],[ai,ar]]);Z=np.zeros((n,n),dtype=object);I=np.eye(n,dtype=object);omega=np.block([[Z,I],[-I,Z]]);check(tag+'exact symplectic',np.array_equal(S@omega@S.T,omega));q=([F(1,2),F(1,2),F(1,4),F(1,2)] if job['strength']=='moderate' else [F(1,4),F(1,8),F(1,2),F(1,4)])*(n//4);big=[1/x for x in q];D=np.diag(big+q);order=job['order'];ix=order+[n+j for j in order];V=(S@D@S.T)[np.ix_(ix,ix)];check(tag+'fixed pure covariance',q==p['small_variances'] and big==p['large_variances'] and V.tolist()==p['covariance'] and np.array_equal(V@omega@V,omega) and all(x>0 for x in q+big));parameters=[(1-x)/(1+x) for x in q];br=(ar@np.diag(parameters)@ar.T-ai@np.diag(parameters)@ai.T)[np.ix_(order,order)];bi=(ar@np.diag(parameters)@ai.T+ai@np.diag(parameters)@ar.T)[np.ix_(order,order)];check(tag+'exact pairing',br.tolist()==p['pairing_real'] and bi.tolist()==p['pairing_imag'] and parameters==p['pairing_parameters']);K=np.eye(2*n,dtype=object)-np.block([[br,bi],[bi,-br]]);v2=2*np.array(inverse(K),dtype=object)-np.eye(2*n,dtype=object);check(tag+'exact pairing covariance identity',np.array_equal(v2,V));name=base+f'/case_{i:02d}.npz';deps[name]=digest(ROOT/name);z=np.load(ROOT/name);err=max(abs(F(float(z['B'][a,b].real))-br[a,b])+abs(F(float(z['B'][a,b].imag))-bi[a,b]) for a in range(n) for b in range(n));check(tag+'entrywise rounding recorded',err==p['entrywise_pairing_l1_rounding_max'] and np.array_equal(z['B'],np.asarray(br,float)+1j*np.asarray(bi,float)));check(tag+'exact native covariance bytes',all(F(float(z['V'][a,b]))==V[a,b] for a in range(2*n) for b in range(2*n)))
        target=rec['target'];vac=target['vacuum_intervals'];check(tag+'vacuum base',vac[0]==[F(1),F(1)])
        for mask in range(1,1<<n):
            modes=[j for j in range(n) if mask>>j&1];ids=modes+[j+n for j in modes];det=exact_determinant([[(V[a,b]+int(a==b))/2 for b in ids] for a in ids]);lo,hi=vac[mask];check(tag+'independent determinant '+str(mask),det==target['determinants'][mask-1]>=1 and 0<=lo<=hi<=1 and lo*lo*det<=1<=hi*hi*det)
        raw,law=mobius(vac,n);check(tag+'independent target',encode(raw)==encode(target['raw_intervals']) and encode(law)==encode(target['target_intervals']) and all(lo>0 for lo,hi in law));print('verified shared target',i,flush=True)
    for i in range(0,8,2):
        n=records[i]['job']['modes'];a=records[i]['target']['target_intervals'];b=records[i+1]['target']['target_intervals']
        for mask in range(1<<n):rev=sum(((mask>>j)&1)<<(n-1-j) for j in range(n));check(f'order{i} target{mask}',a[mask]==b[rev])
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=8,source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
