"""Independent full144 vacuum and single-click target certificate audit."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,math,gzip,hashlib,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration53_independent import expand_record
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def integers(A):
    a=np.asarray(A);values=[F(float(x)) for x in a.flat];d=math.lcm(*(v.denominator for v in values));return np.array([v.numerator*(d//v.denominator) for v in values],dtype=object).reshape(a.shape),d


def outward(a,b):
    scale=1<<256;return F(a.numerator*scale//a.denominator,scale),F(-((-b.numerator*scale)//b.denominator),scale)


def main():
    out=ROOT/'results/iteration57_target_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration57_target','iteration53_independent','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];cert_count=0
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    def real_certificate(c,A,den,label):
        nonlocal cert_count
        cert_count+=1;Z=np.array(c['real'],dtype=object);d=c['denominator'];R=np.array(c['R_real'],dtype=object);s=c['scale'];n=len(Z)
        check(label+' exact matrix',np.array_equal(Z*den,A*d) and np.array_equal(Z,Z.T));check(label+' real triangular witness',not any(int(x) for row in c['imag'] for x in row) and not any(int(x) for row in c['R_imag'] for x in row) and all(int(R[i,i])>0 for i in range(n)) and not any(int(R[i,j]) for i in range(n) for j in range(i+1,n)))
        E=(R@Z)@R.T-np.eye(n,dtype=object)*d*s*s;delta=F(max(sum(abs(int(x)) for x in row) for row in E),d*s*s);trace=F(sum(int(x)**2 for x in R.flat),s*s);lam=(1-delta)/trace;check(label+' exact residual',0<=delta<1 and delta==c['diagnostics']['exact_residual_upper']);check(label+' eigenvalue',lam==c['diagnostics']['minimum_eigenvalue_lower']>0);detR=math.prod((F(int(R[j,j]),s) for j in range(n)),start=F(1));bounds=outward((1-delta)**n/detR**2,(1+delta)**n/detR**2);check(label+' determinant',tuple(c['determinant_interval'])==bounds);return delta,trace,lam
    def model_check(rec,arrays,label):
        B=arrays['B'];noise=arrays['F'];m=len(B);br,dr=integers(B.real);bi,di=integers(B.imag);d=max(dr,di);S=np.block([[br*(d//dr),bi*(d//di)],[bi*(d//di),-br*(d//dr)]]);K=np.eye(2*m,dtype=object)*d-S;check(label+' symmetric B',np.array_equal(B,B.T));delta,trace,_=real_certificate(rec['core'],K,d,label+' core');Z,dz=integers(noise);td=2*dz*dz*d;T=Z.T@(K@Z)+np.eye(noise.shape[1],dtype=object)*td;real_certificate(rec['noise'],T,td,label+' noise');R=np.array(rec['core']['R_real'],dtype=object);dy=rec['core']['scale']**2;dc=max(dy,dz*dz);C=(2*(R.T@R)-np.eye(2*m,dtype=object)*dy)*(dc//dy)+(Z@Z.T)*(dc//(dz*dz));radius=2*delta*trace/(1-delta);check(label+' center and radius',dc==rec['center_denominator'] and np.array_equal(C,np.array(rec['center_numerators'],dtype=object)) and radius==rec['operator_radius']);return C,dc,radius
    def enclosure_check(r,C,dc,radius,label):
        modes=tuple(r['modes']);m=len(C)//2;ix=modes+tuple(j+m for j in modes);A=C[np.ix_(ix,ix)]+np.eye(len(ix),dtype=object)*dc;_,_,lam=real_certificate(r['certificate'],A,2*dc,label);eta=radius/(2*lam);check(label+' relative perturbation',eta==r['relative_perturbation'] and 0<=eta<1);lo,hi=r['certificate']['determinant_interval'];bounds=outward(lo*(1-eta)**len(ix),hi*(1+eta)**len(ix));check(label+' perturbed determinant',tuple(r['perturbed_determinant_interval'])==bounds);a,b=r['raw_vacuum_interval'];check(label+' exact root inequalities',a>=0 and a*a<=1/bounds[1]<=1/bounds[0]<=b*b);check(label+' physical probability intersection',tuple(r['vacuum_interval'])==(max(F(0),a),min(F(1),b)));return tuple(r['vacuum_interval'])
    summary=read('results/iteration57_target/summary.json');check('complete',summary['completed'] and summary['full_outputs']==summary['enclosures']==5 and summary['unchanged_sources'] and summary['unchanged_dependencies'])
    for n,s in summary['source_hashes'].items():check('producer source '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration57_target/source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('producer dependency '+n,digest(ROOT/n)==s);deps[n]=s
    base=read('results/iteration35_panel/case_07.audit.json');arrays=np.load(ROOT/'results/iteration35_panel/inputs.npz');C,dc,radius=model_check(base,arrays,'exact model');storage=read('results/iteration57_target/storage.json');check('five enclosures',len(storage)==5);vac={}
    for meta in storage:
        n='results/iteration57_target/'+meta['path'];b=(ROOT/n).read_bytes();deps[n]=digest(ROOT/n);raw=gzip.decompress(b);check('gzip '+n,len(b)==meta['compressed_bytes'] and len(raw)==meta['raw_bytes'] and hashlib.sha256(raw).hexdigest()==meta['raw_sha256'] and b[4:8]==bytes(4));r=expand_record(decode(json.loads(raw)),C,dc);key=tuple(r['modes']);check('distinct selected modes '+n,key not in vac and list(key)==meta['modes']);vac[key]=enclosure_check(r,C,dc,radius,'enclosure '+n)
    allm=tuple(range(144));expected={allm}|{tuple(i for i in allm if i!=j) for j in (0,47,95,143)};check('exact complete output subset set',set(vac)==expected)
    target=read('results/iteration57_target/target.json');a,b=target['model_vacuum_interval'];kl,kh=base['core']['determinant_interval'];tl,th=base['noise']['determinant_interval'];check('full vacuum determinant identity',0<a<=b and a*a<=kl/th<=kh/tl<=b*b and [a,b]==base['vacuum_interval']);check('envelope full vacuum',tuple(target['envelope_vacuum_interval'])==vac[allm]);zero=(max(a,vac[allm][0]),min(b,vac[allm][1]));check('full vacuum methods intersect',zero[0]<=zero[1])
    records=target['targets'];check('five target records',len(records)==5)
    for index,j in enumerate((None,0,47,95,143)):
        output='0'*144 if j is None else ''.join('1' if i==j else '0' for i in allm)
        pair=zero if j is None else (max(F(0),vac[tuple(i for i in allm if i!=j)][0]-zero[1]),min(F(1),vac[tuple(i for i in allm if i!=j)][1]-zero[0]))
        check('complete output probability '+str(j),records[index]['output']==output and tuple(records[index]['probability_interval'])==pair and 0<=pair[0]<=pair[1]<=1)
    check('transfer equality',target['transfer_upper']==base['transfers']['-40']['all_measurement_TV_transfer_upper']);check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=len(checks),physical_certificates=cert_count,complete_outputs=5,source_hashes=hashes,dependencies=deps));print('target checks',len(checks),flush=True)


if __name__=='__main__':main()
