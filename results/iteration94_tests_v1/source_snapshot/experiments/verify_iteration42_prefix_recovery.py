"""Independent actual144-prefix laws and revalidation of physical target certificates.

The matrix certificate checker is retained verbatim from verify_iteration37;
the producer is GaussianEnvelope, and this rechecks its exact witnesses.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from iteration41_independent import layer_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def integers(A):
    a=np.asarray(A);values=[F(float(x)) for x in a.flat];d=math.lcm(*(v.denominator for v in values));return np.array([v.numerator*(d//v.denominator) for v in values],dtype=object).reshape(a.shape),d


def outward(a,b):
    scale=1<<256;return F(a.numerator*scale//a.denominator,scale),F(-((-b.numerator*scale)//b.denominator),scale)


def main():
    out=ROOT/'results/iteration42_prefix_recovery';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration42_prefix_recovery','verify_iteration42_prefix','verify_iteration37','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    checks=[];deps={};cert_count=0;steps=0;depth_rows=[]
    def read(name):
        path=ROOT/name;deps[name]=digest(path);return decode(json.loads(path.read_text()))
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(name)
    def real_certificate(c,A,den,label):
        nonlocal cert_count
        cert_count+=1;Z=np.array(c['real'],dtype=object);d=c['denominator'];R=np.array(c['R_real'],dtype=object);s=c['scale'];n=len(Z)
        check(label+' exact matrix',np.array_equal(Z*den,A*d) and np.array_equal(Z,Z.T));check(label+' real triangular witness',not any(int(x) for row in c['imag'] for x in row) and not any(int(x) for row in c['R_imag'] for x in row) and all(int(R[i,i])>0 for i in range(n)) and not any(int(R[i,j]) for i in range(n) for j in range(i+1,n)))
        E=(R@Z)@R.T-np.eye(n,dtype=object)*d*s*s;delta=F(max(sum(abs(int(x)) for x in row) for row in E),d*s*s);trace=F(sum(int(x)**2 for x in R.flat),s*s);lam=(1-delta)/trace;check(label+' exact residual',0<=delta<1 and delta==c['diagnostics']['exact_residual_upper']);check(label+' eigenvalue',lam==c['diagnostics']['minimum_eigenvalue_lower']>0);detR=math.prod((F(int(R[j,j]),s) for j in range(n)),start=F(1));bounds=outward((1-delta)**n/detR**2,(1+delta)**n/detR**2);check(label+' determinant',tuple(c['determinant_interval'])==bounds);return delta,trace,lam
    def model_check(rec,arrays,label):
        B=arrays['B'];noise=arrays['F'];m=len(B);br,dr=integers(B.real);bi,di=integers(B.imag);d=max(dr,di);S=np.block([[br*(d//dr),bi*(d//di)],[bi*(d//di),-br*(d//dr)]]);K=np.eye(2*m,dtype=object)*d-S;check(label+' symmetric B',np.array_equal(B,B.T));delta,trace,_=real_certificate(rec['core'],K,d,label+' core');Z,dz=integers(noise);td=2*dz*dz*d;T=Z.T@(K@Z)+np.eye(noise.shape[1],dtype=object)*td;real_certificate(rec['noise'],T,td,label+' noise');R=np.array(rec['core']['R_real'],dtype=object);dy=rec['core']['scale']**2;dc=max(dy,dz*dz);C=(2*(R.T@R)-np.eye(2*m,dtype=object)*dy)*(dc//dy)+(Z@Z.T)*(dc//(dz*dz));radius=2*delta*trace/(1-delta);check(label+' center and radius',dc==rec['center_denominator'] and np.array_equal(C,np.array(rec['center_numerators'],dtype=object)) and radius==rec['operator_radius']);return C,dc,radius
    def enclosure_check(r,C,dc,radius,label):
        modes=tuple(r['modes']);m=len(C)//2;ix=modes+tuple(j+m for j in modes);A=C[np.ix_(ix,ix)]+np.eye(len(ix),dtype=object)*dc;_,_,lam=real_certificate(r['certificate'],A,2*dc,label);eta=radius/(2*lam);check(label+' relative perturbation',eta==r['relative_perturbation'] and 0<=eta<1);lo,hi=r['certificate']['determinant_interval'];bounds=outward(lo*(1-eta)**len(ix),hi*(1+eta)**len(ix));check(label+' perturbed determinant',tuple(r['perturbed_determinant_interval'])==bounds);a,b=r['raw_vacuum_interval'];check(label+' exact root inequalities',a>=0 and a*a<=1/bounds[1]<=1/bounds[0]<=b*b);check(label+' physical probability intersection',tuple(r['vacuum_interval'])==(max(F(0),a),min(F(1),b)));return tuple(r['vacuum_interval'])
    def law_check(law,vac,label):
        modes=law['modes'];n=len(modes)
        for clicks in range(1<<n):
            zeros=((1<<n)-1)^clicks;lo=hi=F(0)
            for subset in range(1<<n):
                if subset&~clicks:continue
                mask=zeros|subset;key=tuple(modes[j] for j in range(n) if mask>>j&1);a,b=vac[key]
                if subset.bit_count()%2:lo-=b;hi-=a
                else:lo+=a;hi+=b
            check(label+f' atom {clicks}',tuple(law['raw_intervals'][clicks])==(lo,hi) and tuple(law['intervals'][clicks])==(max(F(0),lo),min(F(1),hi)))
        check(label+' mass',sum(a for a,b in law['intervals'])<=1<=sum(b for a,b in law['intervals']))
    panel='results/iteration42_prefix';plan=read(panel+'/plan.json');summary=read(panel+'/summary.json');check('nine completed jobs',summary['planned']==summary['finished']==summary['completed']==9 and summary['unchanged_sources'])
    for folder,record in ((panel,plan),('results/iteration42_build',read('results/iteration42_build/build.json'))):
        for n,sha in record['source_hashes'].items():check(folder+' source '+n,digest(ROOT/n)==sha);check(folder+' snapshot '+n,digest(ROOT/folder/'source_snapshot'/n)==sha)
        for n,sha in record.get('dependencies',{}).items():check(folder+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
    for name in ('results/iteration39_calibration/validation.json','results/iteration35_validation/validation.json','results/iteration37_validation/validation.json'):
        record=read(name);check(name+' passed',record['count']==record['passed'])
    inputs=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=read('results/iteration35_panel/case_07.audit.json');C,dc,radius=model_check(base,inputs,'physical model');physical=read('results/iteration37_local/target_000.json');vac={(): (F(1),F(1))}
    for i,enclosure in enumerate(physical['enclosures']):vac[tuple(enclosure['modes'])]=enclosure_check(enclosure,C,dc,radius,'target '+str(i))
    check('all ten-mode subsets',len(vac)==1024);law_check(physical['law'],vac,'physical target');target=physical['law']['intervals'];transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper']
    invalid=read(panel+'/invalid_depth.json');check('three depth guards',len(invalid)==3 and all(r['returncode']!=0 and not r['stdout'] for r in invalid))
    global_trees={}
    for job in plan['jobs']:
        label=job['label'];raw=read(panel+'/'+label+'.stdout.json');record=read(panel+'/'+label+'.audit.json');old=read(f"results/iteration39_{'local' if job['modes']==10 else 'global'}/case_{job['cutoff']-1:02d}.stdout.json")
        check(label+' identity',record['job']==job and raw['modes']==job['modes'] and raw['cutoff']==job['cutoff'] and raw['samples']==17 and raw['prefix_depth']==10 and raw['prefix_reverse_verified'])
        check(label+' frozen public outputs',raw['outputs']==old['outputs'][:17] and raw['audit_paths']==old['audit_paths'][:17]);steps+=replay({**raw,'prefix_tree':[]})
        table={r['prefix']:r['threshold_hex'] for r in raw['prefix_tree']};by_prefix={r['prefix']:r for r in raw['prefix_tree']};check(label+' complete prefix set',len(table)==1023 and all(''.join(str((m>>j)&1) for j in range(k)) in table for k in range(10) for m in range(1<<k)))
        for prefix,row in by_prefix.items():
            q,prev,z=(float.fromhex(row[k]) for k in ('threshold_hex','previous_hex','proposed_hex'))
            check(label+' quotient '+prefix,math.isnan(q) and prev==z==0 if prev==0 else q==float(F(z)/F(prev)))
            if prefix:
                parent=by_prefix[prefix[:-1]];pp=float.fromhex(parent['previous_hex']);pz=float.fromhex(parent['proposed_hex']);expected=pz if prefix[-1]=='0' else float(F(pp)-F(pz));check(label+' prefix continuity '+prefix,prev==expected)
        for s in range(17):
            for j in range(10):check(label+f' sampled prefix {s}:{j}',table[raw['outputs'][s][:j]]==raw['audit_paths'][s][j][1])
        if job['modes']==10:check(label+' old full tree',raw['prefix_tree']==old['prefix_tree'])
        else:
            V=np.loadtxt(ROOT/f"results/iteration39_global/case_{job['cutoff']-1:02d}.csv",delimiter=',');check(label+' actual full input',np.array_equal(V,inputs['shifted']) and V.shape==(288,288));global_trees[job['cutoff']]=by_prefix
        for model in ('continuous','uint64'):
            law=layer_law(table,10,model);check(label+' independent law '+model,tuple(law)==tuple(record['laws'][model]['distribution']) and sum(law)==1 and min(law)>=0)
            lo=sum((max(F(0),a-q,q-b) for q,(a,b) in zip(law,target)),F(0))/2;hi=sum((max(abs(q-a),abs(q-b)) for q,(a,b) in zip(law,target)),F(0))/2
            check(label+' independent TV '+model,(lo,hi)==tuple(record['prefix_TV_intervals'][model]))
        law=record['laws']['uint64']['distribution'];event=record['event'];selected=event['masks'];check(label+' output event',selected==[i for i,(q,(a,b)) in enumerate(zip(law,target)) if q>b] and sum(law[i] for i in selected)==event['sampler_probability'] and tuple(event['target_probability_interval'])==(sum(target[i][0] for i in selected),sum(target[i][1] for i in selected)))
        gap=event['sampler_probability']-event['target_probability_interval'][1];check(label+' event lower',event['TV_lower']==max(F(0),gap))
        if job['modes']==144:
            lower=record['prefix_TV_intervals']['uint64'][0];check(label+' full-law data processing',record['full144_TV_lower']==lower and record['full144_shifted_TV_lower']==max(F(0),lower-transfer) and record['transfer_upper']==transfer)
            for depth in range(1,11):
                size=1<<depth;q=[sum(law[i] for i in range(k,1024,size)) for k in range(size)];p=[(sum(target[i][0] for i in range(k,1024,size)),sum(target[i][1] for i in range(k,1024,size))) for k in range(size)];low=sum((max(F(0),a-x,x-b) for x,(a,b) in zip(q,p)),F(0))/2;high=sum((max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,p)),F(0))/2;depth_rows.append(dict(cutoff=job['cutoff'],depth=depth,TV_lower=low,TV_upper=high,shifted_lower=max(F(0),low-transfer)))
        print('verified prefix',label,flush=True)
    check('four actual full targets',set(global_trees)=={1,2,3,4});check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));save(out/'depth_bounds.json',encode(depth_rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),physical_certificates=cert_count,replayed_steps=steps,prefix_trees=9,source_hashes=hashes,dependencies=deps,scope='Certified proper-prefix lower bounds for actual144-mode laws; no full-law upper bound'));print('prefix verified',len(checks),flush=True)
if __name__=='__main__':main()
