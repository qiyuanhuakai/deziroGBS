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
    out=ROOT/'results/iteration49_prefix_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration49_prefix','verify_iteration42_prefix_recovery','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};checks=[];deps={};cert_count=0;steps=0;depth_rows=[];bounds=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
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
    panel='results/iteration49_prefix';plan=read(panel+'/plan.json');summary=read(panel+'/summary.json');targetdir='results/iteration49_target';targetplan=read(targetdir+'/plan.json');targetsummary=read(targetdir+'/summary.json')
    check('four completed runs',summary['planned']==summary['finished']==summary['completed']==4 and summary['unchanged_sources'] and summary['unchanged_dependencies']);check('target complete',targetsummary['completed'] and targetsummary['enclosures']==16383 and targetsummary['atoms']==16384 and targetsummary['unchanged_sources'])
    for folder,p in ((panel,plan),(targetdir,targetplan)):
        for n,sha in p['source_hashes'].items():check(folder+' source '+n,digest(ROOT/n)==sha==digest(ROOT/folder/'source_snapshot'/n))
        for n,sha in p['dependencies'].items():check(folder+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
    inputs=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=read('results/iteration35_panel/case_07.audit.json');C,dc,radius=model_check(base,inputs,'physical model');physical=read(targetdir+'/target.json');vac={(): (F(1),F(1))}
    selected={}
    check('incremental target counts',targetsummary['new_enclosures']==8192 and targetsummary['reused_enclosures']==8191 and targetplan['reused_nonempty_subsets']==8191 and targetplan['new_nonempty_subsets']==8192)
    oldmanifest=read('results/iteration47_manifest.json')
    for directory in ('results/iteration47_target',targetdir):
        for shard in range(32):
            name=directory+f'/certificates_{shard:02d}.json';records=read(name)
            if directory!=targetdir:check('bound reused shard '+name,deps[name]==oldmanifest['artifacts'][name]==targetplan['dependencies'][name])
            for enclosure in records:
                key=tuple(enclosure['modes']);check('unique and correct old/new subset '+str(key),key not in vac and ((13 in key)==(directory==targetdir)));vac[key]=enclosure_check(enclosure,C,dc,radius,'target '+str(key))
                if directory!=targetdir and key==tuple(range(len(key))):selected[key]=enclosure
    controls=read(targetdir+'/recomputed_controls.json');check('thirteen fresh old controls',len(controls)==13 and all(r==selected[tuple(r['modes'])] for r in controls) and {tuple(r['modes']) for r in controls}=={tuple(range(k)) for k in range(1,14)})
    check('all fourteen-mode subsets',len(vac)==16384 and set(vac)=={tuple(j for j in range(14) if mask>>j&1) for mask in range(16384)})
    # Independent in-place interval Mobius transform (producer enumerates subsets).
    transformed=[vac[tuple(j for j in range(14) if mask>>j&1)] for mask in range(16384)]
    for bit in range(14):
        for mask in range(16384):
            if not(mask>>bit)&1:
                a,b=transformed[mask];c,d=transformed[mask|(1<<bit)];transformed[mask]=(a-d,b-c)
    law=physical['law'];check('physical modes',law['modes']==list(range(14)))
    for mask in range(16384):
        a,b=transformed[16383^mask];check('physical atom '+str(mask),tuple(law['raw_intervals'][mask])==(a,b) and tuple(law['intervals'][mask])==(max(F(0),a),min(F(1),b)))
    target=law['intervals'];check('target mass',sum(a for a,b in target)<=1<=sum(b for a,b in target));transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper'];check('transfer identity',physical['transfer_upper']==transfer)
    oldtarget=read('results/iteration47_target/target.json')['law']['intervals']
    for k in range(8192):check('old physical marginal '+str(k),sum(target[i][0] for i in range(k,16384,8192))<=oldtarget[k][1] and oldtarget[k][0]<=sum(target[i][1] for i in range(k,16384,8192)))
    for job in plan['jobs']:
        label=job['label'];raw=read(panel+'/'+label+'.stdout.json');r=read(panel+'/'+label+'.audit.json');old=read(job['reference']);tag=label+' '
        check(tag+'identity',r['job']==job and raw['modes']==144 and raw['cutoff']==job['cutoff'] and raw['samples']==17 and raw['prefix_depth']==14 and raw['prefix_reverse_verified']);check(tag+'unchanged public samples',raw['outputs']==old['outputs'] and raw['audit_paths']==old['audit_paths']);steps+=replay({**raw,'prefix_tree':[]})
        V=np.loadtxt(ROOT/job['csv'],delimiter=',');check(tag+'actual144input',V.shape==(288,288) and np.array_equal(V,inputs['shifted']))
        table={x['prefix']:x['threshold_hex'] for x in raw['prefix_tree']};by_prefix={x['prefix']:x for x in raw['prefix_tree']};check(tag+'all prefixes',len(table)==16383 and all(''.join(str((m>>j)&1) for j in range(k)) in table for k in range(14) for m in range(1<<k)));check(tag+'old thirteen levels',all(by_prefix[x['prefix']]==x for x in old['prefix_tree']))
        for prefix,row in by_prefix.items():
            q,prev,z=(float.fromhex(row[k]) for k in ('threshold_hex','previous_hex','proposed_hex'));check(tag+'quotient '+prefix,(math.isnan(q) if z==0 else math.isinf(q) and math.copysign(1,q)==math.copysign(1,z)*math.copysign(1,prev)) if prev==0 else q==float(F(z)/F(prev)))
            if prefix:
                parent=by_prefix[prefix[:-1]];pp=float.fromhex(parent['previous_hex']);pz=float.fromhex(parent['proposed_hex']);expected=pz if prefix[-1]=='0' else float(F(pp)-F(pz));check(tag+'continuity '+prefix,prev==expected)
        for s in range(17):
            for j in range(14):check(tag+f'sampled prefix {s}:{j}',table[raw['outputs'][s][:j]]==raw['audit_paths'][s][j][1])
        tv={}
        for model in ('continuous','uint64'):
            q=layer_law(table,14,model);check(tag+'independent law '+model,q==r['laws'][model]['distribution'] and sum(q)==1 and min(q)>=0);lo=sum(max(F(0),a-x,x-b) for x,(a,b) in zip(q,target))/2;hi=sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,target))/2;tv[model]=(lo,hi)
        q=r['laws']['uint64']['distribution'];event=[i for i,(x,(a,b)) in enumerate(zip(q,target)) if x>b];Q=sum(q[i] for i in event);P=(sum(target[i][0] for i in event),sum(target[i][1] for i in event));gap=max(F(0),Q-P[1]);lower=tv['uint64'][0];oldbounds=read('results/iteration47_prefix_validation/bounds.json');oldbound=next(x['full144_TV_lower'] for x in oldbounds if x['cutoff']==job['cutoff']);check(tag+'inherited bound positive',oldbound>0)
        bounds.append(dict(cutoff=job['cutoff'],prefix_TV_intervals=tv,full144_TV_lower=lower,full144_shifted_TV_lower=max(F(0),lower-transfer),old_thirteen_mode_lower=oldbound,strictly_strengthened=lower>oldbound,transfer_upper=transfer,event=dict(masks=event,sampler_probability=Q,target_probability_interval=P,TV_lower=gap)))
        for depth in range(1,15):
            size=1<<depth;qs=[sum(q[i] for i in range(k,16384,size)) for k in range(size)];ps=[(sum(target[i][0] for i in range(k,16384,size)),sum(target[i][1] for i in range(k,16384,size))) for k in range(size)];lo=sum(max(F(0),a-x,x-b) for x,(a,b) in zip(qs,ps))/2;hi=sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(qs,ps))/2;depth_rows.append(dict(cutoff=job['cutoff'],depth=depth,TV_lower=lo,TV_upper=hi,shifted_lower=max(F(0),lo-transfer)))
        print('verified',label,'lower',float(lower),flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'bounds.json',encode(bounds));save(out/'depth_bounds.json',encode(depth_rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),physical_certificates=cert_count,replayed_steps=steps,prefix_trees=4,source_hashes=hashes,dependencies=deps,scope='Full144law lower bounds only; no upper bound'));print('verified',len(checks),flush=True)
if __name__=='__main__':main()
