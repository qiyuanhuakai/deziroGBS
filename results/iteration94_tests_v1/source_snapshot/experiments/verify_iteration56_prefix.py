"""Independent actual144-prefix laws and revalidation of physical target certificates.

The matrix certificate checker is retained verbatim from verify_iteration37;
the producer is GaussianEnvelope, and this rechecks its exact witnesses.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,sys,shutil,math,gzip,hashlib,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from iteration41_independent import layer_law
from iteration53_independent import expand_record,direct_integer_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def integers(A):
    a=np.asarray(A);values=[F(float(x)) for x in a.flat];d=math.lcm(*(v.denominator for v in values));return np.array([v.numerator*(d//v.denominator) for v in values],dtype=object).reshape(a.shape),d


def outward(a,b):
    scale=1<<256;return F(a.numerator*scale//a.denominator,scale),F(-((-b.numerator*scale)//b.denominator),scale)


def main():
    out=ROOT/'results/iteration56_prefix_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration56_prefix','iteration53_independent','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};checks=[];deps={};check_count=0;check_shards=[];cert_count=0;steps=0;depth_rows=[];bounds=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):
        deps[n]=digest(ROOT/n);b=(ROOT/n).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def flush():
        if checks:
            name=f'checks_{len(check_shards):03d}.json';save(out/name,checks);check_shards.append(name);checks.clear()
    def compressed(n,meta,sizekey,hashkey):
        b=(ROOT/n).read_bytes();deps[n]=digest(ROOT/n);payload=gzip.decompress(b);check('gzip bytes '+n,meta['path']==Path(n).name and len(b)==meta['compressed_bytes'] and len(payload)==meta[sizekey] and hashlib.sha256(payload).hexdigest()==meta[hashkey] and b[4:8]==bytes(4));return decode(json.loads(payload))
    def check(n,ok):
        nonlocal check_count
        check_count+=1;checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
        if len(checks)>=20000:flush()
    def real_certificate(c,A,den,label):
        nonlocal cert_count
        cert_count+=1;Z=np.array(c['real'],dtype=object);d=c['denominator'];R=np.array(c['R_real'],dtype=object);s=c['scale'];n=len(Z)
        check(label+' exact matrix',np.array_equal(Z*den,A*d) and np.array_equal(Z,Z.T));check(label+' real triangular witness',not any(int(x) for row in c['imag'] for x in row) and not any(int(x) for row in c['R_imag'] for x in row) and all(int(R[i,i])>0 for i in range(n)) and not any(int(R[i,j]) for i in range(n) for j in range(i+1,n)))
        E=(R@Z)@R.T-np.eye(n,dtype=object)*d*s*s;delta=F(max(sum(abs(int(x)) for x in row) for row in E),d*s*s);trace=F(sum(int(x)**2 for x in R.flat),s*s);lam=(1-delta)/trace;check(label+' exact residual',0<=delta<1 and delta==c['diagnostics']['exact_residual_upper']);check(label+' eigenvalue',lam==c['diagnostics']['minimum_eigenvalue_lower']>0);detR=math.prod((F(int(R[j,j]),s) for j in range(n)),start=F(1));bounds=outward((1-delta)**n/detR**2,(1+delta)**n/detR**2);check(label+' determinant',tuple(c['determinant_interval'])==bounds);return delta,trace,lam
    def model_check(rec,arrays,label):
        B=arrays['B'];noise=arrays['F'];m=len(B);br,dr=integers(B.real);bi,di=integers(B.imag);d=max(dr,di);S=np.block([[br*(d//dr),bi*(d//di)],[bi*(d//di),-br*(d//dr)]]);K=np.eye(2*m,dtype=object)*d-S;check(label+' symmetric B',np.array_equal(B,B.T));delta,trace,_=real_certificate(rec['core'],K,d,label+' core');Z,dz=integers(noise);td=2*dz*dz*d;T=Z.T@(K@Z)+np.eye(noise.shape[1],dtype=object)*td;real_certificate(rec['noise'],T,td,label+' noise');R=np.array(rec['core']['R_real'],dtype=object);dy=rec['core']['scale']**2;dc=max(dy,dz*dz);C=(2*(R.T@R)-np.eye(2*m,dtype=object)*dy)*(dc//dy)+(Z@Z.T)*(dc//(dz*dz));radius=2*delta*trace/(1-delta);check(label+' center and radius',dc==rec['center_denominator'] and np.array_equal(C,np.array(rec['center_numerators'],dtype=object)) and radius==rec['operator_radius']);return C,dc,radius
    def enclosure_check(r,C,dc,radius,label):
        modes=tuple(r['modes']);m=len(C)//2;ix=modes+tuple(j+m for j in modes);A=C[np.ix_(ix,ix)]+np.eye(len(ix),dtype=object)*dc;_,_,lam=real_certificate(r['certificate'],A,2*dc,label);eta=radius/(2*lam);check(label+' relative perturbation',eta==r['relative_perturbation'] and 0<=eta<1);lo,hi=r['certificate']['determinant_interval'];bounds=outward(lo*(1-eta)**len(ix),hi*(1+eta)**len(ix));check(label+' perturbed determinant',tuple(r['perturbed_determinant_interval'])==bounds);a,b=r['raw_vacuum_interval'];check(label+' exact root inequalities',a>=0 and a*a<=1/bounds[1]<=1/bounds[0]<=b*b);check(label+' physical probability intersection',tuple(r['vacuum_interval'])==(max(F(0),a),min(F(1),b)));return tuple(r['vacuum_interval'])
    panel='results/iteration56_prefix';plan=read(panel+'/plan.json');summary=read(panel+'/summary.json');targetdir='results/iteration56_target';targetplan=read(targetdir+'/plan.json');targetsummary=read(targetdir+'/summary.json')
    check('four completed runs',summary['planned']==summary['finished']==summary['completed']==4 and summary['unchanged_sources'] and summary['unchanged_dependencies']);check('target complete',targetsummary['completed'] and targetsummary['enclosures']==65535 and targetsummary['atoms']==65536 and targetsummary['unchanged_sources'])
    for folder,p in ((panel,plan),(targetdir,targetplan)):
        for n,sha in p['source_hashes'].items():check(folder+' source '+n,digest(ROOT/n)==sha==digest(ROOT/folder/'source_snapshot'/n))
        for n,sha in p['dependencies'].items():check(folder+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
    inputs=np.load(ROOT/'results/iteration35_panel/inputs.npz');base=read('results/iteration35_panel/case_07.audit.json');C,dc,radius=model_check(base,inputs,'physical model');physical=read(targetdir+'/target.json');vac={(): (F(1),F(1))}
    selected={}
    check('incremental target counts',targetsummary['new_enclosures']==32768 and targetsummary['reused_enclosures']==32767 and targetplan['reused_nonempty_subsets']==32767 and targetplan['new_nonempty_subsets']==32768)
    manifests={i:read(f'results/iteration{i}_manifest.json') for i in (47,49,53)};storage_by_iteration={53:read('results/iteration53_target/storage.json'),56:read(targetdir+'/storage.json')};check('128 new compact storage entries',len(storage_by_iteration[56])==128)
    for iteration,count in ((47,32),(49,32),(53,64),(56,128)):
        directory=f'results/iteration{iteration}_target'
        for shard in range(count):
            name=directory+f'/certificates_{shard:02d}.json'+('.gz' if iteration in (53,56) else '')
            if iteration in (53,56):
                meta=storage_by_iteration[iteration][shard];check('compact path '+name,meta['path']==Path(name).name);records=compressed(name,meta,'compact_json_bytes','compact_json_sha256');check('compact record count '+name,len(records)==meta['records']);records=[expand_record(r,C,dc) for r in records];full=(json.dumps(encode(records),indent=2)+'\n').encode('utf-8');check('expanded canonical bytes '+name,len(full)==meta['expanded_json_bytes'])
                if iteration==53:check('bound reused compact shard '+name,deps[name]==manifests[53]['artifacts'][name]==targetplan['dependencies'][name])
            else:
                records=read(name);check('bound reused shard '+name,deps[name]==manifests[iteration]['artifacts'][name]==targetplan['dependencies'][name])
            for enclosure in records:
                key=tuple(enclosure['modes']);check('unique and correct old/new subset '+str(key),key not in vac and ((15 in key)==(iteration==56)) and (iteration!=53 or 14 in key) and (iteration!=49 or 13 in key));vac[key]=enclosure_check(enclosure,C,dc,radius,'target '+str(key))
                if iteration!=56 and key==tuple(range(len(key))):selected[key]=enclosure
            print('verified certificate shard',iteration,shard,flush=True)
    controls=read(targetdir+'/recomputed_controls.json');check('fifteen fresh old controls',len(controls)==15 and all(r==selected[tuple(r['modes'])] for r in controls) and {tuple(r['modes']) for r in controls}=={tuple(range(k)) for k in range(1,16)})
    check('all sixteen-mode subsets',len(vac)==65536 and set(vac)=={tuple(j for j in range(16) if mask>>j&1) for mask in range(65536)})
    # Independent explicit inclusion over every clicked subset: no Mobius transform.
    ordered=[vac[tuple(j for j in range(16) if mask>>j&1)] for mask in range(65536)];begin=time.perf_counter();raw_intervals,intervals,terms=direct_integer_law(ordered);direct_seconds=time.perf_counter()-begin
    law=physical['law'];check('physical modes',law['modes']==list(range(16)));check('integer operation counts',terms==3**16 and law['interval_subtractions']==16*2**15 and all((x*law['common_denominator']).denominator==1 for pair in ordered for x in pair))
    for mask in range(65536):check('physical atom '+str(mask),tuple(law['raw_intervals'][mask])==raw_intervals[mask] and tuple(law['intervals'][mask])==intervals[mask])
    save(out/'transform.json',dict(direct_integer_inclusion_seconds=direct_seconds,direct_terms=terms,producer_interval_subtractions=law['interval_subtractions']))
    target=law['intervals'];check('target mass',sum(a for a,b in target)<=1<=sum(b for a,b in target));transfer=base['transfers']['-40']['all_measurement_TV_transfer_upper'];check('transfer identity',physical['transfer_upper']==transfer)
    oldlaw=read('results/iteration53_target/target.json')['law'];oldtarget=oldlaw['intervals']
    oldraw,oldintervals,oldterms=direct_integer_law(ordered[:32768]);check('old15 inclusion operation count',oldterms==3**15)
    for i in range(32768):check('exact old15 transform '+str(i),tuple(oldlaw['raw_intervals'][i])==oldraw[i] and tuple(oldtarget[i])==oldintervals[i])
    for k in range(32768):check('old physical marginal '+str(k),sum(target[i][0] for i in range(k,65536,32768))<=oldtarget[k][1] and oldtarget[k][0]<=sum(target[i][1] for i in range(k,65536,32768)))
    for job in plan['jobs']:
        label=job['label'];storage=read(panel+'/'+label+'.storage.json');raw=compressed(panel+'/'+label+'.stdout.json.gz',storage['stdout'],'raw_bytes','raw_sha256');r=compressed(panel+'/'+label+'.audit.json.gz',storage['audit'],'raw_bytes','raw_sha256');old=read(job['reference']);tag=label+' '
        check(tag+'identity',r['job']==job and raw['modes']==144 and raw['cutoff']==job['cutoff'] and raw['samples']==17 and raw['prefix_depth']==16 and raw['prefix_reverse_verified']);check(tag+'unchanged public samples',raw['outputs']==old['outputs'] and raw['audit_paths']==old['audit_paths']);steps+=replay({**raw,'prefix_tree':[]})
        V=np.loadtxt(ROOT/job['csv'],delimiter=',');check(tag+'actual144input',V.shape==(288,288) and np.array_equal(V,inputs['shifted']))
        table={x['prefix']:x['threshold_hex'] for x in raw['prefix_tree']};by_prefix={x['prefix']:x for x in raw['prefix_tree']};check(tag+'all prefixes',len(table)==65535 and all(''.join(str((m>>j)&1) for j in range(k)) in table for k in range(16) for m in range(1<<k)));check(tag+'old fifteen levels',all(by_prefix[x['prefix']]==x for x in old['prefix_tree']))
        for prefix,row in by_prefix.items():
            q,prev,z=(float.fromhex(row[k]) for k in ('threshold_hex','previous_hex','proposed_hex'));check(tag+'quotient '+prefix,(math.isnan(q) if z==0 else math.isinf(q) and math.copysign(1,q)==math.copysign(1,z)*math.copysign(1,prev)) if prev==0 else q==float(F(z)/F(prev)))
            if prefix:
                parent=by_prefix[prefix[:-1]];pp=float.fromhex(parent['previous_hex']);pz=float.fromhex(parent['proposed_hex']);expected=pz if prefix[-1]=='0' else float(F(pp)-F(pz));check(tag+'continuity '+prefix,prev==expected)
        for s in range(17):
            for j in range(16):check(tag+f'sampled prefix {s}:{j}',table[raw['outputs'][s][:j]]==raw['audit_paths'][s][j][1])
        tv={}
        for model in ('continuous','uint64'):
            q=layer_law(table,16,model);check(tag+'independent law '+model,q==r['laws'][model]['distribution'] and sum(q)==1 and min(q)>=0);lo=sum(max(F(0),a-x,x-b) for x,(a,b) in zip(q,target))/2;hi=sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,target))/2;tv[model]=(lo,hi)
        q=r['laws']['uint64']['distribution'];event=[i for i,(x,(a,b)) in enumerate(zip(q,target)) if x>b];Q=sum(q[i] for i in event);P=(sum(target[i][0] for i in event),sum(target[i][1] for i in event));gap=max(F(0),Q-P[1]);lower=tv['uint64'][0];oldbounds=read('results/iteration53_prefix_validation/bounds.json');oldbound=next(x['full144_TV_lower'] for x in oldbounds if x['cutoff']==job['cutoff']);check(tag+'inherited bound positive',oldbound>0)
        bounds.append(dict(cutoff=job['cutoff'],prefix_TV_intervals=tv,full144_TV_lower=lower,full144_shifted_TV_lower=max(F(0),lower-transfer),old_fifteen_mode_lower=oldbound,strictly_strengthened=lower>oldbound,transfer_upper=transfer,event=dict(masks=event,sampler_probability=Q,target_probability_interval=P,TV_lower=gap)))
        for depth in range(1,17):
            size=1<<depth;qs=[sum(q[i] for i in range(k,65536,size)) for k in range(size)];ps=[(sum(target[i][0] for i in range(k,65536,size)),sum(target[i][1] for i in range(k,65536,size))) for k in range(size)];lo=sum(max(F(0),a-x,x-b) for x,(a,b) in zip(qs,ps))/2;hi=sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(qs,ps))/2;depth_rows.append(dict(cutoff=job['cutoff'],depth=depth,TV_lower=lo,TV_upper=hi,shifted_lower=max(F(0),lo-transfer)))
        print('verified',label,'lower',float(lower),flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'bounds.json',encode(bounds));save(out/'depth_bounds.json',encode(depth_rows));flush();save(out/'validation.json',dict(count=check_count,passed=check_count,check_shards=check_shards,physical_certificates=cert_count,old_transform_atoms=32768,replayed_steps=steps,prefix_trees=4,source_hashes=hashes,dependencies=deps,scope='Full144law lower bounds only; no upper bound'));print('verified',check_count,flush=True)
if __name__=='__main__':main()
