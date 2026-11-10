"""Independent label, matrix, word-law and distance audit of fixed-order panels."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,gzip,hashlib,shutil,math,time
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
import iteration41_independent as independent
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
independent.word_probability=lru_cache(maxsize=262144)(independent.word_probability)

def tv(q,p):
    return (sum(max(F(0),a-x,x-b) for x,(a,b) in zip(q,p))/2,sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,p))/2)

def main():
    out=ROOT/'results/iteration61_order_validation';out.mkdir(exist_ok=False);started=time.perf_counter()
    names=[f'experiments/{n}.py' for n in ('verify_iteration61_order','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];count=0;shards=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):
        p=ROOT/n;deps[n]=digest(p);b=p.read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def flush():
        if checks:name=f'checks_{len(shards):03d}.json';save(out/name,checks);shards.append(name);checks.clear()
    def check(n,ok):
        nonlocal count
        count+=1;checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
        if len(checks)>=50000:flush()
    def compressed(n,meta):
        payload=(ROOT/n).read_bytes();b=gzip.decompress(payload);check('stored bytes '+n,len(payload)==meta['compressed_bytes'] and len(b)==meta['raw_bytes'] and hashlib.sha256(b).hexdigest()==meta['raw_sha256'] and payload[4:8]==bytes(4));return read(n)
    panel='results/iteration61_order';plan=read(panel+'/plan.json');summary=read(panel+'/summary.json');check('all15 finished',summary['planned']==summary['finished']==summary['completed']==15 and summary['unchanged_sources'] and summary['unchanged_dependencies'])
    for n,s in plan['source_hashes'].items():check('producer source '+n,digest(ROOT/n)==s==digest(ROOT/panel/'source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('producer dependency '+n,digest(ROOT/n)==s);deps[n]=s
    manifests={it:read(f'results/iteration{it}_manifest.json') for it in (35,37,39,42,56)}
    for n,s in plan['dependencies'].items():
        check('frozen dependency '+n,summary['dependencies'][n]==s)
        if not n.endswith('_manifest.json'):check('previous published byte '+n,any(m['artifacts'].get(n)==s for m in manifests.values()))
    validation=read('results/iteration56_prefix_validation/validation.json');physical=read('results/iteration56_target/target.json');check('reused physical validation',validation['count']==validation['passed'] and validation['dependencies']['results/iteration56_target/target.json']==deps['results/iteration56_target/target.json']);transfer=physical['transfer_upper'];check('unchanged transfer',plan['transfer_upper']==transfer)
    targets={144:physical['law']['intervals'],10:read('results/iteration37_local/target_000.json')['law']['intervals']};V=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];check('base covariance symmetric',np.array_equal(V,V.T))
    jobs=plan['jobs'];expected=[]
    for kind in ('identity','reverse','even_odd'):
        for k in (1,4):expected.append((10,10,kind,k,391000+k,'control10'))
    expected.append((144,16,'identity',1,391440,'control144'))
    for kind in ('reverse','even_odd'):
        for k in range(1,5):expected.append((144,16,kind,k,391440,'main144'))
    check('predeclared jobs',[(j['modes'],j['depth'],j['kind'],j['cutoff'],j['seed'],j['scope']) for j in jobs]==expected and plan['samples']==17)
    baselines={};laws={};rows=[];steps=0;tree_nodes=0;new_nodes=0
    def audit_tree(label,raw,record,target,perm,old=False):
        nonlocal steps,tree_nodes,new_nodes
        depth=len(perm);table={r['prefix']:r for r in raw['prefix_tree']};check(label+' tree complete',len(table)==(1<<depth)-1 and all(set(p)<=set('01') and len(p)<depth for p in table));tree_nodes+=len(table);new_nodes+=0 if old else len(table);steps+=replay({**raw,'prefix_tree':[]})
        for prefix,r in table.items():
            q,prev,z=(float.fromhex(r[k]) for k in ('threshold_hex','previous_hex','proposed_hex'))
            quotient=(math.isnan(q) if z==0 else math.isinf(q) and math.copysign(1,q)==math.copysign(1,z)*math.copysign(1,prev)) if prev==0 else q==float(F(z)/F(prev));check(label+' quotient '+prefix,quotient)
            if prefix:
                parent=table[prefix[:-1]];pp=float.fromhex(parent['previous_hex']);pz=float.fromhex(parent['proposed_hex']);check(label+' continuity '+prefix,prev==(pz if prefix[-1]=='0' else float(F(pp)-F(pz))))
        for s,output in enumerate(raw['outputs']):
            for j in range(depth):check(label+f' public {s}:{j}',table[output[:j]]['threshold_hex']==raw['audit_paths'][s][j][1])
        inverse=[perm.index(j) for j in range(depth)];indices=[]
        for physical_mask in range(1<<depth):
            bits=tuple((physical_mask//(2**j))%2 for j in range(depth));sequence_mask=sum(bits[p]*2**j for j,p in enumerate(perm));check(label+' inverse label '+str(physical_mask),sum(((sequence_mask>>inverse[j])&1)<<j for j in range(depth))==physical_mask);indices.append(sequence_mask)
        table_hex={p:r['threshold_hex'] for p,r in table.items()};canonical={};metrics={}
        for model in ('continuous','uint64'):
            q=independent.layer_law(table_hex,depth,model);check(label+' independent law '+model,q==record['laws'][model]['distribution'] and sum(q)==1 and min(q)>=0);canonical[model]=[q[i] for i in indices];metrics[model]=tv(canonical[model],target)
            if not old:check(label+' TV '+model,tuple(record['TV_intervals'][model])==metrics[model])
        coupling=sum(abs(a-b) for a,b in zip(canonical['continuous'],canonical['uint64']))/2;check(label+' RNG coupling',coupling<=depth*(F(1,1<<53)+F(1,1<<64)))
        if not old:check(label+' RNG exact',coupling==record['uniform_model_distance'])
        return canonical['uint64'],metrics,coupling
    for k in range(1,5):
        label=f'baseline_k{k}';raw=read(f'results/iteration56_prefix/global_{k}.stdout.json.gz');rec=read(f'results/iteration56_prefix/global_{k}.audit.json.gz');q,metrics,coupling=audit_tree(label,raw,rec,targets[144],list(range(16)),True);previous=next(x for x in read('results/iteration56_prefix_validation/bounds.json') if x['cutoff']==k);check(label+' previous interval',all(tuple(previous['prefix_TV_intervals'][m])==metrics[m] for m in metrics));baselines[k]=q;laws[(144,'identity',k)]=q;rows.append(dict(label=label,modes=144,kind='identity',cutoff=k,TV_intervals=metrics,uniform_model_distance=coupling,reused=True));print('verified',label,flush=True)
    for job in jobs:
        label=job['label'];n=job['modes'];depth=job['depth'];k=job['cutoff'];kind=job['kind'];perm=job['permutation'];prefix=list(range(depth))
        if kind=='reverse':prefix=sorted(prefix,reverse=True)
        elif kind=='even_odd':prefix=sorted(prefix,key=lambda x:(x%2,x))
        check(label+' fixed permutation',perm==prefix+list(range(depth,n)) and sorted(perm)==list(range(n)))
        csv=ROOT/panel/(label+'.csv');deps[csv.relative_to(ROOT).as_posix()]=digest(csv);cov=np.loadtxt(csv,delimiter=',');expected_matrix=np.empty_like(cov)
        for i in range(2*n):
            for j in range(2*n):expected_matrix[i,j]=V[perm[i%n]+144*(i//n),perm[j%n]+144*(j//n)]
        check(label+' complete q/p covariance',np.array_equal(cov,expected_matrix) and np.array_equal(cov,cov.T))
        storage=read(panel+'/'+label+'.storage.json');raw=compressed(panel+'/'+label+'.stdout.json.gz',storage['stdout']);record=compressed(panel+'/'+label+'.audit.json.gz',storage['audit']);check(label+' identity',record['job']==job and raw['modes']==n and raw['cutoff']==k and raw['samples']==17 and raw['prefix_depth']==depth and raw['prefix_reverse_verified'])
        if kind=='identity':
            old=read('results/iteration56_prefix/global_1.stdout.json.gz') if n==144 else read(f'results/iteration39_local/case_{k-1:02d}.stdout.json')
            check(label+' exact old outputs',raw['outputs']==old['outputs'][:17] and raw['audit_paths']==old['audit_paths'][:17] and {r['prefix']:r for r in raw['prefix_tree']}=={r['prefix']:r for r in old['prefix_tree']})
        inverse=[perm.index(j) for j in range(n)];mapped=[''.join(s[inverse[j]] for j in range(n)) for s in raw['outputs']];check(label+' physical output labels',mapped==record['mapped_outputs'])
        q,metrics,coupling=audit_tree(label,raw,record,targets[n],perm[:depth]);check(label+' replay count',record['replayed_steps']==17*n)
        if n==144 and kind=='identity':check(label+' exact baseline law',q==baselines[k])
        else:laws[(n,kind,k)]=q;rows.append(dict(label=label,modes=n,kind=kind,cutoff=k,TV_intervals=metrics,uniform_model_distance=coupling,reused=False))
        print('verified',label,'TV',float(metrics['uint64'][0]),flush=True)
    groups=[]
    for n,ks in ((144,(1,2,3,4)),(10,(1,4))):
        for k in ks:
            qs=[laws[(n,kind,k)] for kind in ('identity','reverse','even_odd')];bounds=[tv(q,targets[n]) for q in qs];mixture=[sum(x,F(0))/3 for x in zip(*qs)];check(f'{n} K{k} mixture normalized',sum(mixture)==1 and min(mixture)>=0);mix=tv(mixture,targets[n]);average=(sum(x[0] for x in bounds)/3,sum(x[1] for x in bounds)/3);check(f'{n} K{k} mixture convexity',mix[1]<=average[1]);pair_distances={kind:sum(abs(a-b) for a,b in zip(qs[0],q))/2 for kind,q in zip(('reverse','even_odd'),qs[1:])}
            variants=[]
            for kind,b in zip(('identity','reverse','even_odd','mixture'),bounds+[mix]):
                status='baseline' if kind=='identity' else 'improved' if b[1]<bounds[0][0] else 'worsened' if b[0]>bounds[0][1] else 'unresolved'
                ratio=(bounds[0][0]/b[1],bounds[0][1]/b[0]) if b[0]>0 else None
                variants.append(dict(kind=kind,TV_interval=b,relative_to_identity=status,improvement_factor_interval=ratio,full_law_TV_lower=b[0],shifted_full_law_lower=max(F(0),b[0]-transfer) if n==144 else None))
            groups.append(dict(modes=n,cutoff=k,variants=variants,program_TV_from_identity=pair_distances,mixture_average_component_TV=average,mixture_cancellation_interval=(average[0]-mix[1],average[1]-mix[0]),mixture_rule='ideal uniform choice among three orders'))
    check('expected traversal totals',new_nodes==9*65535+6*1023 and tree_nodes==13*65535+6*1023 and steps==13*17*144+6*17*10)
    check('unchanged sources',all(digest(ROOT/n)==s for n,s in hashes.items()));check('unchanged dependencies',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'comparisons.json',encode(groups));flush();save(out/'validation.json',dict(count=count,passed=count,check_shards=shards,new_runs=15,main_new_runs=8,control_new_runs=7,reused_baselines=4,new_prefix_nodes=new_nodes,all_prefix_nodes=tree_nodes,replayed_steps=steps,source_hashes=hashes,dependencies=deps,seconds=time.perf_counter()-started,scope='Actual144 16-bit marginal intervals and full-law lower bounds; complete10 controls; prior physical certificate validation reused by hash.'))
    print('verified',count,'new_nodes',new_nodes,flush=True)
if __name__=='__main__':main()
