"""Independent selector cells, actual callback outputs and canonical prefix laws."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,gzip,hashlib,shutil,math,time,random
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
import iteration41_independent as independent
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0);independent.word_probability=lru_cache(maxsize=262144)(independent.word_probability)

def interval(q,p):return (sum(max(F(0),a-x,x-b) for x,(a,b) in zip(q,p))/2,sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,p))/2)

def main():
    started=time.perf_counter();out=ROOT/'results/iteration62_mixture_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration62_mixture','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];count=0;shards=[]
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
        raw=(ROOT/n).read_bytes();payload=gzip.decompress(raw);check('gzip '+n,len(raw)==meta['compressed_bytes'] and len(payload)==meta['raw_bytes'] and hashlib.sha256(payload).hexdigest()==meta['raw_sha256'] and raw[4:8]==bytes(4));return read(n)
    panel='results/iteration62_mixture';plan=read(panel+'/plan.json');summary=read(panel+'/summary.json');check('six groups and12 native complete',summary['planned']==summary['finished']==summary['completed']==6 and summary['new_native_runs']==12 and summary['new_native_samples']==68 and summary['unchanged_sources'] and summary['unchanged_dependencies'])
    for n,s in plan['source_hashes'].items():check('producer source '+n,digest(ROOT/n)==s==digest(ROOT/panel/'source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('producer dependency '+n,digest(ROOT/n)==s);deps[n]=s
    manifests={i:read(f'results/iteration{i}_manifest.json') for i in (35,37,42,56,61)}
    for n,s in plan['dependencies'].items():
        check('frozen dependency '+n,summary['dependencies'][n]==s)
        if not n.endswith('_manifest.json'):check('published dependency '+n,any(m['artifacts'].get(n)==s for m in manifests.values()))
    old37=read('results/iteration37_validation/validation.json');check('published37 validator',deps['results/iteration37_validation/validation.json']==manifests[37]['artifacts']['results/iteration37_validation/validation.json'] and old37['count']==old37['passed']);oldchecks=read('results/iteration37_validation/checks.json');check('published37 checks',deps['results/iteration37_validation/checks.json']==manifests[37]['artifacts']['results/iteration37_validation/checks.json'] and len(oldchecks)==old37['count'] and all(c['passed'] for c in oldchecks));oldnames={c['name'] for c in oldchecks}
    physical=read('results/iteration56_target/target.json');targets={0:physical['law']['intervals']};transfer=physical['transfer_upper'];check('same physical transfer',plan['transfer_upper']==transfer)
    old56=read('results/iteration56_prefix_validation/validation.json');check('published56 target audit',old56['count']==old56['passed'] and old56['dependencies']['results/iteration56_target/target.json']==deps['results/iteration56_target/target.json'])
    for start in (54,134):
        n=f'results/iteration37_local/target_{start:03d}.json';r=read(n);check('old37 target audit '+str(start),r['law']['modes']==list(range(start,start+10)) and manifests[37]['artifacts'][n]==deps[n] and all(f'block {start} atom {i}' in oldnames for i in range(1024)) and f'block {start} certificate count' in oldnames and f'block {start} mass' in oldnames);targets[start]=r['law']['intervals']
    selector=read(panel+'/selector.json');W=1<<64;bounds=[]
    for j in range(4):q,r=divmod(j*W,3);bounds.append(q+int(r!=0))
    counts=tuple(b-a for a,b in zip(bounds,bounds[1:]));weights=tuple(F(c,W) for c in counts);uniform_bias=sum(abs(w-F(1,3)) for w in weights)/2
    check('native selector exact',tuple(selector['native']['boundaries'])==tuple(bounds) and tuple(selector['native']['counts'])==counts and tuple(selector['native']['weights'])==weights and selector['native']['uniform_TV']==uniform_bias==F(2,3*W))
    for r in selector['native_probes']:check('native selector boundary '+str(r['word']),0<=r['choice']<3 and r['choice']*W<=3*r['word']<(r['choice']+1)*W)
    for r in selector['small_probes']:
        size=1<<r['bits'];hist=[0,0,0];check('small selector size '+str(r['bits']),len(r['choices'])==size)
        for w,c in enumerate(r['choices']):check(f'small selector {r["bits"]}:{w}',0<=c<3 and c*size<=3*w<(c+1)*size);hist[c]+=1
        check('small selector law '+str(r['bits']),tuple(hist)==tuple(r['law']['counts']) and tuple(F(c,size) for c in hist)==tuple(r['law']['weights']) and sum(abs(F(c,size)-F(1,3)) for c in hist)/2==r['law']['uniform_TV'])
    V=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];oldcomp=read('results/iteration61_order_validation/comparisons.json');groups=plan['groups'];check('fixed group plan',[(g['start'],g['depth'],g['cutoff'],g['selector_seed'],g['new_native']) for g in groups]==[(s,16 if s==0 else 10,k,620000+i,s!=0) for i,(s,k) in enumerate((s,k) for s in (0,54,134) for k in (3,4))])
    results=[];all_nodes=new_nodes=steps=new_samples=selected_outputs=0
    for gi,g in enumerate(groups):
        label=g['label'];start=g['start'];depth=g['depth'];k=g['cutoff'];size=1<<depth;meta=read(panel+'/'+label+'.storage.json');record=compressed(panel+'/'+label+'.audit.json.gz',meta['audit']);check(label+' group identity',record['group']==g and g['index']==gi)
        d=record['dispatch'];rng=random.Random(620000+gi);words=[rng.getrandbits(64) for _ in range(17)];check(label+' selector replay',d['selector_words']==words and len(d['outputs'])==17);choices=[next(j for j in range(3) if bounds[j]<=w<bounds[j+1]) for w in words];cc=[choices.count(j) for j in range(3)];check(label+' chosen counts',choices==d['choices'] and cc==d['component_counts'] and sum(cc)==17 and d['selector']==selector['native']);selected_outputs+=17
        laws=[];component_metrics=[];raws=[]
        for oi in range(3):
            name=f'{label}_order{oi}';perm=g['permutations'][oi];expected_prefix=sorted(range(start,start+depth),key=(lambda p:p) if oi==0 else (lambda p:-p) if oi==1 else (lambda p:((p-start)%2,p)));check(name+' fixed full order',perm==expected_prefix+[p for p in range(144) if p not in expected_prefix] and sorted(perm)==list(range(144)))
            storage=read(panel+'/'+name+'.storage.json');rec=compressed(panel+'/'+name+'.audit.json.gz',storage['audit']);check(name+' callback request',rec['index']==oi and rec['requested_samples']==cc[oi])
            if g['new_native']:
                csv=ROOT/panel/(name+'.csv');deps[csv.relative_to(ROOT).as_posix()]=digest(csv);cov=np.loadtxt(csv,delimiter=',');expected=np.empty((288,288))
                for a in range(288):
                    for b in range(288):expected[a,b]=V[perm[a%144]+144*(a//144),perm[b%144]+144*(b//144)]
                check(name+' complete covariance',np.array_equal(cov,expected) and np.array_equal(cov,cov.T));raw=compressed(panel+'/'+name+'.stdout.json.gz',storage['stdout']);check(name+' actual native seeds and counts',raw['samples']==cc[oi] and raw['seed']==620100+10*gi+oi==g['component_seeds'][oi]);new_samples+=raw['samples'];check(name+' source path',rec['source']==panel+'/'+name+'.stdout.json.gz')
            else:
                check(name+' frozen replay source',rec['source']==g['reused_sources'][oi] and rec['command'] is None);raw=read(rec['source']);check(name+' old count and seed',raw['samples']==17 and raw['seed']==391440)
            check(name+' raw identity',raw['modes']==144 and raw['cutoff']==k and raw['prefix_depth']==depth and raw['prefix_reverse_verified']);raws.append(raw);replayed=replay({**raw,'prefix_tree':[]});check(name+' replay count',rec['replayed_steps']==replayed);steps+=replayed
            table={r['prefix']:r for r in raw['prefix_tree']};check(name+' complete tree',len(table)==size-1 and all(set(p)<=set('01') and len(p)<depth for p in table));all_nodes+=len(table);new_nodes+=len(table) if g['new_native'] else 0
            for prefix,r in table.items():
                q,prev,z=(float.fromhex(r[t]) for t in ('threshold_hex','previous_hex','proposed_hex'));ok=(math.isnan(q) if z==0 else math.isinf(q) and math.copysign(1,q)==math.copysign(1,z)*math.copysign(1,prev)) if prev==0 else q==float(F(z)/F(prev));check(name+' quotient '+prefix,ok)
                if prefix:
                    parent=table[prefix[:-1]];pp=float.fromhex(parent['previous_hex']);pz=float.fromhex(parent['proposed_hex']);check(name+' continuity '+prefix,prev==(pz if prefix[-1]=='0' else float(F(pp)-F(pz))))
            for i,s in enumerate(raw['outputs']):
                for j in range(depth):check(name+f' public prefix {i}:{j}',raw['audit_paths'][i][j][1]==table[s[:j]]['threshold_hex'])
            sequence=independent.layer_law({p:r['threshold_hex'] for p,r in table.items()},depth,'uint64');positions=[perm.index(start+j) for j in range(depth)];canonical=[]
            for mask in range(size):
                seqmask=sum(((mask>>j)&1)<<positions[j] for j in range(depth));q=sequence[seqmask];check(name+' canonical atom '+str(mask),q==rec['canonical_distribution'][mask]);canonical.append(q)
            check(name+' normalized component',sum(canonical)==1 and min(canonical)>=0);b=interval(canonical,targets[start]);check(name+' component interval',tuple(rec['TV_interval'])==b);laws.append(canonical);component_metrics.append(b)
        ranks=[0,0,0]
        for i,c in enumerate(choices):
            rank=ranks[c];ranks[c]+=1;raw=raws[c]['outputs'][rank];perm=g['permutations'][c];expected=''.join(raw[perm.index(j)] for j in range(144));check(label+' merged output '+str(i),d['component_indices'][i]==rank and d['outputs'][i]==expected)
        ideal=[sum(x,F(0))/3 for x in zip(*laws)];finite=[a+(2*q0-q1-q2)/(3*W) for a,q0,q1,q2 in zip(ideal,*laws)];mix=record['mixture'];check(label+' selector binding',mix['selector']==selector['native'])
        for i,(a,b) in enumerate(zip(ideal,finite)):check(label+' exact finite mixture '+str(i),a==mix['ideal_distribution'][i] and b==mix['distribution'][i])
        check(label+' normalized mixture',sum(finite)==1 and min(finite)>=0);delta=sum(abs(a-b) for a,b in zip(ideal,finite))/2;refined=sum(abs(q0-(q1+q2)/2) for q0,q1,q2 in zip(*laws))/(3*W);check(label+' exact selector error',delta==mix['ideal_distance']==refined and delta<=uniform_bias)
        fv=interval(finite,targets[start]);iv=interval(ideal,targets[start]);check(label+' complete mixture distances',fv==tuple(record['TV_interval']) and iv==tuple(record['ideal_TV_interval']) and [list(b) for b in component_metrics]==record['component_TV_intervals']);check(label+' mixture convexity',fv[1]<=sum(w*b[1] for w,b in zip(weights,component_metrics)))
        if start==0:
            old=next(r for r in oldcomp if r['modes']==144 and r['cutoff']==k);check(label+' exact old ideal anchor',tuple(old['variants'][3]['TV_interval'])==iv and all(tuple(old['variants'][j]['TV_interval'])==b for j,b in enumerate(component_metrics)))
        variants=[]
        for kind,b in zip(('identity','reverse','even_odd','ideal_mixture','finite_mixture'),component_metrics+[iv,fv]):
            baseline=component_metrics[0];status='baseline' if kind=='identity' else 'improved' if b[1]<baseline[0] else 'worsened' if b[0]>baseline[1] else 'unresolved';variants.append(dict(kind=kind,TV_interval=b,relative_to_identity=status,improvement_factor_interval=(baseline[0]/b[1],baseline[1]/b[0]) if b[0]>0 else None,full144_TV_lower=b[0],shifted_full144_lower=max(F(0),b[0]-transfer)))
        results.append(dict(label=label,start=start,depth=depth,cutoff=k,new_native=g['new_native'],variants=variants,selector_counts=counts,selector_bias_bound=uniform_bias,finite_to_ideal_TV=delta,public_component_counts=cc,mixture_cancellation_interval=(sum(w*b[0] for w,b in zip(weights,component_metrics))-fv[1],sum(w*b[1] for w,b in zip(weights,component_metrics))-fv[0])));print('verified',label,'TV',float(fv[0]),'change%',100*(float(fv[0]/component_metrics[0][0])-1),flush=True)
    check('traversal totals',all_nodes==405486 and new_nodes==12276 and steps==24480 and new_samples==68 and selected_outputs==102);check('unchanged sources',all(digest(ROOT/n)==s for n,s in hashes.items()));check('unchanged dependencies',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(results));flush();save(out/'validation.json',dict(count=count,passed=count,check_shards=shards,groups=6,new_native_runs=12,new_native_samples=new_samples,public_mixture_outputs=selected_outputs,all_prefix_nodes=all_nodes,new_prefix_nodes=new_nodes,replayed_steps=steps,source_hashes=hashes,dependencies=deps,seconds=time.perf_counter()-started,scope='Finite-word dispatch and actual144 prefix marginals on the same physical input; no full144 error improvement theorem.'));print('verified',count,flush=True)
if __name__=='__main__':main()
