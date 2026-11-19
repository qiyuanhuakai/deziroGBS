"""Actual finite-word mixture dispatch on new144 windows plus frozen replay anchors."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,gzip,hashlib,shutil,subprocess,time,traceback,random
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from gbslab.frugal_law import full_law,distance_interval
from gbslab.finite_order_mixture import select_order,selector_law,mixture_law,sample_order_mixture
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration62_mixture';out.mkdir(exist_ok=False);names=['research/iteration62_mixture_plan.md',*[f'experiments/{n}.py' for n in ('iteration62_mixture','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};manifests={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def reuse(n,it=None):
        deps[n]=digest(ROOT/n)
        if it is not None:
            if it not in manifests:
                name=f'results/iteration{it}_manifest.json';deps[name]=digest(ROOT/name);manifests[it]=json.loads((ROOT/name).read_text())
            assert deps[n]==manifests[it]['artifacts'][n],n
        return ROOT/n
    def read(n,it=None):
        b=reuse(n,it).read_bytes();return decode(json.loads(gzip.decompress(b) if n.endswith('.gz') else b))
    def compressed(name,record):
        raw=record if isinstance(record,bytes) else (json.dumps(encode(record),indent=2)+'\n').encode();data=gzip.compress(raw,compresslevel=6,mtime=0);(out/name).write_bytes(data);return dict(path=name,raw_bytes=len(raw),compressed_bytes=len(data),raw_sha256=hashlib.sha256(raw).hexdigest())
    exe=reuse('results/iteration42_build/frugal_prefix.exe',42);build=read('results/iteration42_build/build.json',42);assert digest(exe)==build['executable_sha256']
    for p in exe.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix(),42)
    V=np.load(reuse('results/iteration35_panel/inputs.npz',35))['shifted'];physical=read('results/iteration56_target/target.json',56);targets={0:physical['law']['intervals']}
    read('results/iteration56_prefix_validation/validation.json',56);read('results/iteration61_order_validation/validation.json',61);read('results/iteration61_order_validation/comparisons.json',61)
    for start in (54,134):targets[start]=read(f'results/iteration37_local/target_{start:03d}.json',37)['law']['intervals']
    groups=[]
    for start in (0,54,134):
        for k in (3,4):
            g=len(groups);depth=16 if start==0 else 10;physical_modes=list(range(start,start+depth));perms=[];sources=[]
            for kind in ('identity','reverse','even_odd'):
                prefix=physical_modes if kind=='identity' else physical_modes[::-1] if kind=='reverse' else physical_modes[::2]+physical_modes[1::2];perms.append(prefix+[j for j in range(144) if j not in physical_modes])
                if start==0:
                    name=f'results/iteration56_prefix/global_{k}.stdout.json.gz' if kind=='identity' else f'results/iteration61_order/global_{kind}_k{k}.stdout.json.gz';reuse(name,56 if kind=='identity' else 61);sources.append(name)
            groups.append(dict(label=f'window_{start:03d}_k{k}',index=g,start=start,depth=depth,cutoff=k,selector_seed=620000+g,component_seeds=[620100+10*g+j for j in range(3)],permutations=perms,reused_sources=sources,new_native=start!=0))
    save(out/'plan.json',encode(dict(groups=groups,samples=17,source_hashes=hashes,dependencies=deps,transfer_upper=physical['transfer_upper'],component_kinds=['identity','reverse','even_odd'])))
    save(out/'selector.json',encode(dict(native=selector_law(),native_probes=[dict(word=w,choice=select_order(w)) for w in sorted({0,(1<<64)-1,*[b+d for b in selector_law()['boundaries'][1:-1] for d in (-1,0,1)]})],small_probes=[dict(bits=b,choices=[select_order(w,bits=b) for w in range(1<<b)],law=selector_law(bits=b)) for b in range(2,11)])))
    rows=[];component_rows=[]
    for group in groups:
        begin=time.perf_counter();label=group['label'];print('start',label,flush=True);row=dict(label=label,start=group['start'],cutoff=group['cutoff'],new_native=group['new_native'])
        try:
            rng=random.Random(group['selector_seed']);words=[rng.getrandbits(64) for _ in range(17)];component_laws=[];components=[]
            def component_callback(index):
                def call(count):
                    tick=time.perf_counter();name=f'{label}_order{index}';perm=group['permutations'][index];depth=group['depth'];storage={}
                    if group['new_native']:
                        cov=V[np.ix_(perm+[144+j for j in perm],perm+[144+j for j in perm])];csv=out/(name+'.csv');np.savetxt(csv,cov,delimiter=',',fmt='%.17g');assert np.array_equal(np.loadtxt(csv,delimiter=','),cov)
                        cmd=[str(exe),'--input',str(csv),'--modes','144','--cutoff',str(group['cutoff']),'--samples',str(count),'--seed',str(group['component_seeds'][index]),'--threads','1','--audit','--prefix-tree','--prefix-depth',str(depth)]
                        proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=1800);(out/(name+'.stderr.log')).write_bytes(proc.stderr);storage['stdout']=compressed(name+'.stdout.json.gz',proc.stdout);save(out/(name+'.storage.json'),storage);assert proc.returncode==0;raw=decode(json.loads(proc.stdout));source=str((out/(name+'.stdout.json.gz')).relative_to(ROOT).as_posix())
                    else:source=group['reused_sources'][index];raw=read(source);cmd=None
                    assert raw['modes']==144 and raw['prefix_depth']==depth and raw['prefix_reverse_verified'] and raw['cutoff']==group['cutoff'] and raw['samples']>=count;steps=replay({**raw,'prefix_tree':[]});law=full_law(depth,raw['prefix_tree'],model='uint64');sequence=law['distribution'];canonical=[F(0)]*(1<<depth)
                    for mask,mass in enumerate(sequence):
                        physical_mask=sum(((mask>>j)&1)<<(p-group['start']) for j,p in enumerate(perm[:depth]));canonical[physical_mask]=mass
                    component_laws.append(tuple(canonical));metrics=distance_interval(canonical,targets[group['start']]);rec=dict(index=index,requested_samples=count,source=source,command=cmd,replayed_steps=steps,canonical_distribution=canonical,TV_interval=metrics,law_diagnostics={key:val for key,val in law.items() if key!='distribution'})
                    storage['audit']=compressed(name+'.audit.json.gz',rec);save(out/(name+'.storage.json'),storage);components.append(rec);cr=dict(label=name,new_native=group['new_native'],requested_samples=count,available_samples=raw['samples'],TV_lower=float(metrics[0]),seconds=time.perf_counter()-tick,timings=raw['timings_seconds']);component_rows.append(cr);save(out/'component_rows.json',component_rows);print('component',json.dumps(cr),flush=True);return raw['outputs'][:count]
                return call
            dispatch=sample_order_mixture(words,[component_callback(j) for j in range(3)],group['permutations']);assert all(c>0 for c in dispatch['component_counts']) and len(components)==3
            mixture=mixture_law(component_laws);tv=distance_interval(mixture['distribution'],targets[group['start']]);ideal_tv=distance_interval(mixture['ideal_distribution'],targets[group['start']]);record=dict(group=group,dispatch=dispatch,mixture=mixture,TV_interval=tv,ideal_TV_interval=ideal_tv,component_TV_intervals=[r['TV_interval'] for r in components]);storage=compressed(label+'.audit.json.gz',record);save(out/(label+'.storage.json'),dict(audit=storage));row.update(status='completed',component_counts=dispatch['component_counts'],TV_lower=float(tv[0]),TV_upper=float(tv[1]),ideal_distance=float(mixture['ideal_distance']),baseline_TV=float(components[0]['TV_interval'][0]))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-begin;rows.append(row);save(out/'rows.json',rows);print('end',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=6,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),new_native_runs=sum(r['new_native'] for r in component_rows),new_native_samples=sum(r['requested_samples'] for r in component_rows if r['new_native']),unchanged_sources=True,unchanged_dependencies=True,dependencies=deps))
if __name__=='__main__':main()
