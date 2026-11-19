"""Independent complete native output laws on exactly shared physical inputs."""
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
from iteration41_independent import layer_law
from iteration45_independent import tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration47_native_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration47_native','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent','iteration41_independent','iteration45_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[];steps=0
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration47_native';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('24 native configurations',summary['planned']==summary['finished']==summary['completed']==24);validation=read('results/iteration47_input_validation/validation.json');check('shared targets validated',validation['count']==validation['passed'])
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for n,s in plan['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    for job in plan['jobs']:
        index=job['case'];cutoff=job['cutoff'];tag=f'case{index}K{cutoff} ';name=f'results/iteration47_inputs/case_{index:02d}.json';source=read(name);check(tag+'bound physical target',deps[name]==validation['dependencies'][name]);n=source['job']['modes'];V=source['physical']['covariance'];physical=source['target']['target_intervals'];label=f'case_{index:02d}_k{cutoff}';raw=read(base+'/'+label+'.stdout.json');reverse=read(base+'/'+label+'.reverse.json');audit=read(base+'/'+label+'.audit.json');check(tag+'identity',audit['job']==job and raw['modes']==10 and raw['cutoff']==cutoff and raw['seed']==472000+index and raw['samples']==3);table={x['prefix']:x['threshold_hex'] for x in raw['prefix_tree']};by_prefix={x['prefix']:x for x in raw['prefix_tree']};check(tag+'complete/reverse',len(table)==len(raw['prefix_tree'])==1023 and by_prefix=={x['prefix']:x for x in reverse['prefix_tree']} and all(''.join(str((m>>j)&1) for j in range(d)) in table for d in range(10) for m in range(1<<d)));steps+=replay(raw);csvname=base+'/'+label+'.csv';deps[csvname]=digest(ROOT/csvname);stored=np.loadtxt(ROOT/csvname,delimiter=',');ix=list(range(n))+list(range(10,10+n));check(tag+'exact same covariance',all(F(float(stored[i,j]))==(V[ix.index(i)][ix.index(j)] if i in ix and j in ix else F(i==j)) for i in range(20) for j in range(20)))
        for prefix,x in by_prefix.items():
            q,prev,z=(float.fromhex(x[k]) for k in ('threshold_hex','previous_hex','proposed_hex'));check(tag+'quotient '+prefix,(math.isnan(q) if z==0 else math.isinf(q) and math.copysign(1,q)==math.copysign(1,z)*math.copysign(1,prev)) if prev==0 else q==float(F(z)/F(prev)))
            if prefix:
                par=by_prefix[prefix[:-1]];pp=float.fromhex(par['previous_hex']);pz=float.fromhex(par['proposed_hex']);check(tag+'continuity '+prefix,prev==(pz if prefix[-1]=='0' else float(F(pp)-F(pz))))
        padded=physical+[(F(0),F(0))]*(1024-(1<<n));row=dict(case=index,cutoff=cutoff)
        for model in ('continuous','uint64'):
            law=layer_law(table,10,model);check(tag+'independent law '+model,law==audit['laws'][model]['distribution'] and min(law)>=0 and sum(law)==1);value=tv(law,padded);check(tag+'full TV '+model,value==tuple(audit['physical_TV_intervals'][model]) and sum(law[1<<n:])==audit['vacuum_padding_leak'][model]);row[model+'_TV']=value
        rows.append(row);print('verified native',index,cutoff,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=24,replayed_steps=steps,source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
