"""Independent complete-law verification with exact physical target determinants."""
import json,shutil,sys
from pathlib import Path
from fractions import Fraction as F
from iteration41_independent import forward,parity_law,determinant,target_from_vacua,layer_law
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration39_independent import replay
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration41_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration41','iteration41_independent','iteration22_external','iteration33_run','verify_iteration34','iteration39_independent')];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    checks=[];deps={};read=lambda p:decode(json.loads(p.read_text()));dets={};rows=[];steps=0;full_atoms=0
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:save(out/'checks.failed.json',checks);raise AssertionError(name)
    for directory in ('iteration41_pilot','iteration41_panel'):
        folder=ROOT/'results'/directory;plan=read(folder/'plan.json');summary=read(folder/'summary.json');check(directory+' completion',summary['completed']==summary['planned'] and summary['unchanged_sources'])
        for n,sha in plan['source_hashes'].items():check(directory+' source '+n,digest(ROOT/n)==sha);check(directory+' snapshot '+n,digest(folder/'source_snapshot'/n)==sha)
        for n,sha in plan.get('dependencies',{}).items():check(directory+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
    panel=ROOT/'results/iteration41_panel';plan=read(panel/'plan.json');check('44 fixed jobs',len(plan['jobs'])==44)
    for sequence,job in enumerate(plan['jobs']):
        record=read(panel/f'case_{sequence:02d}.audit.json');raw=read(panel/f'case_{sequence:02d}.stdout.json');reverse=read(panel/f'case_{sequence:02d}.reverse.json');n=job['active'];w=job['weights'];a=job['intensity'];label=str(sequence)+':'
        check(label+'job identity',record['job']==job and raw['modes']==10 and raw['cutoff']==job['cutoff'] and raw['samples']==65 and raw['seed']==411001+sequence)
        matrix=[[F(float(x)) for x in line.split(',')] for line in (panel/f'case_{sequence:02d}.csv').read_text().splitlines()];weights=w+[F(0)]*(10-n)
        check(label+'exact physical input',matrix==[[F(i==j)+(2*a*weights[i%10]*weights[j%10] if i//10==j//10 else 0) for j in range(20)] for i in range(20)] and a>0)
        vac=[]
        for mask in range(1<<n):
            subset=tuple(sorted(w[j] for j in range(n) if (mask>>j)&1));key=(a,subset)
            if key not in dets:
                block=[[F(i==j)+a*x*y for j,y in enumerate(subset)] for i,x in enumerate(subset)];dets[key]=determinant(block)
            d=dets[key];check(label+'vacuum determinant '+str(mask),d==1+a*sum((x*x for x in subset),F(0)) and d>0);vac.append(1/d)
        target=target_from_vacua(vac,n);check(label+'target positive normalized',min(target)>=0 and sum(target)==1);check(label+'independent target',record['target']==target+[F(0)]*((1<<10)-len(target)))
        table={r['prefix']:r['threshold_hex'] for r in raw['prefix_tree']};check(label+'complete threshold tree',len(table)==1023 and all(''.join(str((m>>j)&1) for j in range(k)) in table for k in range(10) for m in range(1<<k)))
        check(label+'reverse tree',{r['prefix']:r for r in raw['prefix_tree']}=={r['prefix']:r for r in reverse['prefix_tree']});steps+=replay(raw)
        for model in ('continuous','uint64'):
            law=layer_law(table,10,model);check(label+model+' law',law==record['native'][model] and sum(law)==1 and min(law)>=0);full_atoms+=len(law)
            tv=sum((abs(x-y) for x,y in zip(law,record['target'])),F(0))/2;check(label+model+' TV',tv==record['TV'][model]);check(label+model+' vacuum padding retained',sum(law[1<<n:])==record['vacuum_padding_leak'][model])
        if job['cutoff']==2:
            p=[vac[1<<i] for i in range(n)];c={(j,i):vac[(1<<j)|(1<<i)]-p[j]*p[i] for i in range(n) for j in range(i)};control=record['controls'];walsh=parity_law(p,c);check(label+'parity truncation',walsh==control['matching']);clip_count=0
            exported=read(panel/f'case_{sequence:02d}.coefficients.json');np=[(1+F(float.fromhex(x)))/2 for x in exported['first_hex'][:n]];nc={(j,i):F(float.fromhex(x))/4 for j,i,x in exported['pairs'] if i<n}
            for mask in range(1<<n):
                prefix=''.join(str((mask>>j)&1) for j in range(n));unclamped=forward(prefix,p,c,False);clamped=forward(prefix,p,c,True);native=forward(prefix,np,nc,True);clip_count+=clamped['clipped']
                check(label+'unclamped '+prefix,unclamped['mass']==control['unclamped'][mask]);check(label+'clamped '+prefix,clamped['mass']==control['clamped'][mask]);check(label+'binary coefficients '+prefix,native['mass']==record['native_coefficient_reference_signed'][mask])
                for j,(previous,zero) in enumerate(clamped['records']):
                    row=control['tree'][prefix[:j]];check(label+'forward branch '+prefix+':'+str(j),previous==row['previous'] and zero==row['zero'] and row['threshold']==(None if previous==0 else zero/previous))
            probabilities={'':F(1)};saturated=[]
            for length in range(n):
                for prefix,mass in list(probabilities.items()):
                    if len(prefix)!=length or mass==0:continue
                    row=control['tree'][prefix];check(label+'reachable denominator '+prefix,row['previous']>0);q=row['threshold']
                    if not 0<=q<=1:saturated.append(dict(prefix=prefix,threshold=q,output_prefix_probability=mass))
                    q=max(F(0),min(F(1),q));probabilities[prefix+'0']=mass*q;probabilities[prefix+'1']=mass*(1-q)
            real=[probabilities.get(''.join(str((mask>>j)&1) for j in range(n)),F(0)) for mask in range(1<<n)];check(label+'real sampler',real==control['sampler'] and sum(real)==1);check(label+'saturation',sorted(saturated,key=lambda x:x['prefix'])==sorted(control['saturated'],key=lambda x:x['prefix']))
            for k in ('matching','unclamped','clamped'):check(label+'negative '+k,sum((-x for x in control[k] if x<0),F(0))==record['negative_mass'][k] and sum(control[k])==1)
            check(label+'ablation difference',sum((abs(x-y) for x,y in zip(walsh,control['unclamped'])),F(0))/2==record['approximation_half_L1']);check(label+'real TV',sum((abs(x-y) for x,y in zip(real,target)),F(0))/2==record['real_sampler_TV']);check(label+'native-real TV',sum((abs(x-y) for x,y in zip(record['native']['uint64'],real+[F(0)]*(1024-len(real)))),F(0))/2==record['native_to_physical_real_sampler_TV'])
            rows.append(dict(sequence=sequence,clamp_activations_over_all_terminal_evaluations=clip_count))
        print('verified case',sequence,flush=True)
    for sequence in range(15):
        pilot=read(ROOT/f'results/iteration41_pilot/case_{sequence:02d}.json');n=pilot['job']['modes'];a=pilot['job']['intensity'];index=next(i for i,j in enumerate(plan['jobs']) if j['active']==n and j['intensity']==a and j['family']=='equal' and j['cutoff']==2);main_record=read(panel/f'case_{index:02d}.audit.json');check('pilot controls '+str(sequence),pilot['laws']==main_record['controls'] and pilot['target']==main_record['target'][:1<<n])
    check('verifier unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));save(out/'determinants.json',encode([dict(intensity=a,weights=w,determinant=d) for (a,w),d in dets.items()]));save(out/'rows.json',rows);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=44,replayed_steps=steps,native_distribution_atoms_checked=full_atoms,unique_exact_determinants=len(dets),source_hashes=hashes,dependencies=deps));print('passed',len(checks),flush=True)
if __name__=='__main__':main()
