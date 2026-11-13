"""Independent parity-fiber, breakpoint optimality and nonextendability checks."""
import json,shutil
from pathlib import Path
from fractions import Fraction as F
from itertools import combinations
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration41_independent import determinant,target_from_vacua,parity_law,forward
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'results/iteration42_repair_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration42_repair','iteration22_external','iteration33_run','verify_iteration34','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    checks=[];deps={};rows=[];det_cache={};dual=None
    def read(name):deps[name]=digest(ROOT/name);return decode(json.loads((ROOT/name).read_text()))
    def check(label,condition):
        checks.append(dict(name=label,passed=bool(condition)))
        if not condition:save(out/'checks.failed.json',checks);raise AssertionError(label)
    def marginal(q,subset):
        return [sum((v for m,v in enumerate(q) if all(((m>>j)&1)==((k>>i)&1) for i,j in enumerate(subset))),F(0)) for k in range(1<<len(subset))]
    for folder,expected in (('iteration42_repair',9),('iteration42_fresh',15)):
        directory='results/'+folder;plan=read(directory+'/plan.json');summary=read(directory+'/summary.json');check(folder+' completion',summary['planned']==summary['completed']==expected and summary['unchanged_sources'])
        for n,sha in plan['source_hashes'].items():check(folder+' source '+n,digest(ROOT/n)==sha);check(folder+' snapshot '+n,digest(ROOT/directory/'source_snapshot'/n)==sha)
        for n,sha in plan.get('dependencies',{}).items():check(folder+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
        for sequence,job in enumerate(plan['jobs']):
            case=job if folder=='iteration42_repair' else sequence;label=folder+':'+str(case);r=read(directory+f'/case_{case:02d}.json');q=r['base'];p=r['target'];a=r['job']['intensity'];w=r['job']['weights'];check(label+' dimension',len(q)==len(p)==256 and sum(q)==sum(p)==1)
            if folder=='iteration42_repair':
                old=read(f'results/iteration41_panel/case_{case:02d}.audit.json');check(label+' published base',r['job']==old['job'] and q==old['controls']['unclamped'] and p==old['target'][:256]);matching=old['controls']['matching']
            else:
                check(label+' fixed job',r['job']==job);vac=[]
                for mask in range(256):
                    subset=tuple(sorted(w[j] for j in range(8) if (mask>>j)&1));key=(a,subset)
                    if key not in det_cache:det_cache[key]=determinant([[F(i==j)+a*x*y for j,y in enumerate(subset)] for i,x in enumerate(subset)])
                    d=det_cache[key];check(label+' determinant '+str(mask),d==1+a*sum((x*x for x in subset),F(0)) and d>0);vac.append(1/d)
                check(label+' independent physical target',target_from_vacua(vac,8)==p and min(p)>=0);p0=[vac[1<<j] for j in range(8)];c={(j,i):vac[(1<<i)|(1<<j)]-p0[j]*p0[i] for i in range(8) for j in range(i)};matching=parity_law(p0,c);check(label+' parity truncation',matching==r['controls']['matching'])
                for mask in range(256):
                    prefix=''.join(str((mask>>j)&1) for j in range(8));check(label+' forward unclamped '+str(mask),forward(prefix,p0,c,False)['mass']==q[mask]);check(label+' forward clamped '+str(mask),forward(prefix,p0,c,True)['mass']==r['controls']['clamped'][mask])
            signs=[(-1)**m.bit_count() for m in range(256)];D=r['desired_delta'];check(label+' missing term correction',all(t-x==s*D for t,x,s in zip(matching,q,signs)))
            lo=max(-q[m] for m in range(256) if signs[m]==1);hi=min(q[m] for m in range(256) if signs[m]==-1);interval=r['correction']['interval'];e,o=interval['even_witness'],interval['odd_witness'];check(label+' exact feasible interval',interval['lower']==lo and interval['upper']==hi and interval['feasible']==(lo<=hi) and signs[e]==1 and signs[o]==-1 and -q[e]==lo and q[o]==hi and interval['pair_mass']==q[e]+q[o])
            first=None;mins=[]
            for size in range(1,8):
                for subset in combinations(range(8),size):
                    m=marginal(q,subset);mins.append(dict(subset=subset,minimum=min(m)))
                    if first is None:
                        for atom,value in enumerate(m):
                            if value<0:first=dict(subset=subset,atom=atom,mass=value,order=size);break
                    check(label+' normalized proper marginal '+str(subset),sum(m)==1)
            check(label+' lowest negative marginal',first==r['first_negative_proper_marginal'])
            if lo>hi:
                check(label+' infeasible certificate',q[e]+q[o]<0 and r['correction']['law'] is None and r['oracle']['law'] is None and r['oracle']['TV'] is None)
            else:
                delta=min(hi,max(lo,D));corrected=[x+s*delta for x,s in zip(q,signs)];check(label+' feasible correction',r['correction']['delta']==delta and list(r['correction']['law'])==corrected and min(corrected)>=0 and sum(corrected)==1)
                for omitted in range(8):subset=tuple(j for j in range(8) if j!=omitted);check(label+' fixed maximal proper marginal '+str(omitted),marginal(corrected,subset)==marginal(q,subset))
                objective=lambda d:sum((abs(x+s*d-y) for x,s,y in zip(q,signs,p)),F(0))/2
                candidates={lo,hi,*[s*(y-x) for x,s,y in zip(q,signs,p) if lo<=s*(y-x)<=hi]};best=min(objective(d) for d in candidates);oracle=r['oracle'];check(label+' exhaustive breakpoint optimum',lo<=oracle['delta']<=hi and objective(oracle['delta'])==oracle['TV']==best)
                check(label+' oracle table',list(oracle['law'])==[x+s*oracle['delta'] for x,s in zip(q,signs)] and min(oracle['law'])>=0);check(label+' corrected TV',r['correction_TV']==objective(delta))
            if folder=='iteration42_fresh' and case==1:
                check('all proper marginals nonnegative but no extension',first is None and all(t['minimum']>=0 for t in mins) and lo>hi)
                coefficients=[];expectation=F(0)
                for subset_mask in range(256):
                    coefficient=F((-1)**(subset_mask&e).bit_count()+(-1)**(subset_mask&o).bit_count(),256)
                    moment=sum((x*(-1)**(subset_mask&m).bit_count() for m,x in enumerate(q)),F(0));expectation+=coefficient*moment
                    if coefficient:coefficients.append(dict(subset_mask=subset_mask,coefficient=coefficient,moment=moment))
                check('dual contains no full parity',all(t['subset_mask']!=255 for t in coefficients));check('dual negative expectation',expectation==q[e]+q[o]==F(-55898916,6684099653));dual=dict(even_atom=e,odd_atom=o,negative_expectation=expectation,Walsh_terms=coefficients,proper_marginal_minima=mins,scope='Nonnegative indicator of two atoms depends only on fixed proper marginals, but their specified expectation is negative')
            rows.append(dict(folder=folder,case=case,feasible=lo<=hi,negative_order=None if first is None else first['order']));print('verified repair',label,flush=True)
    check('verifier unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));save(out/'nonextendability.json',encode(dual));save(out/'rows.json',rows);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=len(rows),unique_exact_determinants=len(det_cache),source_hashes=hashes,dependencies=deps));print('repair verified',len(checks),flush=True)
if __name__=='__main__':main()
