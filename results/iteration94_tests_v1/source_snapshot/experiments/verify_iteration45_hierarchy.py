"""Independent generalized moment, positivity and maximum-entropy bounds."""
import sys,json,shutil,itertools,math
from pathlib import Path
from fractions import Fraction as F
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from verify_iteration43_completion_recovery import direct_log
from iteration45_independent import mobius,tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration45_hierarchy_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration45_hierarchy','iteration45_independent','iteration22_external','iteration33_run','verify_iteration34','verify_iteration43_completion_recovery')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[];source_cache={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration45_hierarchy';plan=read(base+'/plan.json');summary=read(base+'/summary.json');check('all30complete',summary['planned']==summary['finished']==summary['completed']==30)
    for n,s in plan['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for n,s in plan['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    for index,job in enumerate(plan['jobs']):
        tag=f'case{index} ';rec=read(base+f'/case_{index:02d}.json');n=job['modes'];order=job['order'];size=1<<n;check(tag+'job',job==rec['job']);source=job['source'];path=source['path']
        if path not in source_cache:
            old=read(path);manifest=read(f"results/iteration{source['round']}_manifest.json");check(tag+'published input',deps[path]==manifest['artifacts'][path])
            if source['round']==41:
                a=old['job']['intensity'];w=old['job']['weights'];vac=[(1/(1+a*sum(w[j]**2 for j in range(n) if m>>j&1)),)*2 for m in range(size)];_,target=mobius(vac,n);check(tag+'independent classical target',[p[0] for p in target]==old['target'][:size])
            else:
                V=old['physical']['covariance'];vac=old['target']['vacuum_intervals']
                for m in range(1,size):
                    ids=[j for j in range(n) if m>>j&1];ix=ids+[j+n for j in ids];det=exact_determinant([[(V[a][b]+int(a==b))/2 for b in ix] for a in ix]);lo,hi=vac[m];check(tag+'independent physical vacuum '+str(m),det>0 and 0<=lo<=hi<=1 and lo*lo*det<=1<=hi*hi*det)
                _,target=mobius(vac,n);check(tag+'published physical target',encode(target)==encode(old['target']['target_intervals']))
            source_cache[path]=(vac,target)
        vac,target=source_cache[path];check(tag+'target unchanged',encode(vac)==encode(rec['vacua']) and encode(target)==encode(rec['target']));subsets=[subset for degree in range(1,order+1) for subset in sorted(itertools.combinations(range(n),degree),key=lambda s:sum(1<<j for j in s))];masks=[sum(1<<j for j in s) for s in subsets];X=[[math.prod(1-2*((a>>j)&1) for j in subset) for subset in subsets] for a in range(size)];mi=[]
        for subset in subsets:
            lo=hi=F(0)
            for degree in range(len(subset)+1):
                for t in itertools.combinations(subset,degree):
                    a,b=vac[sum(1<<j for j in t)];c=2**degree*(-1)**(len(subset)-degree);lo+=c*(a if c>=0 else b);hi+=c*(b if c>=0 else a)
            mi.append((lo,hi))
        midpoint=[(a+b)/2 for a,b in mi];check(tag+'independent moments',encode(mi)==encode(rec['moment_intervals']) and midpoint==rec['moment_midpoints']);projection=rec['projection'];R=projection['law'];check(tag+'positive normalized exact feature set',len(R)==size and min(R)>0 and sum(R)==1 and masks==projection['features'])
        for j,mu in enumerate(midpoint):check(tag+'moment '+str(j),sum(R[a]*X[a][j] for a in range(size))==mu)
        q=rec['fit']['probabilities'];q=[p/sum(q) for p in q];delta=[mu-sum(q[a]*X[a][j] for a in range(size)) for j,mu in enumerate(midpoint)];check(tag+'orthogonal projection',q==projection['normalized_candidate'] and delta==projection['corrections'] and all(R[a]==q[a]+sum(d*x for d,x in zip(delta,X[a]))/size for a in range(size)))
        b=rec['bridge'];radii=[(hi-lo)/2 for lo,hi in mi];rho=sum(radii)/size;dtv=sum(radii)/2;atoms=[(p-rho,p+rho) for p in R];theta=rec['fit']['theta'];check(tag+'positive true moment bridge',b['features']==masks and b['rational_moments']==midpoint and encode(mi)==encode(b['moment_intervals']) and b['moment_radii']==radii and b['atom_radius']==rho and b['TV_to_true_feasible_upper']==dtv and b['theta']==theta and encode(atoms)==encode(b['feasible_atom_intervals']) and min(R)>rho);residual=[]
        for a,(lo,hi) in enumerate(atoms):
            ll,lh=direct_log(lo);hl,hh=direct_log(hi);l,h=b['log_intervals'][a];check(tag+'independent log '+str(a),l<=ll<=lh<=h and l<=hl<=hh<=h);energy=sum(t*x for t,x in zip(theta,X[a]));residual.append((l-energy,h-energy))
        floor=min(l for l,h in residual);gap=sum(hi*(h-floor) for (lo,hi),(l,h) in zip(atoms,residual));eps=b['TV_to_true_maxent_upper'];root=eps-dtv;check(tag+'entropy and Pinsker',gap==b['entropy_gap_upper_for_true_feasible']>=0 and floor==b['residual_floor'] and 0<=root and root**2>=gap/2 and (root-F(1,1<<80))**2<gap/2);physical=tv(R,target);true=(max(F(0),physical[0]-eps),min(F(1),physical[1]+eps));check(tag+'separate target TV',list(physical)==rec['physical_TV_interval'] and list(true)==rec['true_maxent_physical_TV_interval']);sharp=true
        if order==n:
            basis=[[1]+row for row in X];check(tag+'complete Walsh basis',len(masks)==size-1 and all(sum(basis[a][i]*basis[a][j] for a in range(size))==size*int(i==j) for i in range(size) for j in range(size)));check(tag+'unique midpoint law',all(R[a]==(1+sum(mu*x for mu,x in zip(midpoint,X[a])))/size for a in range(size)));sharp=(F(0),F(0))
        rows.append(dict(case=index,input=job['input'],source=source,modes=n,order=order,features=len(masks),physical_TV=physical,true_maxent_TV=sharp,generic_true_maxent_TV=true,distance_to_true_maxent_upper=eps,full_information=order==n));print('verified hierarchy',index,flush=True)
    comparisons=[]
    for input_index in range(12):
        pair={r['order']:r for r in rows if r['input']==input_index};a,b=pair[2],pair[3];better=b['true_maxent_TV'][1]<a['true_maxent_TV'][0];check('input'+str(input_index)+'K3 improves certified true maxent',better);comparisons.append(dict(input=input_index,degree2=a['true_maxent_TV'],degree3=b['true_maxent_TV'],certified_improvement_lower=a['true_maxent_TV'][0]-b['true_maxent_TV'][1],relative_midpoint_improvement=1-float(b['physical_TV'][0]/a['physical_TV'][0])))
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'comparisons.json',encode(comparisons));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),cases=30,certified_degree3_improvements=12,full_information_controls=6,source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
