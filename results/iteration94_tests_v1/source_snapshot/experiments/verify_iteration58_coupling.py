"""Independent raw-moment, covariance and coupling certificate verification."""
import sys,json,shutil,itertools,time
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration45_independent import inverse
from iteration48_independent import bisect_sqrt
from iteration51_normal_independent import cdf,exp_negative,pi_bounds
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def add(*args):return sum(a for a,b in args),sum(b for a,b in args)
def scale(c,p):return (c*p[0],c*p[1]) if c>=0 else (c*p[1],c*p[0])
def outward(p):
    D=1<<256;return F((p[0]*D).__floor__(),D),F((p[1]*D).__ceil__(),D)
@lru_cache(None)
def C(x):return outward(cdf(x))
@lru_cache(None)
def D(x):
    el,eh=exp_negative(x*x/2);pl,ph=pi_bounds();return outward((el*bisect_sqrt(1/(2*ph))[0],eh*bisect_sqrt(1/(2*pl))[1]))
def raw(a,b):return add(C(b),scale(-1,C(a))),add(D(a),scale(-1,D(b))),add(C(b),scale(-1,C(a)),scale(a,D(a)),scale(-b,D(b)))
def matmul(a,b):return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]
def transpose(a):return list(map(list,zip(*a)))
def cov(br,bi):
    n=len(br);S=[br[i]+bi[i] for i in range(n)]+[bi[i]+[-x for x in br[i]] for i in range(n)];T=[[F(i==j)-S[i][j] for j in range(2*n)] for i in range(2*n)];inv=inverse(T);return [[2*inv[i][j]-F(i==j) for j in range(2*n)] for i in range(2*n)]

def main():
    out=ROOT/'results/iteration58_coupling_validation';out.mkdir(exist_ok=False);start=time.perf_counter();deps={};checks=[]
    names=[f'experiments/{n}.py' for n in ('verify_iteration58_coupling','iteration22_external','iteration33_run','verify_iteration34','iteration45_independent','iteration48_independent','iteration51_normal_independent')];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:save(out/'checks.json',checks);raise AssertionError(name)
    def enclosed(name,narrow,wide):check(name,wide[0]<=narrow[0]<=narrow[1]<=wide[1])
    prefix='results/iteration58_coupling/';summary=read(prefix+'summary.json');check('producer complete',summary['completed'])
    for n,s in summary['source_hashes'].items():check('frozen producer '+n,digest(ROOT/n)==s==digest(ROOT/prefix/'source_snapshot'/n))
    oldmanifest=read('results/iteration55_manifest.json');oldvalidation=read('results/iteration55_noise_validation/validation.json');check('old independent validation',oldvalidation['passed']==oldvalidation['count'] and deps['results/iteration55_noise_validation/validation.json']==oldmanifest['artifacts']['results/iteration55_noise_validation/validation.json'])
    index=read(prefix+'axis_index.json');axes={};independent=[]
    for ent in index:
        a=read(prefix+f"axis_{ent['index']:02d}.json");sigma=a['sigma'];N=a['cells'];R=a['radius'];tag=f"axis{ent['index']}";check(tag+' specification',(sigma,N,R)==(ent['sigma'],ent['cells'],ent['radius']));check(tag+' declared symmetric grid',N in (4,16) and R=={4:2,16:3}[N] and a['mean_residual']==0 and len(a['rows'])==N)
        r1=[];r2=[]
        for j,r in enumerate(a['rows']):
            lo=-R+2*R*F(j,N);hi=-R+2*R*F(j+1,N);c=(lo+hi)/2;check(tag+f' bin{j} coordinates',(r['a'],r['b'],r['center'])==(lo,hi,c));m0,m1,m2=raw(lo,hi);l0,l1,_=raw(lo,c);h0,h1,_=raw(c,hi);v1=scale(sigma,add(scale(c,l0),scale(-1,l1),h1,scale(-c,h0)));v2=scale(sigma*sigma,add(m2,scale(-2*c,m1),scale(c*c,m0)))
            for label,ind,prod in [('mass',m0,r['mass_interval']),('absolute',v1,r['absolute_interval']),('squared',v2,r['squared_interval'])]:enclosed(tag+f' bin{j} '+label,ind,prod)
            other=a['rows'][N-1-j];check(tag+f' bin{j} odd quantizer',r['a']==-other['b'] and r['center']==-other['center']);r1.append(v1);r2.append(v2)
        t0=add((F(1),F(1)),scale(-1,C(R)));t1=D(R);t2=add(t0,scale(R,D(R)));tail1=scale(2*sigma,add(t1,scale(-R,t0)));tail2=scale(2*sigma*sigma,add(t2,scale(-2*R,t1),scale(R*R,t0)))
        for label,ind in [('tail_absolute',tail1),('tail_squared',tail2),('absolute',add(tail1,*r1)),('squared',add(tail2,*r2))]:enclosed(tag+' '+label,ind,a[label+'_interval'])
        check(tag+' production moment sums',tuple(a['absolute_interval'])==add(a['tail_absolute_interval'],*(r['absolute_interval'] for r in a['rows'])) and tuple(a['squared_interval'])==add(a['tail_squared_interval'],*(r['squared_interval'] for r in a['rows'])))
        axes[(sigma,N,R)]=a;independent.append(dict(index=ent['index'],absolute_interval=add(tail1,*r1),squared_interval=add(tail2,*r2)))
    def physical(tag,br,bi,directions,expected):
        V=cov(br,bi);check(tag+' covariance from inverse',V==expected['physical']['covariance']);n=len(br);J=[[F(j==i+n)-F(i==j+n) for j in range(2*n)] for i in range(2*n)];check(tag+' purity VJV=J',matmul(matmul(V,J),V)==J)
        for k in range(1,2*n+1):check(tag+f' SPD leading{k}',exact_determinant([row[:k] for row in V[:k]])>0)
        w=[[F(x) for x,y in v]+[F(y) for x,y in v] for v in directions];u=[q[n:]+[-x for x in q[:n]] for q in w];G=matmul(matmul(u,V),transpose(u));check(tag+' generator Gram',G==expected['generator_gram']);check(tag+' overlap inverse identity',G==matmul(matmul(w,inverse(V)),transpose(w)))
        for k in range(1,len(G)+1):
            for ids in itertools.combinations(range(len(G)),k):check(tag+' Gram minor'+str(ids),exact_determinant([[G[i][j] for j in ids] for i in ids])>=0)
        return G,V,w
    def bound(tag,b,G):
        check(tag+' Gram bound',b['gram']==G);mm=[axes[(a['sigma'],a['cells'],a['radius'])] for a in b['axes']];mean=add(*(scale(G[i][i],m['squared_interval']) for i,m in enumerate(mm)));check(tag+' independent zero cross moments',tuple(b['mean_quadratic_interval'])==mean)
        old=sum(bisect_sqrt(G[i][i],192)[1]*((1-m['grid']['tail_interval'][0])*m['grid']['interior_half_width']+m['grid']['tail_mean_excess_interval'][1]) for i,m in enumerate(mm));l1=sum(bisect_sqrt(G[i][i],192)[1]*m['absolute_interval'][1] for i,m in enumerate(mm));rms=bisect_sqrt(mean[1],192)[1];exp=b['exponent_at_mean_upper_interval'];check(tag+' supported exponent',0<=mean[1]<=64);enclosed(tag+' independent positive exponential',outward(exp_negative(mean[1])),exp);curve=bisect_sqrt(max(F(0),1-exp[0]),192)[1]
        expected=dict(old_axis_upper=min(F(1),old),exact_l1_upper=min(F(1),l1),rms_upper=min(F(1),rms),joint_curved_upper=min(F(1),curve))
        for key,v in expected.items():check(tag+' '+key,b[key]==v)
        check(tag+' minimum and strict improvement',b['best_upper']==min(expected.values())<b['old_axis_upper']);check(tag+' curvature improves RMS',b['joint_curved_upper']<b['rms_upper'])
    rows=read(prefix+'retrospective_rows.json');check('retrospective all24',len(rows)==24)
    for i,r in enumerate(rows):
        name=f'results/iteration55_noise/case_{i:02d}.json';old=read(name);tag=f'old{i}';check(tag+' old certified exact bytes',deps[name]==oldmanifest['artifacts'][name]==oldvalidation['dependencies'][name]);check(tag+' case job and axes',r['case']==i and r['job']==old['job'] and r['bound']['axes']==[dict(sigma=g['sigma'],cells=g['cells'],radius=g['standard_radius']) for g in old['grids']]);G,V,w=physical(tag,old['pairing_real'],old['pairing_imag'],old['job']['directions'],old['target']);bound(tag,r['bound'],G)
        for a,g in zip(r['bound']['axes'],old['grids']):check(tag+' unchanged axis'+str(a),axes[(a['sigma'],a['cells'],a['radius'])]['grid']==g)
        other=old['bridge']['full_trace_interval'][1]+old['selector_TV'][1]+old['weighted_arithmetic_TV_upper'];check(tag+' unchanged other errors and law',other==r['other_errors_upper'] and r['actual_TV']==old['full_Gaussian_TV'] and r['ideal_grid_TV']==old['ideal_grid_TV']);check(tag+' complete bound',r['old_total_upper']==old['total_TV_upper'] and r['new_total_upper']==min(F(1),other+r['bound']['best_upper']) and r['actual_TV'][1]<=r['new_total_upper']<=r['old_total_upper'] and r['ideal_grid_TV'][1]<=r['bound']['best_upper'])
    models=read(prefix+'rank4_models.json');rank4=read(prefix+'rank4_rows.json');check('rank4 counts',len(models)==2 and len(rank4)==32);directions=[[(1,0),(0,0)],[(0,1),(0,0)],[(0,0),(1,0)],[(0,0),(0,1)]];expected_specs=[('real_pair',F(1,3),(1,0),False,[F(1,8),F(3,16),F(1,4),F(1,8)]),('complex_rotated',F(2,5),(0,1),True,[F(3,16),F(1,8),F(1,4),F(3,16)])]
    for m,spec in zip(models,expected_specs):
        fam,lam,phase,rot,sigmas=spec;tag=fam;check(tag+' preregistered input',(m['family'],m['pairing'],tuple(m['phase']),m['rotated'],m['sigmas'])==spec and m['directions']==[[list(z) for z in v] for v in directions]);shape=[[F(24,25),-F(7,25)],[-F(7,25),-F(24,25)]] if rot else [[F(0),F(1)],[F(1),F(0)]];br=[[lam*phase[0]*x for x in row] for row in shape];bi=[[lam*phase[1]*x for x in row] for row in shape];check(tag+' pairing',br==m['pairing_real'] and bi==m['pairing_imag']);G,V,w=physical(tag,br,bi,directions,m);noise=[[4*sum(sigmas[k]**2*w[k][i]*w[k][j] for k in range(4)) for j in range(4)] for i in range(4)];check(tag+' rank4 positive determinant',exact_determinant(noise)>0 and exact_determinant(G)>0);check(tag+' full covariance',noise==m['noise_covariance'] and m['mixed_covariance']==[[V[i][j]+noise[i][j] for j in range(4)] for i in range(4)])
        rr=[r for r in rank4 if r['family']==fam];check(tag+' configurations',len(rr)==16)
        for r,(idx,ns) in zip(rr,enumerate(itertools.product((4,16),repeat=4))):
            check(tag+f'{idx} fixed configuration',r['index']==idx and tuple(r['cells'])==ns and r['bound']['axes']==[dict(sigma=s,cells=N,radius={4:2,16:3}[N]) for s,N in zip(sigmas,ns)]);bound(tag+str(idx),r['bound'],G)
    save(out/'independent_moments.json',encode(independent));save(out/'checks.json',checks);check('unchanged independent sources',all(digest(ROOT/n)==s for n,s in hashes.items()));check('unchanged dependencies',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),axes=8,retrospective_cases=24,rank4_cases=32,source_hashes=hashes,dependencies=deps,seconds=time.perf_counter()-start,scope='Fresh independent moment/covariance/coupling checks; old complete program laws reused through prior manifest and validation. No rank4 sampler or fresh two-drive holdout.'));print('independent coupling checks',len(checks),flush=True)

if __name__=='__main__':main()
