"""Independent recursive interval proof, including exact physical cumulants."""
import sys,json,shutil,math,time
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
from itertools import permutations
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0);sys.setrecursionlimit(10000)
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode


def recursive(first,pairs,prefix,bits=512):
    scale=1<<bits;one=(scale,scale);zero=(0,0);sign=[1 if x=='0' else -1 for x in prefix];records={};operations=0
    def interval(bounds):
        a,b=(F(x)*scale for x in bounds);return a.numerator//a.denominator,-((-b.numerator)//b.denominator)
    def plus(a,b):return a[0]+b[0],a[1]+b[1]
    def minus(a):return -a[1],-a[0]
    def times(a,b):
        nonlocal operations
        operations+=1;v=[x*y for x in a for y in b];return min(v)//scale,-((-max(v))//scale)
    def clip(a):return max(0,min(scale,a[0])),max(0,min(scale,a[1]))
    p0=[interval(((1+a)/2,(1+b)/2)) for a,b in first];ps=[interval(((1+a)/2,(1+b)/2) if sign[i]==1 else ((1-b)/2,(1-a)/2)) for i,(a,b) in enumerate(first)];K={(j,i):interval((a/4,b/4) if sign[j]==1 else (-b/4,-a/4)) for (j,i),(a,b) in pairs.items()}
    def combine(previous,total,last):return clip(plus(times(previous,ps[last]),total if sign[last]==1 else minus(total)))
    @lru_cache(None)
    def main(end):
        if end==0:return one
        last=end-1;previous=main(last);total=zero
        for j in range(last):total=plus(total,times(K[j,last],elided(j,last)))
        z=clip(plus(times(previous,p0[last]),total));assert previous[0]>0;quotients=[F(x,y) for x in z for y in previous];records[last]=dict(mode=last,previous=(F(previous[0],scale),F(previous[1],scale)),zero=(F(z[0],scale),F(z[1],scale)),threshold=(min(quotients),max(quotients)));following=z if sign[last]==1 else plus(previous,minus(z));assert following[0]>0;return following
    @lru_cache(None)
    def partial(begin,end):
        if begin==end:return one
        if begin==0:return main(end)
        last=end-1;total=zero
        for j in range(begin,last):total=plus(total,times(K[j,last],partial_elided(begin,j,last)))
        return combine(partial(begin,last),total,last)
    @lru_cache(None)
    def elided(e,end):
        if e==0:return partial(1,end)
        if e==end-1:return main(end-1)
        last=end-1;total=zero
        for j in range(last):
            if j!=e:total=plus(total,times(K[j,last],double_elided(min(j,e),max(j,e),last)))
        return combine(elided(e,last),total,last)
    @lru_cache(None)
    def partial_elided(begin,e,end):
        if begin==e:return partial(begin+1,end)
        if begin==0:return elided(e,end)
        if e==end-1:return partial(begin,end-1)
        last=end-1;total=zero
        for j in range(begin,last):
            if j!=e:
                a,b=min(j,e),max(j,e);term=times(partial_elided(begin,a,b),partial(b+1,last));total=plus(total,times(K[j,last],term))
        return combine(partial_elided(begin,e,last),total,last)
    @lru_cache(None)
    def double_elided(a,b,end):
        if a==0 and b==1:return partial(2,end)
        if a==0 and b==end-1:return partial(1,end-1)
        if a==end-2 and b==end-1:return main(end-2)
        if b==end-1:return elided(a,end-1)
        last=end-1;total=zero
        for j in range(last):
            if j!=a and j!=b:
                x,y,z=sorted((a,b,j));term=times(double_elided(x,y,z),partial(z+1,last));total=plus(total,times(K[j,last],term))
        return combine(double_elided(a,b,last),total,last)
    main(len(prefix));return dict(bits=bits,records=[records[i] for i in range(len(prefix))],multiplications=operations,cache_entries=sum(f.cache_info().currsize for f in (main,partial,elided,partial_elided,double_elided)))


def main():
    out=ROOT/'results/iteration40_saturation_validation';out.mkdir(exist_ok=False);names=['experiments/verify_iteration40_saturation.py','experiments/iteration22_external.py','experiments/iteration33_run.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:shutil.copy2(ROOT/n,out/Path(n).name)
    read=lambda p:decode(json.loads(Path(p).read_text()));checks=[];deps={}
    def check(name,ok):
        checks.append(dict(name=name,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(name)
    for label,filename in (('saturation','summary.json'),('physical_coefficients','summary.json'),('target_build','build.json')):
        folder=ROOT/f'results/iteration40_{label}';record=read(folder/filename)
        for n,sha in record['source_hashes'].items():check(label+' source '+n,digest(ROOT/n)==sha);check(label+' snapshot '+n,digest(folder/'source_snapshot'/n)==sha)
        for n,sha in record.get('dependencies',{}).items():check(label+' dependency '+n,digest(ROOT/n)==sha);deps[n]=sha
    physical=ROOT/'results/iteration40_physical_coefficients';s=read(physical/'summary.json');V=np.load(ROOT/'results/iteration35_panel/inputs.npz')['shifted'];vac={};delta=F(1,1<<256)
    for shard in range(s['shards']):
        path=physical/f'certificates_{shard:02d}.json';deps[path.relative_to(ROOT).as_posix()]=digest(path)
        for r in read(path):
            modes=tuple(r['modes']);ix=modes+tuple(j+144 for j in modes);n=len(ix);Q=[[(F(float(V[i,j]))+int(i==j))/2 for j in ix] for i in ix];det=F(0)
            for perm in permutations(range(n)):
                term=math.prod((Q[i][perm[i]] for i in range(n)),start=F(1));det+=(-1 if sum(perm[i]>perm[j] for i in range(n) for j in range(i+1,n))%2 else 1)*term
            lo,hi=r['vacuum_interval'];check('physical determinant '+str(modes),det==r['determinant'] and det>0);check('physical reciprocal root '+str(modes),0<=lo<=hi<=1 and hi-lo in (0,delta) and lo*lo*det<=1<=hi*hi*det and (lo+delta)**2*det>1);check('unique physical subset '+str(modes),modes not in vac);vac[modes]=(lo,hi)
    coeff=read(physical/'coefficients.json');deps['results/iteration40_physical_coefficients/coefficients.json']=digest(physical/'coefficients.json');selection=read(ROOT/'results/iteration40_saturation/selection.json');n=len(selection['prefix']);check('all physical singleton and pair subsets',len(vac)==n*(n+1)//2==s['certificates']);first=[];pairs={}
    for i in range(n):
        a,b=vac[i,];first.append((2*a-1,2*b-1));check(f'physical first {i}',list(first[-1])==coeff['first_intervals'][i])
    for r in coeff['pairs']:
        j,i=r['modes'];a,b=vac[j,i];x,y=vac[j,];z,w=vac[i,];interval=(4*(a-y*w),4*(b-x*z));check(f'physical cumulant {j}:{i}',list(interval)==r['interval']);pairs[j,i]=interval
    check('physical pair count',len(pairs)==n*(n-1)//2);native=read(ROOT/'results/iteration40_saturation/global_target.json');original=read(ROOT/'results/iteration40_saturation/proof.json');raw=read(ROOT/'results/iteration39_global/case_01.stdout.json');deps['results/iteration40_saturation/proof.json']=digest(ROOT/'results/iteration40_saturation/proof.json');native_first=[(F(float.fromhex(x)),)*2 for x in native['first_hex'][:n]];native_pairs={(j,i):(F(float.fromhex(x)),)*2 for j,i,x in native['pairs'] if i<n};cases=[]
    for label,one,two in (('binary64_coefficients',native_first,native_pairs),('physical_coefficients',first,pairs)):
        begin=time.perf_counter();print('recursive proof starting',label,flush=True);result=recursive(one,two,selection['prefix']);max_native=F(0)
        for j,row in enumerate(result['records']):
            a,b=row['threshold'];q=F(float.fromhex(raw['audit_paths'][selection['sample']][j][1]));u=F(float.fromhex(raw['audit_paths'][selection['sample']][j][0]));bit=selection['prefix'][j];check(f'{label} branch {j}',u>b if bit=='1' else u<=a);max_native=max(max_native,abs(a-q),abs(b-q))
            if label=='binary64_coefficients':
                c,d=original['reference']['records'][j]['threshold'];check(f'independent interval overlap {j}',max(a,c)<=min(b,d))
        lo,hi=result['records'][-1]['threshold'];check(label+' strict saturation',1<lo<=hi);record=dict(label=label,result=result,final_threshold=(lo,hi),strict_excess_lower=lo-1,maximum_native_threshold_difference=max_native,seconds=time.perf_counter()-begin);save(out/(label+'.json'),encode(record));cases.append(dict(label=label,threshold_lower=float(lo),threshold_upper=float(hi),width=float(hi-lo),max_native_difference=float(max_native),seconds=record['seconds'],multiplications=result['multiplications']));print('recursive proof finished',cases[-1],flush=True)
    check('verifier unchanged',hashes=={n:digest(ROOT/n) for n in names});save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),physical_determinants=len(vac),source_hashes=hashes,dependencies=deps,cases=cases,scope='Independent recursive outward intervals. Saturation persists with exact physical shifted-target cumulants, but no full-distribution TV bound is inferred'));print('verified saturation',len(checks),flush=True)
if __name__=='__main__':main()
