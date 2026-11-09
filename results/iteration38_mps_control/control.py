"""Physical product-TMSV control: maximum Schmidt tail is not output TV."""
import json,math,hashlib
from fractions import Fraction as F
from pathlib import Path
OUT=Path(__file__).parent
def determinant(matrix):
    a=[row[:] for row in matrix];d=F(1)
    for k in range(len(a)):
        pivot=next(i for i in range(k,len(a)) if a[i][k])
        if pivot!=k:a[k],a[pivot]=a[pivot],a[k];d=-d
        v=a[k][k];d*=v
        for i in range(k+1,len(a)):
            multiplier=a[i][k]/v
            for j in range(k+1,len(a)):a[i][j]-=multiplier*a[k][j]
    return d
def encode(v):
    if isinstance(v,F):return dict(numerator=v.numerator,denominator=v.denominator)
    if isinstance(v,dict):return {k:encode(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [encode(x) for x in v]
    return v
def main():
    p=OUT/'result.json'
    if p.exists():raise FileExistsError(p)
    b=F(1,10);epsilon=b*b;pair_count=4;m=2*pair_count;V=[[F(0) for j in range(2*m)] for i in range(2*m)]
    for j in range(m):
        V[j][j]=V[j+m][j+m]=(1+b*b)/(1-b*b);partner=j^1;V[j][partner]=2*b/(1-b*b);V[j+m][partner+m]=-2*b/(1-b*b)
    vacua=[];matrix_checks=0
    for mask in range(1<<m):
        chosen=[j for j in range(m) if mask>>j&1];ix=chosen+[j+m for j in chosen];Q=[[(V[i][j]+int(i==j))/2 for j in ix] for i in ix];det=determinant(Q);a=math.isqrt(det.numerator);c=math.isqrt(det.denominator);assert a*a==det.numerator and c*c==det.denominator;prob=F(c,a);touched=sum(bool(mask&(3<<(2*j))) for j in range(pair_count));assert prob==(1-epsilon)**touched;vacua.append(prob);matrix_checks+=1
    law=[]
    for clicks in range(1<<m):
        zero=((1<<m)-1)^clicks;sub=clicks;prob=F(0)
        while True:
            prob+=(-1)**sub.bit_count()*vacua[zero|sub]
            if not sub:break
            sub=(sub-1)&clicks
        expected=F(1)
        for j in range(pair_count):
            bits=(clicks>>(2*j))&3;expected*=1-epsilon if bits==0 else epsilon if bits==3 else 0
        assert prob==expected;law.append(prob)
    tv=sum(abs(v-int(i==0)) for i,v in enumerate(law))/2;assert sum(law)==1 and tv==1-(1-epsilon)**pair_count;cases=[dict(pairs=k,modes=2*k,maximum_discarded_Schmidt_weight=epsilon,normalized_rank_one_output_TV=1-(1-epsilon)**k,TV_decimal=float(1-(1-epsilon)**k)) for k in (1,2,4,10,72,100,512)];result=dict(pairing_amplitude=b,epsilon=epsilon,physical_construction='Tensor product of normalized sqrt(1-b^2) sum_n b^n |n,n> states; adjacent mode pairs; rank-one approximation is global vacuum',gaussian_vacuum_determinants_checked=matrix_checks,complete_eight_mode_atoms_checked=len(law),eight_mode_law=law,cases=cases,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Analytic physical control, not the 144-mode research target and not an assertion that a cited paper claims maximum bond tail equals sampling TV');p.write_text(json.dumps(encode(result),indent=2)+'\n');print(json.dumps(cases,default=str),flush=True)
if __name__=='__main__':main()
