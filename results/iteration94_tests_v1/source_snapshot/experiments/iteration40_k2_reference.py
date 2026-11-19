"""Outward dyadic enclosure of the real-arithmetic official K=2 recurrence.

The target coefficients are fixed binary64 values. This encloses the same
approximate marginal recurrence with exact real arithmetic, including its
absolute [0,1] clamps; it does not enclose exact physical conditional masses.
"""
from fractions import Fraction as F


class Dyadic:
    def __init__(self,bits):
        if type(bits) is not int or bits<128:raise ValueError('At least128 fractional bits required')
        self.bits=bits;self.scale=1<<bits;self.one=(self.scale,self.scale);self.zero=(0,0);self.operations=0
    def enclose(self,x):
        x=F(x)*self.scale;return x.numerator//x.denominator,-((-x.numerator)//x.denominator)
    def exact(self,x):
        a,b=self.enclose(x)
        if a!=b:raise ValueError('Coefficient is not exactly represented on this grid')
        return a
    def add(self,a,b):self.operations+=1;return a[0]+b[0],a[1]+b[1]
    def neg(self,a):return -a[1],-a[0]
    def mul(self,a,b):
        self.operations+=1
        if a[0]>=0 and b[0]>=0:lo=a[0]*b[0];hi=a[1]*b[1]
        else:v=(a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]);lo=min(v);hi=max(v)
        return lo//self.scale,-((-hi)//self.scale)
    def coefficient(self,a,c):
        self.operations+=1;lo,hi=(a[0]*c,a[1]*c) if c>=0 else (a[1]*c,a[0]*c)
        return lo//self.scale,-((-hi)//self.scale)
    def clamp(self,a):self.operations+=1;return max(0,min(self.scale,a[0])),max(0,min(self.scale,a[1]))
    def fractions(self,a):return F(a[0],self.scale),F(a[1],self.scale)
    def ratio(self,a,b):
        if b[0]<=0:raise ArithmeticError('Unresolved/nonpositive denominator')
        values=(F(a[0],b[0]),F(a[0],b[1]),F(a[1],b[0]),F(a[1],b[1]));return min(values),max(values)


def parameters(target,prefix,bits):
    if target['cutoff']!=2 or len(prefix)>target['modes'] or set(prefix)-{'0','1'}:raise ValueError('K=2 target and binary prefix required')
    ar=Dyadic(bits);c=[F(float.fromhex(x)) for x in target['first_hex']];pairs={(j,i):F(float.fromhex(x)) for j,i,x in target['pairs']};m=target['modes'];assert len(c)==m and len(pairs)==m*(m-1)//2;sign=[1 if b=='0' else -1 for b in prefix];positive=[ar.exact((1+x)/2) for x in c];selected=[ar.exact((1+sign[i]*c[i])/2) for i in range(len(prefix))];weighted={(j,i):ar.exact(F(sign[j],4)*pairs[j,i]) for i in range(len(prefix)) for j in range(i)};return ar,sign,positive,selected,weighted


def forward(target,prefix,*,bits=384,progress=None):
    ar,sign,pos,selected,k=parameters(target,prefix,bits);P={};E={};PE={};DE={};previous=ar.one;records=[]
    def partial(a,end):return ar.one if a==end else P[a,end]
    def partial_elided(a,e,end):return partial(a+1,end) if a==e else PE[a,e,end]
    def step(previous,coefficient,total,last):return ar.clamp(ar.add(ar.coefficient(previous,coefficient),total if sign[last]==1 else ar.neg(total)))
    for i in range(len(prefix)):
        for begin in range(i):
            if begin==0:P[begin,i]=previous;continue
            last=i-1;total=ar.zero
            for j in range(begin,last):total=ar.add(total,ar.coefficient(partial_elided(begin,j,last),k[j,last]))
            P[begin,i]=step(partial(begin,last),selected[last],total,last)
        for e in range(i):
            if e==0:E[e,i]=partial(1,i);continue
            if e==i-1:E[e,i]=partial(0,i-1);continue
            last=i-1;total=ar.zero
            for j in range(last):
                if j!=e:total=ar.add(total,ar.coefficient(DE[min(j,e),max(j,e),last],k[j,last]))
            E[e,i]=step(E[e,last],selected[last],total,last)
        for e in range(i):
            for begin in range(e):
                if begin==0:PE[begin,e,i]=E[e,i];continue
                if e==i-1:PE[begin,e,i]=partial(begin,i-1);continue
                last=i-1;total=ar.zero
                for j in range(begin,last):
                    if j==e:continue
                    left,right=min(j,e),max(j,e);factor=ar.mul(partial_elided(begin,left,right),partial(right+1,last));total=ar.add(total,ar.coefficient(factor,k[j,last]))
                PE[begin,e,i]=step(partial_elided(begin,e,last),selected[last],total,last)
        for b in range(i):
            for a in range(b):
                if a==0 and b==1:DE[a,b,i]=partial(2,i);continue
                if a==0 and b==i-1:DE[a,b,i]=partial(1,i-1);continue
                if a==i-2 and b==i-1:DE[a,b,i]=partial(0,i-2);continue
                if b==i-1:DE[a,b,i]=E[a,i-1];continue
                last=i-1;total=ar.zero
                for j in range(last):
                    if j==a or j==b:continue
                    x,y,z=sorted((j,a,b));factor=ar.mul(DE[x,y,z],partial(z+1,last));total=ar.add(total,ar.coefficient(factor,k[j,last]))
                DE[a,b,i]=step(DE[a,b,last],selected[last],total,last)
        total=ar.zero
        for j in range(i):total=ar.add(total,ar.coefficient(E[j,i],k[j,i]))
        zero=ar.clamp(ar.add(ar.coefficient(previous,pos[i]),total));ratio=ar.ratio(zero,previous);records.append(dict(mode=i,previous=ar.fractions(previous),zero=ar.fractions(zero),threshold=ratio));previous=zero if prefix[i]=='0' else ar.add(previous,ar.neg(zero))
        if previous[0]<=0:raise ArithmeticError('Selected history not certified to have positive recurrence mass')
        if progress is not None and (i+1)%16==0:progress(i+1,ar.operations)
    return dict(bits=bits,records=records,operations=ar.operations,table_entries=len(P)+len(E)+len(PE)+len(DE),final_previous=ar.fractions(previous))
