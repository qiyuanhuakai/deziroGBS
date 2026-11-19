"""Independent forward tables, parity transform, determinant and word CDF."""
import math
from fractions import Fraction as F

def forward(prefix,p,c,clamps=True):
    sign=[1 if x=='0' else -1 for x in prefix];selected=[p[i] if sign[i]==1 else 1-p[i] for i in range(len(prefix))]
    k={(j,i):sign[j]*c[j,i] for i in range(len(prefix)) for j in range(i)}
    P={};E={};PE={};DE={};previous=F(1);records=[];clipped=0
    def clip(x):
        nonlocal clipped
        if not clamps:return x
        y=max(F(0),min(F(1),x));clipped+=y!=x;return y
    def partial(a,end):return F(1) if a==end else P[a,end]
    def partial_elided(a,e,end):return partial(a+1,end) if a==e else PE[a,e,end]
    def step(previous,total,last):return clip(previous*selected[last]+sign[last]*total)
    for i in range(len(prefix)):
        for begin in range(i):
            if begin==0:P[begin,i]=previous;continue
            last=i-1;total=sum((partial_elided(begin,j,last)*k[j,last] for j in range(begin,last)),F(0))
            P[begin,i]=step(partial(begin,last),total,last)
        for e in range(i):
            if e==0:E[e,i]=partial(1,i);continue
            if e==i-1:E[e,i]=partial(0,i-1);continue
            last=i-1;total=sum((DE[min(j,e),max(j,e),last]*k[j,last] for j in range(last) if j!=e),F(0))
            E[e,i]=step(E[e,last],total,last)
        for e in range(i):
            for begin in range(e):
                if begin==0:PE[begin,e,i]=E[e,i];continue
                if e==i-1:PE[begin,e,i]=partial(begin,i-1);continue
                last=i-1;total=F(0)
                for j in range(begin,last):
                    if j!=e:
                        left,right=min(j,e),max(j,e);total+=partial_elided(begin,left,right)*partial(right+1,last)*k[j,last]
                PE[begin,e,i]=step(partial_elided(begin,e,last),total,last)
        for b in range(i):
            for a in range(b):
                if a==0 and b==1:DE[a,b,i]=partial(2,i);continue
                if a==0 and b==i-1:DE[a,b,i]=partial(1,i-1);continue
                if a==i-2 and b==i-1:DE[a,b,i]=partial(0,i-2);continue
                if b==i-1:DE[a,b,i]=E[a,i-1];continue
                last=i-1;total=F(0)
                for j in range(last):
                    if j not in (a,b):
                        x,y,z=sorted((j,a,b));total+=DE[x,y,z]*partial(z+1,last)*k[j,last]
                DE[a,b,i]=step(DE[a,b,last],total,last)
        z=clip(previous*p[i]+sum((E[j,i]*k[j,i] for j in range(i)),F(0)))
        records.append((previous,z));previous=z if sign[i]==1 else previous-z
    return dict(mass=previous,records=records,clipped=clipped)

def parity_law(p,c):
    n=len(p);moments=[F(0)]*(1<<n);moments[0]=F(1)
    for mask in range(1,1<<n):
        bit=mask&-mask;i=bit.bit_length()-1;rest=mask^bit;value=(2*p[i]-1)*moments[rest]
        for j in range(i+1,n):
            if (rest>>j)&1:value+=4*c[i,j]*moments[rest^(1<<j)]
        moments[mask]=value
    transformed=moments.copy();size=1
    while size<len(transformed):
        for begin in range(0,len(transformed),2*size):
            for j in range(size):
                a,b=transformed[begin+j],transformed[begin+j+size];transformed[begin+j]=a+b;transformed[begin+j+size]=a-b
        size*=2
    return [x/(1<<n) for x in transformed]

def determinant(matrix):
    a=[list(map(F,row)) for row in matrix];n=len(a);det=F(1)
    for i in range(n):
        k=next(k for k in range(i,n) if a[k][i])
        if k!=i:a[i],a[k]=a[k],a[i];det=-det
        pivot=a[i][i];det*=pivot
        for j in range(i+1,n):
            f=a[j][i]/pivot
            for k in range(i+1,n):a[j][k]-=f*a[i][k]
    return det

def target_from_vacua(vac,n):
    # Start with the exact probabilities that each zero subset is all vacuum.
    values=list(vac)
    for bit in range(n):
        for mask in range(1<<n):
            if not (mask>>bit)&1:values[mask]-=values[mask|(1<<bit)]
    return [values[((1<<n)-1)^mask] for mask in range(1<<n)]

def word_probability(q):
    if math.isnan(q):return F(1)
    low,high=0,1<<64
    # Search the monotone uint64-to-double map directly, independent of closed CDF cells.
    while low<high:
        mid=(low+high)//2;value=min(float(mid)*2.**-64,math.nextafter(1.,0.))
        if value<=q:low=mid+1
        else:high=mid
    return F(low,1<<64)

def layer_law(table,n,model):
    layer={0:F(1)}
    for j in range(n):
        next_layer={}
        for mask,mass in layer.items():
            prefix=''.join(str((mask>>k)&1) for k in range(j));q=float.fromhex(table[prefix])
            if model=='uint64':p=word_probability(q)
            else:p=F(1) if math.isnan(q) or q>=1 else F(0) if q<=0 else F(q)
            next_layer[mask]=mass*p;next_layer[mask|(1<<j)]=mass*(1-p)
        layer=next_layer
    return [layer[k] for k in range(1<<n)]
