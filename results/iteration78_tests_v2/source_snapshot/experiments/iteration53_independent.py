"""Independent compact evidence inflation and direct integer inclusion."""
from fractions import Fraction as F
from math import gcd
import json,hashlib

def expand_record(packed,center,dc):
    assert packed['format']=='real-local-enclosure-v1';m=len(center)//2;modes=packed['modes'];ix=modes+[m+j for j in modes];n=len(ix);cert=dict(packed['certificate']);d=cert['denominator'];real=[]
    for i,a in enumerate(ix):
        row=[]
        for j,b in enumerate(ix):
            q=F(int(center[a,b])+int(i==j)*dc,2*dc)*d;assert q.denominator==1;row.append(q.numerator)
        real.append(row)
    values=packed['R_lower'];assert len(values)==n*(n+1)//2;it=iter(values);R=[[next(it) if j<=i else 0 for j in range(n)] for i in range(n)];cert.update(real=real,imag=[[0]*n for _ in range(n)],R_real=R,R_imag=[[0]*n for _ in range(n)]);r={k:v for k,v in packed.items() if k not in ('certificate','format','R_lower','expanded_sha256')};r['certificate']=cert
    def default(value):
        if isinstance(value,F):return dict(numerator=value.numerator,denominator=value.denominator)
        raise TypeError(type(value))
    payload=json.dumps(r,default=default,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8');assert hashlib.sha256(payload).hexdigest()==packed['expanded_sha256'];return r

def direct_integer_law(vac):
    size=len(vac);assert size>1 and not size&(size-1);den=1
    for a,b in vac:
        for x in (a,b):den=den*x.denominator//gcd(den,x.denominator)
    L=[int(a*den) for a,b in vac];H=[int(b*den) for a,b in vac];raw=[];law=[];terms=0
    for click in range(size):
        zeros=(size-1)^click;subset=click;low=high=0
        while True:
            terms+=1;j=zeros|subset
            if subset.bit_count()%2:low-=H[j];high-=L[j]
            else:low+=L[j];high+=H[j]
            if not subset:break
            subset=(subset-1)&click
        a,b=F(low,den),F(high,den);raw.append((a,b));law.append((max(F(0),a),min(F(1),b)))
    return raw,law,terms
