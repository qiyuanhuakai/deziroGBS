"""Lossless real Gaussian enclosure packing and exact interval Mobius transform."""
from fractions import Fraction as F
from math import lcm
import hashlib,json


def _json_value(x):
    if isinstance(x,F):return {'numerator':x.numerator,'denominator':x.denominator}
    if isinstance(x,dict):return {k:_json_value(v) for k,v in x.items()}
    if isinstance(x,(tuple,list)):return [_json_value(v) for v in x]
    return x


def semantic_digest(record):
    return hashlib.sha256(json.dumps(_json_value(record),sort_keys=True,separators=(',',':'),ensure_ascii=False).encode('utf-8')).hexdigest()


def _matrix(modes,center,center_denominator,denominator):
    m=len(center)//2;ix=list(modes)+[j+m for j in modes];values=[]
    for i,a in enumerate(ix):
        row=[]
        for j,b in enumerate(ix):
            value=(int(center[a][b])+int(i==j)*center_denominator)*denominator
            if value%(2*center_denominator):raise ValueError('Nonintegral recovered matrix')
            row.append(value//(2*center_denominator))
        values.append(row)
    return values


def pack_enclosure(record,center,center_denominator):
    modes=list(record['modes']);n=2*len(modes);c=record['certificate'];R=c['R_real']
    if not modes or modes!=sorted(set(modes)) or not 0<=modes[0]<=modes[-1]<len(center)//2:raise ValueError('Invalid local mode list')
    if c['real']!=_matrix(modes,center,center_denominator,c['denominator']) or any(x for row in c['imag'] for x in row) or any(x for row in c['R_imag'] for x in row):raise ValueError('Certificate is not the exact real local matrix')
    if len(R)!=n or any(len(row)!=n for row in R) or any(R[i][j] for i in range(n) for j in range(i+1,n)):raise ValueError('Real lower triangular witness required')
    packed={k:v for k,v in record.items() if k!='certificate'};packed['certificate']={k:v for k,v in c.items() if k not in ('real','imag','R_real','R_imag')};packed['R_lower']=[int(R[i][j]) for i in range(n) for j in range(i+1)];packed['format']='real-local-enclosure-v1';packed['expanded_sha256']=semantic_digest(record);return packed


def unpack_enclosure(packed,center,center_denominator):
    if packed['format']!='real-local-enclosure-v1':raise ValueError('Unknown enclosure format')
    modes=list(packed['modes']);n=2*len(modes);flat=packed['R_lower']
    if len(flat)!=n*(n+1)//2:raise ValueError('Wrong witness length')
    R=[[0]*n for _ in range(n)];offset=0
    for i in range(n):
        for j in range(i+1):R[i][j]=int(flat[offset]);offset+=1
    record={k:v for k,v in packed.items() if k not in ('format','R_lower','expanded_sha256','certificate')};c=dict(packed['certificate']);c.update(real=_matrix(modes,center,center_denominator,c['denominator']),imag=[[0]*n for _ in range(n)],R_real=R,R_imag=[[0]*n for _ in range(n)]);record['certificate']=c
    if semantic_digest(record)!=packed['expanded_sha256']:raise ValueError('Expanded certificate digest mismatch')
    return record


def integer_mobius_law(vacuum_intervals):
    size=len(vacuum_intervals)
    if size<2 or size&(size-1):raise ValueError('Power-of-two nontrivial interval table required')
    values=[(F(a),F(b)) for a,b in vacuum_intervals]
    if any(a>b for a,b in values):raise ValueError('Invalid interval')
    den=lcm(*(x.denominator for v in values for x in v));lo=[int(a*den) for a,b in values];hi=[int(b*den) for a,b in values];depth=size.bit_length()-1
    for bit in range(depth):
        step=1<<bit
        for start in range(0,size,step*2):
            for offset in range(step):
                a=start+offset;b=a+step;lo[a],hi[a]=lo[a]-hi[b],hi[a]-lo[b]
    raw=[(F(lo[size-1-mask],den),F(hi[size-1-mask],den)) for mask in range(size)];law=[(max(F(0),a),min(F(1),b)) for a,b in raw]
    if any(a>b for a,b in law):raise ArithmeticError('Inconsistent physical atom interval')
    return dict(raw_intervals=raw,intervals=law,common_denominator=den,interval_subtractions=depth*(size//2))
