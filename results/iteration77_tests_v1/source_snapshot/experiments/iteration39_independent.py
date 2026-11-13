"""Independent mt19937_64 replay, direct CDF cells, and depth-first laws."""
import math,struct
from fractions import Fraction as F
MASK=(1<<64)-1;WORDS=1<<64


class MT64:
    def __init__(self,seed):
        self.state=[int(seed)&MASK]
        for i in range(1,312):
            x=self.state[-1];self.state.append((6364136223846793005*(x^(x>>62))+i)&MASK)
        self.index=312
    def next(self):
        if self.index==312:
            for i in range(312):
                y=(self.state[i]&0xffffffff80000000)|(self.state[(i+1)%312]&0x7fffffff)
                self.state[i]=self.state[(i+156)%312]^(y>>1)^(0xb5026f5aa96619e9 if y&1 else 0)
            self.index=0
        x=self.state[self.index];self.index+=1;x^=(x>>29)&0x5555555555555555;x^=(x<<17)&0x71d67fffeda60000;x^=(x<<37)&0xfff7eee000000000;x^=x>>43;return x&MASK


def mapped_float(word):return min(float(word)*2.**-64,math.nextafter(1.,0.))


def cdf_count(q):
    q=float(q)
    if math.isnan(q):return WORDS
    if q<0:return 0
    if q>=math.nextafter(1.,0.):return WORDS
    point=F(q)*WORDS
    if point<(1<<53):last=point.numerator//point.denominator
    else:
        adjacent=F(math.nextafter(q,math.inf))*WORDS;mid=(point+adjacent)/2;last=mid.numerator//mid.denominator
        bits=struct.unpack('>Q',struct.pack('>d',q))[0]
        if mid.denominator==1 and bits&1:last-=1
    return last+1


def reconstruct(modes,table,model):
    probabilities=[F(0)]*(1<<modes);todo=[('',F(1),0)]
    while todo:
        prefix,mass,mask=todo.pop();j=len(prefix)
        if j==modes:probabilities[mask]=mass;continue
        q=float.fromhex(table[prefix])
        if model=='uint64':zero=F(cdf_count(q),WORDS)
        else:zero=F(1) if math.isnan(q) or q>=1 else F(0) if q<0 else F(q)
        todo.append((prefix+'0',mass*zero,mask));todo.append((prefix+'1',mass*(1-zero),mask|(1<<j)))
    assert sum(probabilities)==1
    return probabilities


def replay(rec):
    assert rec['threads']==1 and rec['audit_enabled'] and rec['audit_identical_outputs'];n=rec['samples'];m=rec['modes'];lanes=rec['simd_lanes'];seed_engine=MT64(rec['seed']);engine=MT64(seed_engine.next()&0xffffffff);checked=0;counts=[0]*(m+1);nonfinite=saturated=0;table={r['prefix']:r['threshold_hex'] for r in rec['prefix_tree']}
    for batch in range((n+lanes-1)//lanes):
        previous=[1.]*lanes
        for j in range(m):
            for lane in range(lanes):
                u=mapped_float(engine.next());s=batch*lanes+lane
                if s>=n:continue
                row=rec['audit_paths'][s][j];stored,q,prev,z=map(float.fromhex,row);assert stored==u and prev==previous[lane]
                if prev==0:assert z==0 and math.isnan(q)
                elif math.isfinite(prev) and math.isfinite(z):
                    try:expected=float(F(z)/F(prev))
                    except OverflowError:expected=math.copysign(math.inf,z/abs(z)*prev/abs(prev))
                    assert q==expected
                bit=int(u>q);assert bit==int(rec['outputs'][s][j]);previous[lane]=prev-z if bit else z;checked+=1;nonfinite+=not math.isfinite(q);saturated+=q<0 or q>1
                if table:
                    other=float.fromhex(table[rec['outputs'][s][:j]]);assert q==other or math.isnan(q) and math.isnan(other)
    for s in rec['outputs']:assert len(s)==m and set(s)<={'0','1'};counts[s.count('1')]+=1
    assert counts==rec['count_histogram'] and nonfinite==rec['audit_nonfinite_steps'] and saturated==rec['audit_saturated_steps'];return checked
