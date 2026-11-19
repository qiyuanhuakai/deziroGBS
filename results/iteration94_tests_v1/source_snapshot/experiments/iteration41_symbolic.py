"""Exact sparse-polynomial identity for the first K2 marginal factorization gap."""
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path
import json,shutil
from iteration41_independent import forward
from iteration22_external import digest,save
from iteration33_run import encode
ROOT=Path(__file__).resolve().parents[1]

class Poly:
    def __init__(self,terms):self.terms={tuple(k):F(v) for k,v in terms.items() if v}
    @staticmethod
    def cast(x):return x if isinstance(x,Poly) else Poly({():F(x)})
    @staticmethod
    def variable(name):return Poly({(name,):F(1)})
    def __add__(self,other):
        result=self.terms.copy()
        for k,v in self.cast(other).terms.items():result[k]=result.get(k,F(0))+v
        return Poly(result)
    __radd__=__add__
    def __neg__(self):return Poly({k:-v for k,v in self.terms.items()})
    def __sub__(self,other):return self+-self.cast(other)
    def __rsub__(self,other):return self.cast(other)+-self
    def __mul__(self,other):
        result={}
        for a,x in self.terms.items():
            for b,y in self.cast(other).terms.items():
                k=tuple(sorted(a+b));result[k]=result.get(k,F(0))+x*y
        return Poly(result)
    __rmul__=__mul__
    def evaluate(self,values):
        total=F(0)
        for monomial,coefficient in self.terms.items():
            for variable in monomial:coefficient*=values[variable]
            total+=coefficient
        return total

def identity(n):
    p=[Poly.variable(f's{i}') for i in range(n)];c={(j,i):Poly.variable(f'c{j}_{i}') for i in range(n) for j in range(i)}
    @lru_cache(None)
    def partitions(indices):
        if not indices:return Poly({():1})
        i=indices[0];rest=indices[1:];out=p[i]*partitions(rest)
        for j in rest:out+=c[i,j]*partitions(tuple(k for k in rest if k!=j))
        return out
    exact=partitions(tuple(range(n)));approx=forward('0'*n,p,c,False)['mass'];difference=exact-approx
    return exact,approx,difference

def main():
    out=ROOT/'results/iteration41_symbolic';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('iteration41_symbolic','iteration41_independent','iteration22_external','iteration33_run')];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    rows=[]
    for n in range(1,9):
        exact,approx,delta=identity(n);row=dict(modes=n,matching_terms=len(exact.terms),recurrence_terms=len(approx.terms),difference=[dict(monomial=list(k),coefficient=v) for k,v in sorted(delta.terms.items())]);save(out/f'case_{n:02d}.json',encode(row))
        if n<=7:assert not delta.terms
        else:
            assert len(delta.terms)==16
            for term,value in delta.terms.items():
                assert value==1 and len(term)==4 and all(t.startswith('c') for t in term)
                assert sorted(int(i) for t in term for i in t[1:].split('_'))==list(range(8))
            numeric={f's{i}':F(4,5) for i in range(8)};numeric.update({f'c{j}_{i}':F(2,75) for i in range(8) for j in range(i)})
            assert delta.evaluate(numeric)==16*F(2,75)**4
        rows.append(dict(modes=n,matching_terms=len(exact.terms),recurrence_terms=len(approx.terms),missing_terms=len(delta.terms)))
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(rows=rows,source_hashes=hashes,unchanged_sources=True,scope='Unclamped real K2 recurrence: symbolic equality through7, 16 missing perfect matchings at8; not a full physical-law equality or a claim about clipped recursion'));print(json.dumps(rows),flush=True)
if __name__=='__main__':main()
