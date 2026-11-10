"""Independent SymPy recursive expansion of the unclamped K2 polynomial."""
from functools import lru_cache
from pathlib import Path
import json,shutil
import sympy as sp
from iteration22_external import digest,save
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1]

def recursive(n):
    p=sp.symbols('s:'+str(n));c={(j,i):sp.Symbol(f'c{j}_{i}') for i in range(n) for j in range(i)}
    expand=sp.expand
    @lru_cache(None)
    def main(end):
        if end==0:return sp.Integer(1)
        last=end-1;return expand(main(last)*p[last]+sum(c[j,last]*elided(j,last) for j in range(last)))
    @lru_cache(None)
    def partial(begin,end):
        if begin==end:return sp.Integer(1)
        if begin==0:return main(end)
        last=end-1;return expand(partial(begin,last)*p[last]+sum(c[j,last]*pe(begin,j,last) for j in range(begin,last)))
    @lru_cache(None)
    def elided(e,end):
        if e==0:return partial(1,end)
        if e==end-1:return main(end-1)
        last=end-1;return expand(elided(e,last)*p[last]+sum(c[j,last]*de(min(j,e),max(j,e),last) for j in range(last) if j!=e))
    @lru_cache(None)
    def pe(begin,e,end):
        if begin==e:return partial(begin+1,end)
        if begin==0:return elided(e,end)
        if e==end-1:return partial(begin,end-1)
        last=end-1;return expand(pe(begin,e,last)*p[last]+sum(c[j,last]*pe(begin,min(j,e),max(j,e))*partial(max(j,e)+1,last) for j in range(begin,last) if j!=e))
    @lru_cache(None)
    def de(a,b,end):
        if a==0 and b==1:return partial(2,end)
        if a==0 and b==end-1:return partial(1,end-1)
        if a==end-2 and b==end-1:return main(end-2)
        if b==end-1:return elided(a,end-1)
        last=end-1;total=sp.Integer(0)
        for j in range(last):
            if j not in (a,b):
                x,y,z=sorted((a,b,j));total+=c[j,last]*de(x,y,z)*partial(z+1,last)
        return expand(de(a,b,last)*p[last]+total)
    # Enumerate set partitions with block sizes one or two, without sparse-poly arithmetic.
    def blocks(items):
        if not items:yield [];return
        i=items[0];rest=items[1:]
        for tail in blocks(rest):yield [(i,)]+tail
        for j in rest:
            for tail in blocks(tuple(k for k in rest if k!=j)):yield [(i,j)]+tail
    terms=[sp.prod(p[b[0]] if len(b)==1 else c[b] for b in partition) for partition in blocks(tuple(range(n)))]
    return sp.Add(*terms),main(n)

def main():
    out=ROOT/'results/iteration41_symbolic_validation';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration41_symbolic','iteration22_external','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    checks=[];deps={}
    def check(label,condition):
        checks.append(dict(name=label,passed=bool(condition)))
        if not condition:save(out/'checks.failed.json',checks);raise AssertionError(label)
    original=ROOT/'results/iteration41_symbolic';summary=json.loads((original/'summary.json').read_text())
    for name,sha in summary['source_hashes'].items():check('source '+name,digest(ROOT/name)==sha);check('snapshot '+name,digest(original/'source_snapshot'/name)==sha)
    for n in range(1,9):
        path=original/f'case_{n:02d}.json';record=decode(json.loads(path.read_text()));deps[path.relative_to(ROOT).as_posix()]=digest(path);exact,approx=recursive(n);delta=sp.expand(exact-approx)
        expected=sum(sp.Rational(r['coefficient'].numerator,r['coefficient'].denominator)*sp.prod(sp.Symbol(v) for v in r['monomial']) for r in record['difference'])
        check('independent polynomial identity '+str(n),sp.expand(delta-expected)==0);check('matching term count '+str(n),len(sp.Add.make_args(exact))==record['matching_terms']);check('recurrence term count '+str(n),len(sp.Add.make_args(approx))==record['recurrence_terms'])
    check('verifier unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps,sympy_version=sp.__version__,scope='Independent symbolic arithmetic and recursive evaluation of unclamped real K2 formulas'));print('symbolic verified',len(checks),flush=True)
if __name__=='__main__':main()
