"""Immutable outputs and predeclared graph panel for iteration 63."""
import gzip, hashlib, json, shutil
from fractions import Fraction as F
from itertools import combinations
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    with open(path,'rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def encode(x):
    if isinstance(x,F): return {'rational':str(x)}
    if isinstance(x,dict): return {str(k):encode(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)): return [encode(v) for v in x]
    return x

def decode(x):
    if isinstance(x,dict):
        if set(x)=={'rational'}: return F(x['rational'])
        return {k:decode(v) for k,v in x.items()}
    if isinstance(x,list): return [decode(v) for v in x]
    return x

def save(path,obj):
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    payload=(json.dumps(encode(obj),indent=2)+'\n').encode()
    if path.suffix=='.gz':
        with path.open('wb') as raw:
            with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as f: f.write(payload)
    else: path.write_bytes(payload)

def read(path):
    return decode(json.loads(gzip.decompress(path.read_bytes()) if path.suffix=='.gz' else path.read_bytes()))

def snapshot(out):
    out.mkdir(exist_ok=False)
    names=sorted({*ROOT.glob('experiments/iteration63*.py'),*ROOT.glob('experiments/verify_iteration63*.py'),ROOT/'src/gbslab/graph_matching_gbs.py',ROOT/'tests/test_graph_matching_gbs.py',ROOT/'research/iteration63_graph_plan.md'})
    hashes={p.relative_to(ROOT).as_posix():digest(p) for p in names}
    for p in names:
        target=out/'source_snapshot'/p.relative_to(ROOT); target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
    return hashes

def panel():
    return [dict(name=name,n=n,edges=edges) for name,n,edges in (
        ('isolated_1',1,[]),('empty_4',4,[]),('path_6',6,[(i,i+1) for i in range(5)]),
        ('cycle_6',6,[(i,i+1) for i in range(5)]+[(0,5)]),('star_6',6,[(0,i) for i in range(1,6)]),
        ('complete_6',6,list(combinations(range(6),2))),
        ('disjoint_triangles',6,[(0,1),(0,2),(1,2),(3,4),(3,5),(4,5)]),
        ('bowtie_5',5,[(0,1),(0,2),(1,2),(0,3),(0,4),(3,4)]),
        ('complete_bipartite_3_3',6,[(i,j) for i in range(3) for j in range(3,6)]),
        ('complete_8',8,list(combinations(range(8),2))))]

def tv(a,b): return sum(abs(x-y) for x,y in zip(a,b))/2
