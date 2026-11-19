"""Build one native library in a new evidence directory."""
import argparse
import re
from iteration96_group_common_v1 import ROOT, Path, H, save, read, manifest, subprocess
from time import perf_counter

p=argparse.ArgumentParser()
p.add_argument('kind', choices=['tree','frugal'])
p.add_argument('output',type=Path)
args=p.parse_args()
out=args.output.resolve()
out.mkdir(parents=True,exist_ok=False)
if args.kind=='tree':
    source=ROOT/'native/gaussian_group_tree96_v1.cpp'
    base=ROOT/'results/iteration95_loop_alloc_v1/build.json'
    cmd=read(base)['command']
    cmd[-3]=str(source)
else:
    source=ROOT/'native/frugal_tree96_v1.cpp'
    base=ROOT/'results/iteration83_frugal_build_v2/build.json'
    old=str(ROOT.parent/'deziroGBS')+str(Path('/'))
    cmd=[v.replace(old,str(ROOT)+str(Path('/'))) for v in read(base)['command']]
    cmd=[str(source) if v.endswith('frugal_full83_adapter_v1.cpp') else v for v in cmd]
cmd[-1]=str(out/(args.kind+'.dll'))
sources={Path(__file__),source,base,ROOT/'experiments/iteration96_group_common_v1.py'}
todo=[source]
while todo:
    parent=todo.pop()
    for name in re.findall(r'^#include\s+"([^"]+)"',parent.read_text(encoding='utf8'),re.M):
        f=parent.parent/name
        if f.exists() and f not in sources:
            sources.add(f)
            todo.append(f)
save(out,'plan.json',dict(parent='a5b9a10aabac46874779e592a8f1ac33b90017d3',
    hashes={f.relative_to(ROOT).as_posix():H(f) for f in sorted(sources)}))
start=perf_counter()
run=subprocess.run(cmd,capture_output=True,text=True)
save(out,'build.json',dict(command=cmd,code=run.returncode,stderr=run.stderr,seconds=perf_counter()-start))
if run.returncode:
    raise RuntimeError(run.stderr[-4000:])
manifest(out)
