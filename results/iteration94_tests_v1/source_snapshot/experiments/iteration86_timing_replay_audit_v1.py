"""Separate repeated-execution drift from old/new trajectory differences.
After the first differing bit, conditional probabilities refer to different
histories; only the prefix through that decision is compared as a common event.
"""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);base=ROOT/'results/iteration86_shared_timing_v1'
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
files=[Path(__file__),base/'summary.json',*sorted(base.glob('raw*.npz'))];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}));repeated=[];cross=[]
for count in (4,64):
 for method in ('old','shared'):
  for sector in range(2):
   original=np.load(base/f'raw_0_{method}_{count}_{sector}.npz')
   for rep in (1,2):
    again=np.load(base/f'raw_{rep}_{method}_{count}_{sector}.npz');row=dict(records=count,method=method,sector=sector,rep=rep,fields={})
    for field in ('clicks','conditional','errors','statistics'):
     x,y=original[field],again[field];row['fields'][field]=dict(different_entries=int(np.count_nonzero(x!=y)),maximum_absolute_difference=float(np.nanmax(abs(x.astype(float)-y.astype(float)))))
    repeated.append(row)
 for sector in range(2):
  old=np.load(base/f'raw_0_old_{count}_{sector}.npz');new=np.load(base/f'raw_0_shared_{count}_{sector}.npz');different=old['clicks']!=new['clicks'];details=[];maximum=0.
  for record in range(count):
   indices=np.flatnonzero(different[record]);stop=int(indices[0]) if len(indices) else 1151;maximum=max(maximum,float(abs(old['conditional'][record,:stop+1]-new['conditional'][record,:stop+1]).max()))
   if len(indices):details.append(dict(record=record,first_different_mode=stop,different_bits=len(indices),old_first_probability=float(old['conditional'][record,stop]),new_first_probability=float(new['conditional'][record,stop])))
  cross.append(dict(records=count,sector=sector,different_records=int(different.any(1).sum()),different_bits=int(different.sum()),maximum_conditional_difference_on_same_prefix=maximum,details=details))
save('summary.json',dict(repeated=repeated,cross=cross,scope=__doc__));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}));print(json.dumps(dict(cross=cross,old_repeat_probability_difference=max(r['fields']['conditional']['maximum_absolute_difference'] for r in repeated if r['method']=='old'),old_repeat_different_bits=sum(r['fields']['clicks']['different_entries'] for r in repeated if r['method']=='old')),indent=2),flush=True)
