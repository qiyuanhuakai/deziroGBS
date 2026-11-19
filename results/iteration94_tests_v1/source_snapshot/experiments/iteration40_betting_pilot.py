"""Exploratory comparison on prior cohorts; no new confirmation claims."""
import sys,json,shutil,hashlib,time
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.set_int_max_str_digits(0)
from gbslab.betting_mean import mean_interval
from gbslab.mean_witness import product_mean_interval
from iteration33_run import encode
from verify_iteration34 import decode
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'results/iteration40_betting_pilot';out.mkdir(exist_ok=False);names=['experiments/iteration40_betting_pilot.py','src/gbslab/betting_mean.py','src/gbslab/mean_witness.py','experiments/iteration33_run.py','experiments/verify_iteration34.py'];hashes={n:digest(ROOT/n) for n in names};deps={};rows=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    for label,folder,it,n in (('frugal','iteration39_global',39,4),('forest','iteration38_confirmation',38,4)):
        manifest=json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())
        for i in range(n):
            name=f'results/{folder}/case_{i:02d}.'+('stdout.json' if label=='frugal' else 'audit.json');p=ROOT/name;deps[name]=digest(p);assert deps[name]==manifest['artifacts'][name];r=decode(json.loads(p.read_text()));counts=[s.count('1') for s in r['outputs']] if label=='frugal' else [a['clicks'] if a['outcome']=='completed' else None for a in r['attempts']]
            for observable in ('X','Y72'):
                values=[F(0) if c is None else F(c,144) if observable=='X' else F((c-72)**2,5184) for c in counts];begin=time.perf_counter();new=mean_interval(values,F(1,5700));old=product_mean_interval(values,F(1,5700));row=dict(source=name,observable=observable,attempts=len(values),new=new,old=old,seconds=time.perf_counter()-begin,exploratory_only=True);rows.append(row);print(label,i,observable,'new',float(new['lower']),float(new['upper']),'old',float(old['lower']),float(old['upper']),flush=True)
    assert all(digest(ROOT/n)==sha for n,sha in hashes.items());(out/'results.json').write_text(json.dumps(encode(rows),indent=2)+'\n');(out/'summary.json').write_text(json.dumps(dict(rows=len(rows),source_hashes=hashes,dependencies=deps,unchanged_sources=True,exploratory_only=True),indent=2)+'\n')
if __name__=='__main__':main()
