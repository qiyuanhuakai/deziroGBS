"""Audit the completed pilot using independent random words and exact tree laws."""
import sys,json,hashlib,shutil,subprocess
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.set_int_max_str_digits(0)
from gbslab.frugal_law import word_uniform,zero_probability,full_law,uniform_model_transfer_bound
from iteration39_independent import mapped_float,cdf_count,reconstruct,replay
from iteration33_run import encode
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,r):Path(p).write_text(json.dumps(encode(r),indent=2)+'\n')
def main():
    out=ROOT/'results/iteration39_calibration';out.mkdir(exist_ok=False);pilot=ROOT/'results/iteration39_pilot';summary=json.loads((pilot/'summary.json').read_text());assert summary['completed']==summary['planned']==19;names=['experiments/iteration39_calibrate.py','experiments/iteration39_independent.py','src/gbslab/frugal_law.py'];hashes={n:digest(ROOT/n) for n in names}
    for n in names:dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    checks=[];dependencies={};steps=0;trees=[]
    def read(p):dependencies[p.relative_to(ROOT).as_posix()]=digest(p);return json.loads(p.read_text())
    def check(name,ok):checks.append(dict(name=name,passed=bool(ok)));assert ok,name
    for i,(w,u,calls) in enumerate(read(pilot/'rng_probe.json')['rows']):check('uniform word '+str(i),calls==1 and F(float.fromhex(u))==word_uniform(int(w))==F(mapped_float(int(w))))
    for i in range(19):
        r=read(pilot/f'case_{i:02d}.stdout.json');steps+=replay(r);check('independent MT and branch replay '+str(i),True)
        if r['prefix_tree']:
            table={x['prefix']:x['threshold_hex'] for x in r['prefix_tree']}
            for prefix,q in table.items():check(f'CDF {i}:{prefix}',zero_probability(float.fromhex(q))==F(cdf_count(float.fromhex(q)),1<<64))
            laws={}
            for model in ('continuous','uint64'):
                law=full_law(r['modes'],r['prefix_tree'],model=model);independent=reconstruct(r['modes'],table,model);check(f'full law {i}:{model}',tuple(independent)==law['distribution']);laws[model]=law
            tv=sum(abs(a-b) for a,b in zip(laws['continuous']['distribution'],laws['uint64']['distribution']))/2;check('uniform-model coupling '+str(i),tv<=uniform_model_transfer_bound(r['modes']));trees.append(dict(case=i,laws=laws,model_TV=tv))
    exe=ROOT/'results/iteration39_build/frugal_extended.exe';base=[str(exe),'--input',str(pilot/'case_18.csv'),'--modes','144','--samples','0'];guards=[['--cutoff','5'],['--cutoff','2','--max-layout','1'],['--cutoff','2','--prefix-tree'],['--cutoff','6']]
    for i,extra in enumerate(guards):
        p=subprocess.run(base+extra,capture_output=True,text=True);save(out/f'guard_{i}.json',dict(command=base+extra,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr));check('explicit rejection '+str(i),p.returncode!=0 and not p.stdout)
    check('calibration sources unchanged',all(digest(ROOT/n)==sha for n,sha in hashes.items()));save(out/'small_laws.json',trees);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),native_steps_replayed=steps,pilot_cases=19,rng_words=summary['rng_probe_words'],source_hashes=hashes,dependencies=dependencies,scope='Independent deterministic replay and exact finite-word model calibration; no accuracy conclusion from two-sample scaling pilots'));print(json.dumps(dict(checks=len(checks),steps=steps,trees=len(trees))))
if __name__=='__main__':main()
