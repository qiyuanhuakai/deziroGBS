"""Explore an existing saturated path with certified real-recurrence intervals."""
import sys,json,shutil,subprocess,time
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.set_int_max_str_digits(0)
from iteration22_external import digest,save
from iteration33_run import encode
from iteration40_k2_reference import forward


def main():
    out=ROOT/'results/iteration40_saturation';out.mkdir(exist_ok=False);names=['experiments/iteration40_saturation.py','experiments/iteration40_k2_reference.py','experiments/iteration40_target.cpp','experiments/iteration40_target_build.py','experiments/iteration22_external.py','experiments/iteration33_run.py'];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    manifest=json.loads((ROOT/'results/iteration39_manifest.json').read_text())
    def old(n):
        p=ROOT/n;sha=digest(p);assert sha==manifest['artifacts'][n];deps[n]=sha;return p
    exe=ROOT/'results/iteration40_target_build/frugal_target_dump.exe';build=json.loads((exe.parent/'build.json').read_text());assert digest(exe)==build['executable_sha256'];deps[exe.relative_to(ROOT).as_posix()]=digest(exe);deps['results/iteration40_target_build/build.json']=digest(exe.parent/'build.json');raw=json.loads(old('results/iteration39_global/case_01.stdout.json').read_text());candidates=[(j,i) for i,path in enumerate(raw['audit_paths']) for j,a in enumerate(path) if float.fromhex(a[1])>1];mode,sample=min(candidates);prefix=raw['outputs'][sample][:mode+1];selection=dict(rule='earliest mode with q>1, then smallest sample index in completed round39 K=2 cohort',mode=mode,sample=sample,prefix=prefix,exploratory_only=True);save(out/'selection.json',selection);targets={}
    for label,n,csv in (('local',10,'results/iteration39_local/case_01.csv'),('global',144,'results/iteration39_global/case_01.csv')):
        command=[str(exe),str(old(csv)),str(n)]
        with (out/f'{label}_target.json').open('w') as stdout,(out/f'{label}_target.stderr.log').open('w') as stderr:r=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600)
        assert r.returncode==0;targets[label]=json.loads((out/f'{label}_target.json').read_text())
    assert targets['local']['first_hex']==targets['global']['first_hex'][:10] and targets['local']['pairs']==[r for r in targets['global']['pairs'] if r[1]<10]
    local=json.loads(old('results/iteration39_local/case_01.stdout.json').read_text());table={r['prefix']:F(float.fromhex(r['threshold_hex'])) for r in local['prefix_tree']};paths=[local['outputs'][i] for i in (0,1,2,63,128,256)]+['0'*10,'1'*10];save(out/'plan.json',dict(source_hashes=hashes,dependencies=deps,selection=selection,calibration_paths=paths,bits=384,reference_scope='Exact-real version of the official approximate K=2 marginal recurrence at fixed exported binary64 coefficients; NOT physical conditional probabilities'));calibration=[]
    for i,prefix10 in enumerate(paths):
        result=forward(targets['local'],prefix10);differences=[max(abs(q-table[prefix10[:j]]) for q in r['threshold']) for j,r in enumerate(result['records'])];assert max(differences)<F(1,10**11);assert all(r['threshold'][1]-r['threshold'][0]<F(1,10**60) for r in result['records']);calibration.append(dict(prefix=prefix10,max_native_difference=max(differences),result=result))
    save(out/'calibration.json',encode(calibration));begin=time.perf_counter();result=forward(targets['global'],prefix,progress=lambda depth,ops:print('certifying prefix',depth,'operations',ops,flush=True));checks=[]
    for j,r in enumerate(result['records']):
        u=F(float.fromhex(raw['audit_paths'][sample][j][0]));q=F(float.fromhex(raw['audit_paths'][sample][j][1]));lo,hi=r['threshold'];bit=prefix[j];compatible=u>hi if bit=='1' else u<=lo;assert compatible;checks.append(dict(mode=j,native_threshold=q,native_uniform=u,selected_bit=bit,real_interval=(lo,hi),certified_same_branch=compatible,native_difference_upper=max(abs(lo-q),abs(hi-q))))
    last=result['records'][-1]['threshold'];assert last[0]>1;record=dict(selection=selection,reference=result,trajectory_checks=checks,final_threshold=last,strict_excess_lower=last[0]-1,seconds=time.perf_counter()-begin,all_selected_branches_certified=True,scope='Exploratory deterministic path counterexample: saturation persists in exact-real approximate recurrence, not a full-distribution TV certificate');save(out/'proof.json',encode(record));assert hashes=={n:digest(ROOT/n) for n in names} and all(digest(ROOT/n)==sha for n,sha in deps.items());save(out/'summary.json',dict(source_hashes=hashes,dependencies=deps,unchanged_sources=True,calibration_paths=len(paths),certified_steps=len(checks),mode=mode,sample=sample,final_threshold_lower=float(last[0]),final_threshold_upper=float(last[1]),interval_width=float(last[1]-last[0]),strict_excess_lower=float(last[0]-1),operations=result['operations'],seconds=record['seconds'],exploratory_only=True));print('saturation certified',float(last[0]),float(last[1]),flush=True)
if __name__=='__main__':main()
