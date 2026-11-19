"""Separate ideal local projection, construction deviation and sampler error."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration45_independent import tv
from iteration41_independent import layer_law
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration52_comparisons';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration52_comparisons','iteration22_external','iteration33_run','verify_iteration34','iteration45_independent','iteration41_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[];native={};refs={};inputs={};records={};baselines={};timings=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    def valid(n):
        v=read(n+'/validation.json');check(n+' validated',v['count']==v['passed']);return v
    def bound(n,v):
        r=read(n);check('bound '+n,deps[n]==v['dependencies'][n]);return r
    def npz(n,v):deps[n]=digest(ROOT/n);check('bound '+n,deps[n]==v['dependencies'][n]);return np.load(ROOT/n)
    bv=valid('results/iteration52_box_validation');mv=valid('results/iteration52_mps_prefix_validation');pv=valid('results/iteration52_prefix_density_validation');ov=valid('results/iteration50_mps_validation')
    for i in range(8):
        r=bound(f'results/iteration52_box/case_{i:02d}.json',bv);j=r['job'];refs[j['source_round'],j['case'],j['dimension']]=r
    reverse=lambda mask:int(f'{mask:08b}'[::-1],2)
    for it in (46,47):
        v=valid(f'results/iteration{it}_input_validation');nv=valid(f'results/iteration{it}_native_validation')
        for c in range(4,8):
            r=bound(f'results/iteration{it}_inputs/case_{c:02d}.json',v);inputs[it,c]=r
            if c%2:
                forward=inputs[it,c-1];check('exact reversal '+str((it,c)),all(r['physical'][name]==[list(reversed(row)) for row in reversed(forward['physical'][name])] for name in ('pairing_real','pairing_imag')) and r['target']['target_intervals']==[forward['target']['target_intervals'][reverse(m)] for m in range(256)])
            for k in (2,3,4):
                base=f'results/iteration{it}_native/case_{c:02d}_k{k}';raw=bound(base+'.stdout.json',nv);audit=bound(base+'.audit.json',nv);law=layer_law({x['prefix']:x['threshold_hex'] for x in raw['prefix_tree']},10,'uint64');value=tv(law,r['target']['target_intervals']+[(F(0),F(0))]*768);check('full official law '+str((it,c,k)),law==audit['laws']['uint64']['distribution'] and value==tuple(audit['physical_TV_intervals']['uint64']));native[it,c,k]=value
    for it,v in ((50,ov),(52,mv)):
        plan=bound(f'results/iteration{it}_mps/plan.json',v)
        for i,job in enumerate(plan['jobs']):
            rec=bound(f'results/iteration{it}_mps/case_{i:02d}.json',v);key=(job['source_round'],job['case'],job['dimension'],job['rank']);source=inputs[key[:2]];reference=refs[key[0],key[1]-key[1]%2,key[2]];law=reference['reference']['distribution'];law=[law[reverse(m)] for m in range(256)] if key[1]%2 else law;state=rec['represented']['distribution'];word=rec['sampling']['word_law'];target=source['target']['target_intervals'];stateTV=tv(state,target);wordTV=tv(word,target+[(F(0),F(0))]);boxTV=tv(law,target);delta=sum(abs(a-b) for a,b in zip(word,law+[F(0)]))/2;construction=sum(abs(a-b) for a,b in zip(state,law))/2;check('physical and projection comparison '+str((it,i)),stateTV==tuple(rec['state_physical_TV_interval']) and wordTV==tuple(rec['sampler_physical_TV_interval']) and boxTV==tuple(reference['physical_TV']) and wordTV[0]>=max(F(0),boxTV[0]-delta) and wordTV[1]<=boxTV[1]+delta)
            row=dict(run_round=it,case=i,job=job,TV=wordTV,state_TV=stateTV,ideal_box_TV=boxTV,state_vs_box_TV=construction,program_vs_box_TV=delta,projection_probability=reference['reference']['projection_probability_interval'],tail_trace=reference['reference']['tail_trace_interval'],native={str(k):native[key[0],key[1],k] for k in (2,3,4)},arithmetic=rec['arithmetic_TV']);rows.append(row);records[it,i]=rec
            if it==50:baselines[key]=(i,row)
            elif key in baselines:
                oi,oldrow=baselines[key];old=records[50,oi];a=npz(f'results/iteration52_mps/case_{i:02d}.npz',mv);b=npz(f'results/iteration50_mps/case_{oi:02d}.npz',ov);check('reconstructed budget regression '+str(i),a.files==b.files and all(np.array_equal(a[n],b[n]) for n in a.files) and rec['represented']==old['represented'] and word==old['sampling']['word_law'])
    changes=[]
    for r in rows:
        j=r['job'];key=(j['source_round'],j['case'],j['dimension'],16)
        if r['run_round']!=52 or j['rank']!=32 or key not in baselines:continue
        oi,old=baselines[key];diff=(r['TV'][0]-old['TV'][1],r['TV'][1]-old['TV'][0]);changes.append(dict(job=j,old_case=oi,new_case=r['case'],old_TV=old['TV'],new_TV=r['TV'],difference=diff,actual_improved=diff[1]<0,actual_regressed=diff[0]>0,old_vs_box=old['state_vs_box_TV'],new_vs_box=r['state_vs_box_TV'],projection_approach_improved=r['state_vs_box_TV']<old['state_vs_box_TV']))
    rawtiming=bound('results/iteration52_mps/rows.json',mv);raw_bytes=compressed_bytes=0
    for i,row in enumerate(rawtiming):
        rec=records[52,i];storage=rec['sampling']['trace_storage'];raw_bytes+=sum(s['raw_bytes'] for s in storage);compressed_bytes+=sum(s['compressed_bytes'] for s in storage);timings.append(dict(case=i,job=rec['job'],seconds=row['seconds'],phase_seconds=row['phase_seconds'],constructor_internal_seconds=rec['construction']['total_seconds'],stored_complex_elements=rec['construction']['stored_complex_elements'],max_bond_actual=rec['construction']['max_bond_actual']))
    best=[r for r in rows if r['run_round']==52 and r['job']['dimension']==4 and r['job']['rank']==32];summary=dict(new_runs=20,old_runs=24,unique_budget_configs=40,rebuilt_regressions=4,ideal_references=8,official_laws=24,bond16_to32_pairs=len(changes),physical_improved=sum(x['actual_improved'] for x in changes),physical_regressed=sum(x['actual_regressed'] for x in changes),projection_approach_improved=sum(x['projection_approach_improved'] for x in changes),highest_budget_better_native={str(k):sum(r['TV'][1]<r['native'][str(k)][0] for r in best) for k in (2,3,4)},trace_raw_bytes=raw_bytes,trace_compressed_bytes=compressed_bytes,scope='Reused targets; ideal local projection is a reference, not a lower bound for all finite-dimensional states. Timings are single-run observations with concurrent work, not matched-quality benchmarks.')
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'changes.json',encode(changes));save(out/'timings.json',timings);save(out/'summary.json',summary);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
