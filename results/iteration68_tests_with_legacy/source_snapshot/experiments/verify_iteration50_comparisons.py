"""Hash-bound same-target comparisons; no use of invalid outer MPS timings."""
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
    out=ROOT/'results/iteration50_comparisons';out.mkdir(exist_ok=False)
    names=[f'experiments/{n}.py' for n in ('verify_iteration50_comparisons','iteration22_external','iteration33_run','verify_iteration34','iteration45_independent','iteration41_independent')]
    hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];mps=[];noise=[];native={}
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    def validation(base):
        v=read(base+'/validation.json');check(base+' passed',v['count']==v['passed']);return v
    def bound(n,v):
        r=read(n);check('bound '+n,deps[n]==v['dependencies'][n]);return r
    def arrays(n,v):
        deps[n]=digest(ROOT/n);check('bound '+n,deps[n]==v['dependencies'][n]);return np.load(ROOT/n)
    mv=validation('results/iteration50_mps_validation');nv=validation('results/iteration50_noise_validation')
    oldmv=validation('results/iteration46_mps_validation');oldnv=validation('results/iteration49_noise_validation');iv=validation('results/iteration49_integrated_recovery')
    notice=read('results/iteration50_mps/timing_notice.json');check('timing exclusion disclosed','seconds' in json.dumps(notice))
    for it in (46,47):
        v=validation(f'results/iteration{it}_native_validation');inp=validation(f'results/iteration{it}_input_validation')
        for c in range(4,8):
            target=bound(f'results/iteration{it}_inputs/case_{c:02d}.json',inp)['target']['target_intervals'];check('eight mode target',len(target)==256)
            for k in (2,3,4):
                base=f'results/iteration{it}_native/case_{c:02d}_k{k}';raw=bound(base+'.stdout.json',v);audit=bound(base+'.audit.json',v)
                law=layer_law({x['prefix']:x['threshold_hex'] for x in raw['prefix_tree']},10,'uint64');val=tv(law,target+[(F(0),F(0))]*768)
                check(base+' independent full padded law',law==audit['laws']['uint64']['distribution'] and val==tuple(audit['physical_TV_intervals']['uint64']) and sum(law)==1)
                native[(it,c,k)]=val
    plan=bound('results/iteration50_mps/plan.json',mv)
    for i,job in enumerate(plan['jobs']):
        rec=bound(f'results/iteration50_mps/case_{i:02d}.json',mv);it,c=job['source_round'],job['case'];val=tuple(rec['sampler_physical_TV_interval'])
        if it==46 and job['dimension']==3 and job['rank']==8:
            oi=19+4*(c-4);old=bound(f'results/iteration46_mps/case_{oi:02d}.json',oldmv)
            a=arrays(f'results/iteration50_mps/case_{i:02d}.npz',mv);b=arrays(f'results/iteration46_mps/case_{oi:02d}.npz',oldmv)
            check('old budget tensor/law regression '+str(i),a.files==b.files and all(np.array_equal(a[n],b[n]) for n in a.files) and rec['represented']==old['represented'] and rec['sampling']['word_law']==old['sampling']['word_law'] and val==tuple(old['sampler_physical_TV_interval']))
        comparisons={str(k):dict(native_TV=native[it,c,k],MPS_minus_native=(val[0]-native[it,c,k][1],val[1]-native[it,c,k][0]),MPS_strictly_better=val[1]<native[it,c,k][0]) for k in (2,3,4)}
        mps.append(dict(case=i,job=job,TV=val,native=comparisons,failure=rec['sampling']['word_law'][-1]))
    budget=[]
    for it in (46,47):
        for c in range(4,8):
            group=[r for r in mps if r['job']['source_round']==it and r['job']['case']==c];low=next(r for r in group if r['job']['dimension']==3 and r['job']['rank']==8);high=next(r for r in group if r['job']['dimension']==4 and r['job']['rank']==16)
            improvement=(low['TV'][0]-high['TV'][1],low['TV'][1]-high['TV'][0]);check('positive budget improvement '+str((it,c)),improvement[0]>0);budget.append(dict(source_round=it,case=c,low_TV=low['TV'],high_TV=high['TV'],improvement=improvement,improvement_factor_interval=(low['TV'][0]/high['TV'][1],low['TV'][1]/high['TV'][0])))
    for i in range(36):
        n=f'results/iteration50_noise/case_{i:02d}';o=f'results/iteration49_noise/case_{i:02d}';a=bound(n+'.json',nv);b=bound(o+'.json',oldnv);integ=bound(f'results/iteration49_integrated/case_{i:02d}.json',iv)
        ta=arrays(n+'.npz',nv);tb=arrays(o+'.npz',oldnv)
        check('same noise input/reference '+str(i),a['job']==b['job']==integ['job'] and a['continuous_target']==b['continuous_target'] and np.array_equal(ta['tensor_0'],tb['tensor_0']))
        newtv=tv(a['mixture_law'],a['continuous_target']+[(F(0),F(0))]);oldtv=tv(b['mixture_law'],b['continuous_target']+[(F(0),F(0))]);check('recomputed old/new TV '+str(i),newtv==tuple(a['full_Gaussian_TV']) and oldtv==tuple(b['full_Gaussian_TV']))
        diff=(newtv[0]-oldtv[1],newtv[1]-oldtv[0]);status='improved' if diff[1]<0 else 'regressed' if diff[0]>0 else 'unresolved'
        finite=integ['integral']['law'];ftv=tv(a['mixture_law'],finite+[(F(0),F(0))]);signed=(a['mixture_law'][0]-finite[0][1],a['mixture_law'][0]-finite[0][0])
        check('strict bound improvement '+str(i),a['discretization_TV_upper']<b['discretization_TV_upper'] and a['total_TV_upper']<b['total_TV_upper'])
        noise.append(dict(case=i,job=a['job'],old_TV=oldtv,new_TV=newtv,difference=diff,status=status,old_discretization_upper=b['discretization_TV_upper'],new_discretization_upper=a['discretization_TV_upper'],old_total_upper=b['total_TV_upper'],new_total_upper=a['total_TV_upper'],finite_continuous_TV=ftv,program_minus_finite_vacuum=signed,finite_minus_Gaussian_vacuum=integ['finite_continuous_vs_Gaussian_signed_vacuum_bias']))
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()))
    summary=dict(mps_cases=24,native_laws=24,old_budget_regressions=4,noise_cases=36,noise_improved=sum(r['status']=='improved' for r in noise),noise_regressed=sum(r['status']=='regressed' for r in noise),noise_unresolved=sum(r['status']=='unresolved' for r in noise),highest_budget_better_native={str(k):sum(r['native'][str(k)]['MPS_strictly_better'] for r in mps if r['job']['dimension']==4 and r['job']['rank']==16) for k in (2,3,4)},timing_scope='Exclude results/iteration50_mps/rows.json seconds; no end-to-end performance comparison',scope='Reused eight-mode targets and one-mode noise controls; not new holdout, not 144-mode upper bound')
    save(out/'mps.json',encode(mps));save(out/'budget.json',encode(budget));save(out/'noise.json',encode(noise));save(out/'summary.json',summary);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
