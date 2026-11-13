"""Independent exact arithmetic audit of all candidate guarantees and policy."""
import sys,json,shutil
from pathlib import Path
from fractions import Fraction as F
from functools import lru_cache
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration48_independent import bisect_sqrt
from iteration51_normal_independent import exp_negative
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0);exp_negative=lru_cache(None)(exp_negative)
def main():
    out=ROOT/'results/iteration60_selection_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration60_selection','iteration22_external','iteration33_run','verify_iteration34','iteration48_independent','iteration51_normal_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)));assert ok,n
    summary=read('results/iteration60_selection/summary.json');records=read('results/iteration60_selection/selection.json');jobs=read('results/iteration60_selection/jobs.json');ck('complete selection',summary['completed'] and len(records)==3 and len(jobs)==6)
    for n,s in summary['source_hashes'].items():ck('frozen selection '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration60_selection/source_snapshot'/n))
    m59=read('results/iteration59_manifest.json');v59=read('results/iteration59_noise_validation/validation.json');ck('old programs certified',v59['count']==v59['passed'] and deps['results/iteration59_noise_validation/validation.json']==m59['artifacts']['results/iteration59_noise_validation/validation.json']);m58=read('results/iteration58_manifest.json');v58=read('results/iteration58_coupling_validation/validation.json');ck('old moments certified',v58['count']==v58['passed'] and deps['results/iteration58_coupling_validation/validation.json']==m58['artifacts']['results/iteration58_coupling_validation/validation.json']);ix=read('results/iteration58_coupling/axis_index.json');moments={}
    for item in ix:
        n=f"results/iteration58_coupling/axis_{item['index']:02d}.json";r=read(n);ck('moment bytes '+n,deps[n]==m58['artifacts'][n]==v58['dependencies'][n]);moments[(r['sigma'],r['cells'],r['radius'])]=r
    for fi,(r,baseline) in enumerate(zip(records,(1,3,5))):
        n=f'results/iteration59_noise/case_{baseline:02d}.json';old=read(n);ck('baseline bytes '+str(baseline),deps[n]==m59['artifacts'][n]==v59['dependencies'][n]);G=old['target']['generator_gram'];job=old['job'];ck('baseline spec '+str(fi),r['baseline_case']==baseline and r['family']==job['family'] and len(r['candidates'])==4)
        for axis,c in enumerate(r['candidates']):
            tag=f'{fi}:{axis} ';cells=[16 if i==axis else 4 for i in range(4)];aa=[dict(sigma=s,cells=N,radius={4:2,16:3}[N]) for s,N in zip(job['sigmas'],cells)];b=c['bound'];ck(tag+'candidate spec',c['axis']==axis and c['cells']==cells and c['nodes']==3888 and b['axes']==aa and b['gram']==G);mm=[moments[(a['sigma'],a['cells'],a['radius'])] for a in aa];mean=tuple(sum(G[i][i]*m['squared_interval'][j] for i,m in enumerate(mm)) for j in (0,1));oldbound=min(F(1),sum(bisect_sqrt(G[i][i],192)[1]*((1-m['grid']['tail_interval'][0])*m['grid']['interior_half_width']+m['grid']['tail_mean_excess_interval'][1]) for i,m in enumerate(mm)));l1=min(F(1),sum(bisect_sqrt(G[i][i],192)[1]*m['absolute_interval'][1] for i,m in enumerate(mm)));rms=min(F(1),bisect_sqrt(mean[1],192)[1]);el,eh=exp_negative(mean[1]);lo,hi=b['exponent_at_mean_upper_interval'];ck(tag+'independent exponential',lo<=el<=eh<=hi);curve=min(F(1),bisect_sqrt(1-lo,192)[1]);values=dict(old_axis_upper=oldbound,exact_l1_upper=l1,rms_upper=rms,joint_curved_upper=curve)
            ck(tag+'mean and bounds',tuple(b['mean_quadratic_interval'])==mean and all(b[k]==v for k,v in values.items()) and b['best_upper']==min(values.values()));leading=[F(1)]+[exact_determinant([row[:k] for row in G[:k]]) for k in range(1,5)];ck(tag+'PSD pivots',all(d>0 for d in leading) and b['psd_pivots']==[leading[k+1]/leading[k] for k in range(4)])
        chosen=min(range(4),key=lambda i:(r['candidates'][i]['bound']['best_upper'],i));ck(str(fi)+'minimum and deterministic tie',r['selected_axis']==chosen and r['selected_cells']==r['candidates'][chosen]['cells'] and r['selection_upper']==r['candidates'][chosen]['bound']['best_upper']);common=dict(job);common.update(holdout=False,baseline_case=baseline,prior_holdout=job['holdout']);expected=[dict(common,action='grid',dimension=4,cells=r['selected_cells'],selected_axis=chosen),dict(common,action='input',dimension=6,cells=[4]*4,selected_axis=None)];ck(str(fi)+'fixed two actions',jobs[2*fi:2*fi+2]==expected)
    ck('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),baselines=3,candidates=12,jobs=6,source_hashes=hashes,dependencies=deps,scope='All axis candidates audited before any iteration60 complete program. Selection minimizes certified bound,not unknown actual TV.'));print('selection verified',len(checks),flush=True)
if __name__=='__main__':main()
