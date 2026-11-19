"""Independent preexecution covariance, input fidelity and all-axis budget audit."""
import json
from fractions import Fraction as F
from functools import lru_cache
import numpy as np
from iteration65_support import ROOT,snapshot,new_families,digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration48_independent import bisect_sqrt,independent_overlap
from iteration51_normal_independent import exp_negative
from verify_iteration58_coupling import cov
exp_negative=lru_cache(None)(exp_negative)

def main():
    out=ROOT/'results/iteration65_selection_validation';hashes=snapshot(out);deps={};checks=[]
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)));assert ok,n
    summary=read('results/iteration65_selection/summary.json');records=read('results/iteration65_selection/selection.json');jobs=read('results/iteration65_selection/jobs.json');ck('panel counts',summary['completed'] and len(records)==5 and len(jobs)==11)
    for n,s in summary['source_hashes'].items():ck('frozen '+n,digest(ROOT/n)==s==digest(ROOT/'results/iteration65_selection/source_snapshot'/n))
    m59=read('results/iteration59_manifest.json');v59=read('results/iteration59_noise_validation/validation.json');ck('old complete inputs',v59['count']==v59['passed'] and deps['results/iteration59_noise_validation/validation.json']==m59['artifacts']['results/iteration59_noise_validation/validation.json']);m58=read('results/iteration58_manifest.json');v58=read('results/iteration58_coupling_validation/validation.json');ck('old moment audit',v58['count']==v58['passed'] and deps['results/iteration58_coupling_validation/validation.json']==m58['artifacts']['results/iteration58_coupling_validation/validation.json']);ix=read('results/iteration58_coupling/axis_index.json');moments={}
    for item in ix:
        name=f"results/iteration58_coupling/axis_{item['index']:02d}.json";r=read(name);ck('moment bytes '+name,deps[name]==m58['artifacts'][name]==v58['dependencies'][name]);moments[(r['sigma'],r['cells'],r['radius'])]=r
    expected=[]
    for baseline in (1,3,5):
        name=f'results/iteration59_noise/case_{baseline:02d}.json';old=read(name);ck('old input bytes '+name,deps[name]==m59['artifacts'][name]==v59['dependencies'][name]);job=dict(old['job']);job.update(holdout=False,prior_holdout=job['holdout'],baseline_case=baseline);expected.append(job)
    expected+=decode(encode(new_families()));expected_jobs=[]
    for fi,(r,job) in enumerate(zip(records,expected)):
        tag=f'family{fi} ';ck(tag+'fixed input',r['job']==job and r['family_index']==fi)
        shape=[[F(24,25),-F(7,25)],[-F(7,25),-F(24,25)]] if job['rotated'] else [[F(0),F(1)],[F(1),F(0)]];br=[[job['pairing']*job['phase'][0]*x for x in row] for row in shape];bi=[[job['pairing']*job['phase'][1]*x for x in row] for row in shape];V=cov(br,bi);w=[[F(x) for x,y in d]+[F(y) for x,y in d] for d in job['directions']];u=[z[2:]+[-x for x in z[:2]] for z in w];G=[[sum(u[a][i]*V[i][j]*u[b][j] for i in range(4) for j in range(4)) for b in range(4)] for a in range(4)];ck(tag+'physical Gram',r['pairing_real']==br and r['pairing_imag']==bi and r['target']['generator_gram']==G and exact_determinant(G)>0)
        for d,item in zip((4,6),r['inputs']):
            name='results/iteration65_selection/'+item['tensor_path'];deps[name]=digest(ROOT/name);z=np.load(ROOT/name);t=[z['tensor_0'],z['tensor_1']];b=item['bridge'];ind=independent_overlap(t,br,bi);core=[[F(i==j)-F(([br[k]+bi[k] for k in range(2)]+[bi[k]+[-x for x in br[k]] for k in range(2)])[i][j]) for j in range(4)] for i in range(4)];det=exact_determinant(core);lo,hi=bisect_sqrt(det);a,bb=ind['overlap_squared_over_norm_interval'];fl,fh=b['full_fidelity_interval'];dl,dh=b['full_trace_interval'];ck(tag+f'd{d} fidelity',item['dimension']==d and b['physical']['covariance']==V and b['physical']['determinant']==det>0 and fl<=max(F(0),lo*a)<=min(F(1),hi*bb)<=fh and dl*dl<=1-fh<=1-fl<=dh*dh)
        all_candidates=[*r['candidates'],dict(axis=None,cells=[4]*4,nodes=1296,bound=r['coarse_bound'])]
        for ai,c in enumerate(all_candidates):
            cells=[16 if i==ai else 4 for i in range(4)] if ai<4 else [4]*4;axes=[dict(sigma=s,cells=N,radius={4:2,16:3}[N]) for s,N in zip(job['sigmas'],cells)];b=c['bound'];ck(tag+f'candidate{ai}',c['cells']==cells and b['axes']==axes and b['gram']==G and c['nodes']==(3888 if ai<4 else 1296));mm=[moments[(x['sigma'],x['cells'],x['radius'])] for x in axes];mean=tuple(sum(G[i][i]*m['squared_interval'][side] for i,m in enumerate(mm)) for side in (0,1));oldbound=min(F(1),sum(bisect_sqrt(G[i][i],192)[1]*((1-m['grid']['tail_interval'][0])*m['grid']['interior_half_width']+m['grid']['tail_mean_excess_interval'][1]) for i,m in enumerate(mm)));l1=min(F(1),sum(bisect_sqrt(G[i][i],192)[1]*m['absolute_interval'][1] for i,m in enumerate(mm)));rms=min(F(1),bisect_sqrt(mean[1],192)[1]);el,eh=exp_negative(mean[1]);lo,hi=b['exponent_at_mean_upper_interval'];ck(tag+f'exp{ai}',lo<=el<=eh<=hi);curve=min(F(1),bisect_sqrt(1-lo,192)[1]);values=dict(old_axis_upper=oldbound,exact_l1_upper=l1,rms_upper=rms,joint_curved_upper=curve);ck(tag+f'bounds{ai}',tuple(b['mean_quadratic_interval'])==mean and all(b[k]==v for k,v in values.items()) and b['best_upper']==min(values.values()));leading=[F(1)]+[exact_determinant([row[:k] for row in G[:k]]) for k in range(1,5)];ck(tag+f'PSD{ai}',all(v>0 for v in leading) and b['psd_pivots']==[leading[k+1]/leading[k] for k in range(4)])
        selected=min(range(4),key=lambda i:(r['candidates'][i]['bound']['best_upper'],i));ck(tag+'selection',r['selected_axis']==selected and r['selected_cells']==r['candidates'][selected]['cells']);proxy=dict(grid=min(F(1),r['inputs'][0]['bridge']['full_trace_interval'][1]+r['candidates'][selected]['bound']['best_upper']),input=min(F(1),r['inputs'][1]['bridge']['full_trace_interval'][1]+r['coarse_bound']['best_upper']));ck(tag+'frozen proxy',proxy==r['prediction_proxy'] and r['predicted_winner']==min(('grid','input'),key=lambda x:(proxy[x],x)))
        for action in (('joint',) if fi<3 else ('baseline','grid','input','joint')):expected_jobs.append(dict(job,family_index=fi,action=action,dimension=6 if action in ('input','joint') else 4,cells=r['selected_cells'] if action in ('grid','joint') else [4]*4,selected_axis=selected if action in ('grid','joint') else None))
    ck('exact jobs',jobs==expected_jobs);ck('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),families=5,new_holdouts=2,jobs=11,candidates=20,source_hashes=hashes,dependencies=deps));print('selection verified',len(checks),flush=True)
if __name__=='__main__':main()
