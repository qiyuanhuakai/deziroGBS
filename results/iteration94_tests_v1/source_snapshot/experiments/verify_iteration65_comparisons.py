"""Complete factorial resource comparison with explicit reused versus new inputs."""
import json,math
from fractions import Fraction as F
from iteration65_support import ROOT,snapshot,digest,save
from iteration33_run import encode
from verify_iteration34 import decode

def main():
    out=ROOT/'results/iteration65_comparisons';hashes=snapshot(out);deps={};checks=[]
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)));assert ok,n
    valid=read('results/iteration65_noise_validation/validation.json');controls=read('results/iteration65_controls/validation.json');selectionv=read('results/iteration65_selection_validation/validation.json');selection=read('results/iteration65_selection/selection.json');timings=read('results/iteration65_noise/rows.json');verified=read('results/iteration65_noise_validation/rows.json');ck('new complete',all(x['count']==x['passed'] for x in (valid,controls,selectionv)) and valid['nodes']==32400 and valid['cases']==11)
    old={}
    for it in (59,60):
        m=read(f'results/iteration{it}_manifest.json');v=read(f'results/iteration{it}_noise_validation/validation.json');ck('old certified '+str(it),v['count']==v['passed'] and deps[f'results/iteration{it}_noise_validation/validation.json']==m['artifacts'][f'results/iteration{it}_noise_validation/validation.json']);old[it]=(m,v)
    def prior(it,index):
        name=f'results/iteration{it}_noise/case_{index:02d}.json';r=read(name);m,v=old[it];ck('old bytes '+name,deps[name]==m['artifacts'][name]==v['dependencies'][name]);return r
    def fresh(index):
        name=f'results/iteration65_noise/case_{index:02d}.json';r=read(name);ck('new bytes '+name,deps[name]==valid['dependencies'][name]);vr=verified[index];ck('verified metrics '+str(index),vr['actual_TV']==r['full_Gaussian_TV'] and vr['total_upper']==r['total_TV_upper']);return r
    panels=[];new_records=[]
    for fi,s in enumerate(selection):
        if fi<3:
            variants=dict(baseline=prior(59,2*fi+1),grid=prior(60,2*fi),input=prior(60,2*fi+1),joint=fresh(fi));new_records.append(variants['joint'])
        else:
            first=3+4*(fi-3);variants={a:fresh(first+j) for j,a in enumerate(('baseline','grid','input','joint'))};new_records.extend(variants.values())
        tag=s['job']['family']+' ';baseline=variants['baseline'];display={}
        for action,r in variants.items():
            ck(tag+action+' same target',r['target']==baseline['target']==s['target']);ck(tag+action+' complete law',sum(r['mixture_law'])==1 and r['mixture_law'][-1]==0 and r['full_Gaussian_TV'][1]<=r['total_TV_upper']);d=r['job']['dimension'];Ns=r['job']['cells'];ck(tag+action+' resource identity',d==(4 if action in ('baseline','grid') else 6) and Ns==([4]*4 if action in ('baseline','input') else s['selected_cells']) and r['storage']['nodes']==math.prod(x+2 for x in Ns))
            actual=r['full_Gaussian_TV'];finite=r['program_vs_finite_continuous_TV'];reference=r['finite_continuous_vs_Gaussian_TV'];slack=(max(F(0),finite[0]+reference[0]-actual[1]),finite[1]+reference[1]-actual[0]);ck(tag+action+' error triangle',0<=slack[0]<=slack[1]);display[action]=dict(dimension=d,cells=Ns,nodes=r['storage']['nodes'],fock_box_states=d*d,tensor_complex_entries=2*d*d,trace_nodes=r['storage']['trace_nodes'],actual_TV=actual,finite_continuous_TV=finite,reference_TV=reference,ideal_grid_TV=r['ideal_grid_TV'],state_upper=r['bridge']['full_trace_interval'][1],coupling_upper=r['coupling']['best_upper'],total_upper=r['total_TV_upper'],triangle_slack=slack)
        for left,right in [('baseline','grid'),('input','joint')]:ck(tag+left+right+' same finite input',variants[left]['bridge']==variants[right]['bridge'] and variants[left]['finite_continuous']==variants[right]['finite_continuous'])
        for left,right in [('baseline','input'),('grid','joint')]:ck(tag+left+right+' same discretization',all(variants[left][k]==variants[right][k] for k in ('grids','selector_weights','coupling','ideal_grid_law')))
        grid,inp,joint=(variants[a]['full_Gaussian_TV'] for a in ('grid','input','joint'));winner='grid' if grid[1]<inp[0] else 'input' if inp[1]<grid[0] else 'unresolved';boundwinner=min(('grid','input'),key=lambda a:(variants[a]['total_TV_upper'],a));pred=s['predicted_winner'];edges=[]
        for left,right in [('baseline','grid'),('baseline','input'),('baseline','joint'),('grid','joint'),('input','joint')]:
            a,b=(variants[x]['full_Gaussian_TV'] for x in (left,right));edges.append(dict(source=left,destination=right,improved=b[1]<a[0],worsened=b[0]>a[1],factor=(a[0]/b[1],a[1]/b[0])))
        panels.append(dict(family_index=fi,family=s['job']['family'],new_holdout=fi>=3,selected_axis=s['selected_axis'],prediction_proxy=s['prediction_proxy'],predicted_winner=pred,actual_winner=winner,full_bound_winner=boundwinner,prediction_agrees=pred==winner,full_bound_agrees=boundwinner==winner,joint_beats_both=joint[1]<min(grid[0],inp[0]),joint_worse_than_best=joint[0]>min(grid[1],inp[1]),variants=display,edges=edges))
    ck('new run coverage',len(new_records)==11 and sum(r['storage']['nodes'] for r in new_records)==32400 and sum(r['storage']['trace_nodes'] for r in new_records)==valid['trace_nodes']);ck('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'panels.json',encode(panels));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(x['passed'] for x in checks),source_hashes=hashes,dependencies=deps));save(out/'summary.json',dict(completed=True,new_cases=11,reused_cases=9,physical_inputs=5,new_holdouts=2,nodes=valid['nodes'],trace_nodes=valid['trace_nodes'],public_samples=77,joint_beats_both=sum(p['joint_beats_both'] for p in panels),joint_worse_than_best=sum(p['joint_worse_than_best'] for p in panels),new_holdout_prediction_agrees=sum(p['prediction_agrees'] for p in panels if p['new_holdout']),raw_bytes=sum(r['storage']['raw_bytes'] for r in new_records),compressed_bytes=sum(r['storage']['compressed_bytes'] for r in new_records),observed_production_seconds=sum(r['seconds'] for r in timings),scope='Three old-input joint extensions and two preregistered physical holdouts,each with four resource variants. Independent53-bit word model. Different costs,not matched-quality throughput; no universal monotonicity or predictive generalization theorem.'));print('joint comparisons verified',len(checks),flush=True)
if __name__=='__main__':main()
