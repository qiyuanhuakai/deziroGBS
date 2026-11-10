"""All fixed resource comparisons,including unchanged or worsened errors."""
import json
from iteration66_support import ROOT,snapshot,save,digest
from verify_iteration34 import decode

def main():
    out=ROOT/'results/iteration66_comparisons';hashes=snapshot(out);deps={};checks=[]
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    def ck(n,ok):checks.append(dict(name=n,passed=bool(ok)));assert ok,n
    gv=read('results/iteration66_gsvd_validation/validation.json');pv=read('results/iteration66_program_validation/validation.json');gr=read('results/iteration66_gsvd_validation/rows.json');pr=read('results/iteration66_program_validation/rows.json');ck('complete validation',gv['count']==gv['passed'] and pv['count']==pv['passed'] and gv['constructions']==20 and pv['cases']==60);panels=[]
    for fi in range(5):
        for displacement in range(3):
            variants={};reference=None
            for ai,action in enumerate(('baseline','rank','dimension','joint')):
                ci=4*fi+ai;index=3*ci+displacement;r=read(f'results/iteration66_program/case_{index:02d}.json');g=read(f'results/iteration66_gsvd/case_{ci:02d}.json');ck(f'bound inputs {index}',deps[f'results/iteration66_program/case_{index:02d}.json']==pv['dependencies'][f'results/iteration66_program/case_{index:02d}.json'] and deps[f'results/iteration66_gsvd/case_{ci:02d}.json']==gv['dependencies'][f'results/iteration66_gsvd/case_{ci:02d}.json']);ck(f'metrics {index}',pr[index]['actual_TV']==r['full_Gaussian_TV'] and gr[ci]['state_trace']==g['bridge']['full_trace_interval']);reference=r['gaussian_target'] if reference is None else reference;ck(f'same target {index}',r['gaussian_target']==reference and r['job']['family_index']==fi and r['job']['action']==action and r['job']['displacement']==displacement);variants[action]=dict(construction=ci,program=index,dimension=g['job']['dimension'],rank=g['job']['rank'],actual_bond=g['info']['max_bond_actual'],elements=g['info']['stored_complex_elements'],trace_nodes=r['sampling']['nodes'],state_trace=g['bridge']['full_trace_interval'],ideal_float_budget=g['info']['construction_trace_budget'],actual_TV=r['full_Gaussian_TV'],arithmetic_TV=r['arithmetic_TV'],total_upper=r['total_TV_upper'],failure=r['sampling']['word_law'][-1])
            edges=[]
            for a,b in [('baseline','rank'),('baseline','dimension'),('baseline','joint'),('rank','joint'),('dimension','joint')]:
                x,y=variants[a]['actual_TV'],variants[b]['actual_TV'];edges.append(dict(source=a,destination=b,improved=y[1]<x[0],worsened=y[0]>x[1],overlap=not(y[1]<x[0] or y[0]>x[1]),factor=None if y[0]==0 else (x[0]/y[1],x[1]/y[0])))
            joint=variants['joint']['actual_TV'];best=min(variants[a]['actual_TV'][0] for a in ('rank','dimension'));panels.append(dict(family=g['job']['family'],family_index=fi,new_holdout=g['job']['new_holdout'],displacement=displacement,variants=variants,edges=edges,joint_beats_both=joint[1]<best,joint_worse_than_best=joint[0]>min(variants[a]['actual_TV'][1] for a in ('rank','dimension'))))
    ck('coverage',len(panels)==15 and sum(r['nodes'] for r in pr)==pv['nodes']);ck('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));ck('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'panels.json',panels);save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),source_hashes=hashes,dependencies=deps));save(out/'summary.json',dict(completed=True,physical_inputs=5,new_holdouts=1,constructions=20,programs=60,panels=15,public_samples=420,trace_nodes=pv['nodes'],joint_beats_both=sum(p['joint_beats_both'] for p in panels),joint_worse_than_best=sum(p['joint_worse_than_best'] for p in panels),baseline_joint_improved=sum(p['edges'][2]['improved'] for p in panels),scope='Four exploratory-input followups and one new physical holdout. Four actual GSVD budgets,three fixed displacements,costs differ. No continuous-noise integration or large-system throughput conclusion.'));print('all resource panels verified',len(checks),flush=True)
if __name__=='__main__':main()
