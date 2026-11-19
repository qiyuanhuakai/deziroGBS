"""Choose grid axes and freeze resource prediction before complete programs."""
import json,sys
from fractions import Fraction as F
import numpy as np
from iteration65_support import ROOT,snapshot,prepare,new_families,digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.joint_displacement_coupling import joint_coupling

def main():
    out=ROOT/'results/iteration65_selection';hashes=snapshot(out);deps={}
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_bytes()))
    m=read('results/iteration59_manifest.json');v=read('results/iteration59_noise_validation/validation.json');assert v['count']==v['passed'] and deps['results/iteration59_noise_validation/validation.json']==m['artifacts']['results/iteration59_noise_validation/validation.json']
    families=[]
    for baseline in (1,3,5):
        n=f'results/iteration59_noise/case_{baseline:02d}.json';old=read(n);assert deps[n]==m['artifacts'][n]==v['dependencies'][n]
        j=dict(old['job']);j.update(holdout=False,prior_holdout=j['holdout'],baseline_case=baseline);families.append(j)
    families+=new_families();records=[];jobs=[]
    for fi,job in enumerate(families):
        inputs=[]
        for d in (4,6):
            br,bi,t,b,target,reference,s=prepare(job,d,continuous=False);name=f'family_{fi:02d}_d{d}.npz';np.savez_compressed(out/name,tensor_0=t[0],tensor_1=t[1]);inputs.append(dict(dimension=d,bridge=b,tensor_path=name))
        candidates=[]
        for axis in range(4):
            cells=[16 if i==axis else 4 for i in range(4)];bound=joint_coupling(target['generator_gram'],[dict(sigma=s,cells=N,radius={4:2,16:3}[N]) for s,N in zip(job['sigmas'],cells)]);candidates.append(dict(axis=axis,cells=cells,nodes=3888,bound=bound))
        selected=min(candidates,key=lambda x:(x['bound']['best_upper'],x['axis']));coarse=joint_coupling(target['generator_gram'],[dict(sigma=s,cells=4,radius=2) for s in job['sigmas']]);proxy=dict(grid=min(F(1),inputs[0]['bridge']['full_trace_interval'][1]+selected['bound']['best_upper']),input=min(F(1),inputs[1]['bridge']['full_trace_interval'][1]+coarse['best_upper']));winner=min(('grid','input'),key=lambda x:(proxy[x],x))
        records.append(dict(family_index=fi,job=job,pairing_real=br,pairing_imag=bi,target=target,inputs=inputs,candidates=candidates,coarse_bound=coarse,selected_axis=selected['axis'],selected_cells=selected['cells'],prediction_proxy=proxy,predicted_winner=winner))
        actions=('joint',) if fi<3 else ('baseline','grid','input','joint')
        for action in actions:jobs.append(dict(job,family_index=fi,action=action,dimension=6 if action in ('input','joint') else 4,cells=selected['cells'] if action in ('grid','joint') else [4]*4,selected_axis=selected['axis'] if action in ('grid','joint') else None))
    save(out/'selection.json',encode(records));save(out/'jobs.json',encode(jobs));assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(completed=True,families=5,new_holdouts=2,candidates=20,new_jobs=11,new_nodes=32400,source_hashes=hashes,dependencies=deps,unchanged_sources=True,unchanged_dependencies=True));print('frozen axes',[r['selected_axis'] for r in records],flush=True)
if __name__=='__main__':main()
