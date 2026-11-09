import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,json,shutil,sys,traceback
from pathlib import Path
from time import perf_counter
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from gbslab.threshold_blocks import contiguous_density,subset_effects,finite_block_law,gaussian_block_law,prefix_marginal,distance_diagnostics
from gbslab.displaced_vacuum import CoherentMPS
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel
from iteration23_support import digest,save,load_mps,mps_fingerprint
from iteration22_global import load_family
from iteration22_external import ARCHIVE_SHA

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['experiments/iteration32_run.py','experiments/iteration23_support.py','experiments/iteration22_global.py','experiments/iteration22_external.py',*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(ROOT/'research/iteration32_panel_plan.md',out/'frozen_plan.md');dependencies={}
    def require(path,iteration):
        sha=digest(ROOT/path);manifest=json.loads((ROOT/f'results/iteration{iteration}_manifest.json').read_text());assert sha==manifest['artifacts'][path];dependencies[path]=sha;return ROOT/path
    F=np.load(require('results/iteration28_panel/case_14.npz',28))['full_noise_factor'];V=np.load(require('results/iteration31_panel/large_input.npz',31))['covariance'];library=require('results/iteration31_build/libwire_forest.dll',31);require('results/iteration31_build/build.json',31)
    for j in range(16,20):require(f'results/iteration25_panel_retry/run_{j}.state.npz',25)
    assert digest(a.archive)==ARCHIVE_SHA
    families=('new_32','new_64','new_128','new_200','author_pco','author_hafnian');starts=(0,18,54,140);forest_starts=(0,18,54,136)
    jobs=[dict(panel='mps',family=family,start=start,length=4) for family in families for start in starts]+[dict(panel='forest',start=start,length=8,budget=budget,forest=forest) for start in forest_starts for budget in (0,2,4) for forest in ('chain','static_mi_tree')]
    save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=dependencies,archive_sha256=ARCHIVE_SHA))
    block_data={}
    for start in starts:
        indices=list(range(start,start+4))+list(range(start+144,start+148));noise=F[indices];cov=V[np.ix_(indices,indices)];effects=subset_effects(noise,(4,)*4);target,vacua=gaussian_block_law(cov);block_data[start]=(effects,target)
        np.savez_compressed(out/f'block_{start:03d}.npz',noise=noise,covariance=cov,target=target,target_vacua=vacua,**{f'effect_{mask:02d}':E for mask,E in effects.items()})
    rows=[];evaluator=None;loaded_family=None;fullmodel=ThresholdModel(V)
    for i,job in enumerate(jobs):
        begin=perf_counter();row=dict(sequence=i,**job);print('starting',i,job,flush=True)
        try:
            if job['panel']=='mps':
                if loaded_family!=job['family']:
                    evaluator=None;family=job['family'];j=families.index(family)
                    if j<4:tensors=load_mps(ROOT/f'results/iteration25_panel_retry/run_{16+j}.state.npz')
                    else:
                        tensors,entries=load_family(family,ROOT/'results/iteration22_external',a.archive);save(out/(family+'_entries.json'),entries)
                    fingerprint=mps_fingerprint(tensors);evaluator=CoherentMPS(tensors);loaded_family=family;del tensors;row['family_preparation_seconds']=perf_counter()-begin
                rho,dims=contiguous_density(evaluator,job['start'],4);effects,target=block_data[job['start']];law,vacua=finite_block_law(rho,dims,effects);metrics=distance_diagnostics(target,law);curves=[distance_diagnostics(prefix_marginal(target,k),prefix_marginal(law,k))['tv'] for k in range(1,5)]
                np.savez_compressed(out/f'case_{i:02d}.npz',density=rho,law=law,vacua=vacua,target=target,prefix_tv=np.array(curves));row.update(distances=metrics,prefix_tv=curves,raw_mps_fingerprint=fingerprint)
            else:
                evaluator=None;start=job['start'];modes=tuple(range(start,start+8));indices=list(modes)+[j+144 for j in modes];cov=V[np.ix_(indices,indices)];target,vacua=gaussian_block_law(cov);submodel=ThresholdModel(cov);order=modes+tuple(j for j in range(144) if j not in modes)
                with WireForestSampler(submodel,job['budget'],forest=job['forest'],selection='recent',library=library) as sampler:
                    enumeration_start=perf_counter();result=sampler.enumerate_law();row['enumeration_seconds']=perf_counter()-enumeration_start;assert result['failure_mass']==0;law=result['distribution'];conditionals=np.array([sampler.conditional_click_probability(tuple((mask>>j)&1 for j in range(length))) for length in range(8) for mask in range(1<<length)])
                    checks=[]
                    with WireForestSampler(fullmodel,job['budget'],forest=job['forest'],selection='recent',order=order,library=library) as full:
                        for prefix in ((),(1,0,1),(1,)*7,(0,1,0,1,1,0,1)):
                            small=sampler.conditional_details(prefix);large=full.conditional_details(prefix);assert tuple(modes[j] for j in small['retained_clicks'])==large['retained_clicks'];assert tuple(tuple(modes[j] for j in e) for e in small['forest_edges'])==large['forest_edges'];error=abs(small['click_probability']-large['click_probability']);assert error<2e-11
                            checks.append(dict(prefix=prefix,small_probability=small['click_probability'],full_probability=large['click_probability'],absolute_error=error,full_retained=large['retained_clicks'],full_edges=large['forest_edges']))
                    row.update(full_prefix_checks=checks,distances=distance_diagnostics(target,law),native_diagnostics=sampler.kernel.cache_info())
                np.savez_compressed(out/f'case_{i:02d}.npz',law=law,target=target,target_vacua=vacua,covariance=cov,conditionals=conditionals,full_measurement_order=np.array(order))
            row['status']='completed'
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-begin;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished',i,row['status'],row.get('distances',{}).get('tv'),flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged;save(out/'summary.json',dict(planned=len(jobs),finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=dependencies))
if __name__=='__main__':main()
