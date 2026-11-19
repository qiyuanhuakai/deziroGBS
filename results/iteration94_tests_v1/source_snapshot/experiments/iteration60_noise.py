"""Preselected rank4 grid refinement versus finite-input expansion."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,traceback,gzip,hashlib,math
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from iteration46_displaced_trace import enumerate_trace
from gbslab.gaussian_fock_bridge import gaussian_fidelity,hermite_box
from gbslab.extended_displaced_fock import displaced_finite_law
from gbslab.growing_normal_grid import growing_grid,scaled_exp_minus
from gbslab.multidrive_gaussian_noise import integrated_linear_law,gaussian_linear_target,product_coupling_bound
from gbslab.rank_one_noise import interval_law
from gbslab.joint_displacement_coupling import joint_coupling
from verify_iteration34 import decode
from gbslab.sampling_audit import choice_intervals
from gbslab.interval_completion import physical_tv_interval
from gbslab.local_mps import sample_local_threshold
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def interval_tv(a,b):return (sum(max(F(0),l-y,x-h) for (l,h),(x,y) in zip(a,b))/2,sum(max(abs(l-y),abs(h-x)) for (l,h),(x,y) in zip(a,b))/2)

def main():
    out=ROOT/'results/iteration60_noise';out.mkdir(exist_ok=False);names=['research/iteration60_budget_plan.md',*[f'experiments/{n}.py' for n in ('iteration60_noise','iteration46_displaced_trace','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    deps={n:digest(ROOT/n) for n in ('results/iteration59_manifest.json','results/iteration59_noise_validation/validation.json','results/iteration60_selection/summary.json','results/iteration60_selection/selection.json','results/iteration60_selection/jobs.json','results/iteration60_selection_validation/validation.json')};validation=decode(json.loads((ROOT/'results/iteration60_selection_validation/validation.json').read_bytes()));assert validation['count']==validation['passed'] and all(deps[n]==validation['dependencies'][n] for n in ('results/iteration60_selection/selection.json','results/iteration60_selection/jobs.json'));planned=decode(json.loads((ROOT/'results/iteration60_selection/jobs.json').read_bytes()));save(out/'plan.json',encode(dict(jobs=planned,source_hashes=hashes,dependencies=deps,numpy_version=np.__version__)));rows=[];cache={}
    for index,job in enumerate(planned):
        started=time.perf_counter();row=dict(case=index,family=job['family'],action=job['action'],dimension=job['dimension'],cells=job['cells'])
        try:
            d=job['dimension'];directions=job['directions'];sigmas=job['sigmas'];key=(job['family'],d);hit=key in cache
            if not hit:
                shape=[[F(24,25),-F(7,25)],[-F(7,25),-F(24,25)]] if job['rotated'] else [[F(0),F(1)],[F(1),F(0)]];br=[[x*job['pairing']*job['phase'][0] for x in row] for row in shape];bi=[[x*job['pairing']*job['phase'][1] for x in row] for row in shape];occ,H=hermite_box(br,bi,[d,d]);A=np.array([complex(float(a),float(b))/math.sqrt(math.prod(math.factorial(k) for k in o)) for o,(a,b) in zip(occ,H)]).reshape(d,d);A/=np.linalg.norm(A);u,s,vh=np.linalg.svd(A,full_matrices=False);tensors=[np.ascontiguousarray((u*s).reshape(1,d,d)),np.ascontiguousarray(vh.reshape(d,d,1))];bridge=gaussian_fidelity(tensors,br,bi);target=gaussian_linear_target(br,bi,directions,sigmas);reference=integrated_linear_law(tensors,directions,sigmas);cache[key]=(br,bi,tensors,bridge,target,reference,s)
            br,bi,tensors,bridge,target,reference,sv=cache[key];np.savez_compressed(out/f'case_{index:02d}.npz',tensor_0=tensors[0],tensor_1=tensors[1]);grids=[growing_grid(s,N,radius={4:2,16:3}[N]) for s,N in zip(sigmas,job['cells'])];probs=[np.array([float.fromhex(x) for x in g['probabilities_hex']]) for g in grids];counts=[];axes=[]
            for p,g in zip(probs,grids):
                _,c,fcdf=choice_intervals(p);assert [float(x).hex() for x in fcdf]==g['cdf_hex'];counts.append(c);axes.append([F(x,1<<53) for x in c])
            pairs=list(product(*(range(len(a)) for a in axes)));weights=[math.prod(axes[a][i] for a,i in enumerate(ij)) for ij in pairs];idealweights=[(math.prod(grids[a]['weight_intervals'][i][0] for a,i in enumerate(ij)),math.prod(grids[a]['weight_intervals'][i][1] for a,i in enumerate(ij))) for ij in pairs];selector=physical_tv_interval(weights,idealweights);mixture=[F(0)]*5;gridvac=[[F(0),F(0)] for _ in range(4)];arithmetic=F(0);components=[];preparation=time.perf_counter()-started
            for node,(ij,w,(wl,wh)) in enumerate(zip(pairs,weights,idealweights)):
                xs=[g['nodes'][k] for g,k in zip(grids,ij)];exact=[(sum(v[a][0]*x for v,x in zip(directions,xs)),sum(v[a][1]*x for v,x in zip(directions,xs))) for a in range(2)];alpha=np.array([complex(float(a),float(b)) for a,b in exact]);assert all((F(float(z.real)),F(float(z.imag)))==ab for z,ab in zip(alpha,exact));finite=displaced_finite_law(tensors,alpha);sampling=enumerate_trace(tensors,alpha);epsilon=physical_tv_interval(sampling['word_law'],finite['intervals']+[(F(0),F(0))]);arithmetic+=w*epsilon[1]
                for k in range(5):mixture[k]+=w*sampling['word_law'][k]
                gv=[]
                for mask,r in enumerate(target['records']):
                    exponent=sum(xs[a]*r['exponent_matrix'][a][b]*xs[b] for a in range(4) for b in range(4));e=scaled_exp_minus(exponent);p=r['pure_factor_interval'];value=(p[0]*e[0],p[1]*e[1]);gv.append(value);gridvac[mask][0]+=wl*value[0];gridvac[mask][1]+=wh*value[1]
                components.append(dict(node=node,pair=ij,xs=xs,finite_target=finite,sampling=sampling,arithmetic_TV=epsilon,pure_Gaussian_vacuum_intervals=gv))
            gridvac[0]=[F(1),F(1)];_,ideal=interval_law(gridvac);actual=physical_tv_interval(mixture,target['intervals']+[(F(0),F(0))]);finiteerr=physical_tv_interval(mixture,reference['intervals']+[(F(0),F(0))]);referr=interval_tv(reference['intervals'],target['intervals']);griderr=interval_tv(ideal,target['intervals']);old_coupling=product_coupling_bound(target,grids);coupling=joint_coupling(target['generator_gram'],[dict(sigma=g['sigma'],cells=g['cells'],radius=g['standard_radius']) for g in grids]);assert old_coupling['upper']==coupling['old_axis_upper'];bound=min(F(1),bridge['full_trace_interval'][1]+selector[1]+arithmetic+coupling['best_upper']);assert actual[1]<=bound and griderr[1]<=coupling['best_upper'];seed=600000+index;rng=np.random.default_rng(seed);selected=[];outputs=[]
            for _ in range(7):
                ij=[int(rng.choice(len(p),p=p)) for p in probs];xs=[g['nodes'][k] for g,k in zip(grids,ij)];alpha=np.array([[complex(float(sum(v[a][0]*x for v,x in zip(directions,xs))),float(sum(v[a][1]*x for v,x in zip(directions,xs)))) for a in range(2)]]);value,_=sample_local_threshold(tensors,alpha,rng);selected.append(ij);outputs.append(value[0].tolist())
            computation=time.perf_counter()-started-preparation;raw=(json.dumps(encode(components),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode();compressed=gzip.compress(raw,compresslevel=6,mtime=0);name=f'case_{index:02d}.components.json.gz';(out/name).write_bytes(compressed);storage=dict(path=name,raw_bytes=len(raw),compressed_bytes=len(compressed),raw_sha256=hashlib.sha256(raw).hexdigest(),nodes=len(components),trace_nodes=sum(r['sampling']['nodes'] for r in components));save(out/f'case_{index:02d}.json',encode(dict(job=job,stored_strides=[t.strides for t in tensors],singular_values_hex=[float(x).hex() for x in sv],pairing_real=br,pairing_imag=bi,bridge=bridge,target=target,finite_continuous=reference,grids=grids,axis_selector_counts=counts,selector_weights=weights,selector_TV=selector,mixture_law=mixture,weighted_arithmetic_TV_upper=arithmetic,ideal_grid_vacuum_intervals=gridvac,ideal_grid_law=ideal,ideal_grid_TV=griderr,full_Gaussian_TV=actual,program_vs_finite_continuous_TV=finiteerr,finite_continuous_vs_Gaussian_TV=referr,coupling=coupling,old_coupling=old_coupling,total_TV_upper=bound,old_total_TV_upper=min(F(1),bridge['full_trace_interval'][1]+selector[1]+arithmetic+old_coupling['upper']),calibration=dict(seed=seed,samples=7,selected_nodes=selected,outputs=outputs),storage=storage)));row.update(status='completed',TV=[float(x) for x in actual],finite_TV=[float(x) for x in finiteerr],reference_TV=[float(x) for x in referr],arithmetic=float(arithmetic),total_upper=float(bound),input_reference_cache_hit=hit,storage=storage,timing=dict(preparation=preparation,components_and_comparisons=computation,storage=time.perf_counter()-started-preparation-computation))
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-started;rows.append(row);save(out/'rows.json',rows);print('rank4 budget',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=6,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,source_hashes=hashes,dependencies=deps,unchanged_dependencies=all(digest(ROOT/n)==s for n,s in deps.items())))
if __name__=='__main__':main()
