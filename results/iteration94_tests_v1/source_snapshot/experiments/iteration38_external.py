"""Same-input small-system official Frugal diagnostics and forest context checks."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,itertools,traceback
from fractions import Fraction as F
from time import perf_counter
import numpy as np
from iteration37_support import ROOT,trace_details
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from gbslab.frugal import frugal_sample,validate_adapter
from gbslab.wire_forest import WireForestSampler
from gbslab.exact import ThresholdModel
from gbslab.atom_certificate import selected_atoms


def main():
    sys.set_int_max_str_digits(0);out=ROOT/'results/iteration38_external';out.mkdir(exist_ok=False);names=['research/iteration38_external_plan.md',*[f'experiments/{n}.py' for n in ('iteration38_external','iteration37_support','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dst=out/'source_snapshot'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dst)
    shutil.copy2(ROOT/'research/iteration38_external_plan.md',out/'frozen_plan.md');deps={}
    def reuse(p,it=None):
        sha=digest(ROOT/p)
        if it is not None:assert sha==json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text())['artifacts'][p]
        deps[p]=sha;return ROOT/p
    primary=np.load(reuse('results/iteration35_panel/inputs.npz',35))['shifted'];lib=reuse('results/iteration31_build/libwire_forest.dll',31);reuse('results/iteration31_build/build.json',31);exe=reuse('results/iteration38_frugal_runtime/frugal_adapter.exe');reuse('results/iteration38_frugal_runtime/build.json');runtime=json.loads(reuse('results/iteration38_frugal_runtime/runtime.json').read_text())
    for n,v in runtime['files'].items():assert digest(reuse('results/iteration38_frugal_runtime/'+n))==v['sha256']
    targets={s:decode(json.loads(reuse(f'results/iteration37_local/target_{s:03d}.json',37).read_text()))['law']['intervals'] for s in (0,54,134)};previous={}
    for i in range(24):previous[i]=json.loads(reuse(f'results/iteration37_local/case_{i:02d}.first.json',37).read_text())
    jobs=[dict(panel='calibration')]+[dict(panel='frugal',start=s,cutoff=k,samples=65536) for s in (0,54,134) for k in range(1,6)]+[dict(panel='forest',start=s,selection=c,forest=f,budget=4) for s in (0,54,134) for c in ('recent','pairwise','conditional_mi','click_influence') for f in ('chain','static_mi_tree')];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps,frugal_probability_scope='floating continuous-uniform trace; exported vector distance only',forest_probability_tolerance=1e-12));rows=[]
    for i,job in enumerate(jobs):
        begin=perf_counter();row=dict(sequence=i,**job);print('starting external',i,job,flush=True)
        try:
            if job['panel']=='calibration':
                rec=validate_adapter(executable=exe,sample_count=65536);row.update(validation_cases=len(rec['validation_cases']),passed=rec['passed'])
            else:
                start=job['start'];ix=list(range(start,start+10))+list(range(start+144,start+154));V=primary[np.ix_(ix,ix)];np.savez_compressed(out/f'case_{i:02d}.npz',covariance=V);target=targets[start]
                if job['panel']=='frugal':
                    rec=frugal_sample(V,job['cutoff'],65536,seed=382000+i,executable=exe,threads=1);q=tuple(F(float(x)) for x in rec['distribution']);lo=sum(max(F(0),a-p,p-b) for p,(a,b) in zip(q,target))/2;hi=sum(max(abs(p-a),abs(p-b)) for p,(a,b) in zip(q,target))/2;rec.update(exported_vector_half_L1_interval=(lo,hi),exported_vector_normalization_defect=sum(q)-1);row.update(half_L1_export=float((lo+hi)/2),normalization_defect=float(sum(q)-1),saturated_steps=rec['trace']['saturated_threshold_path_steps'],nonfinite_steps=rec['trace']['nonfinite_threshold_path_steps'])
                else:
                    records=[]
                    with WireForestSampler(ThresholdModel(V),4,selection=job['selection'],forest=job['forest'],cache_size=0,precision='auto',library=lib) as sampler:
                        for depth in range(10):
                            for h in itertools.product((0,1),repeat=depth):records.append(trace_details(sampler,h)[1])
                        native=sampler.kernel.cache_info()
                    source=i-16;old=previous[source];assert len(old)==len(records)==1023;identity=[]
                    for a,b in zip(records,old):
                        same=a['prefix']==b['prefix'] and [j+start for j in a.get('retained',())]==b.get('retained',[]) and [[j+start for j in e] for e in a.get('edges',())]==b.get('edges',[]);difference=abs(float.fromhex(a['q_hex'])-float.fromhex(b['q_hex'])) if a['q_hex'] is not None and b['q_hex'] is not None else (0.0 if a['q_hex']==b['q_hex'] else float('inf'));identity.append(dict(prefix=a['prefix'],same_structure=same,q_hex_equal=a['q_hex']==b['q_hex'],probability_difference=difference))
                    table={tuple(map(int,r['prefix'])):r['q_hex'] for r in records};chosen=[tuple(j for j in range(10) if mask>>j&1) for mask in range(1024)]
                    def oracle(h):return None if table[h] is None else F(float.fromhex(table[h]))
                    audits=[{k:v for k,v in selected_atoms(10,chosen,oracle,lambda c:target[sum(1<<j for j in c)],uniform_bits=bits).items() if k not in ('records','patterns','target_intervals')} for bits in (None,53)];rec=dict(job=job,oracle=records,context_source=source,context_checks=identity,audits=audits,native_diagnostics=native);row.update(tv_ideal=float(audits[0]['tv_upper']),tv_grid=float(audits[1]['tv_upper']),native_calls=native['probability_calls'],max_context_probability_difference=max(x['probability_difference'] for x in identity),context_structure_mismatches=sum(not x['same_structure'] for x in identity),q_hex_context_mismatches=sum(not x['q_hex_equal'] for x in identity));assert not row['context_structure_mismatches'] and row['max_context_probability_difference']<=1e-12
            save(out/f'case_{i:02d}.audit.json',encode(rec));row['status']='completed'
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=perf_counter()-begin;rows.append(row);save(out/f'case_{i:02d}.json',row);save(out/'rows.json',rows);print('finished external',i,row,flush=True)
    unchanged=hashes=={n:digest(ROOT/n) for n in names};assert unchanged and all(digest(ROOT/p)==sha for p,sha in deps.items());save(out/'summary.json',dict(planned=40,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=unchanged,dependencies=deps))
if __name__=='__main__':main()
