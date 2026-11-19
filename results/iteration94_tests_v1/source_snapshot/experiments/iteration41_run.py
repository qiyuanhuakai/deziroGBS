"""Frozen physical ablation and unchanged official finite-word full-law panel."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import json,shutil,subprocess,time,sys,traceback
from fractions import Fraction as F
from pathlib import Path
from iteration22_external import digest,save
from iteration33_run import encode
from iteration39_independent import replay,reconstruct
from iteration41_physical import coefficients,target_law,covariance,half_l1
from iteration41_ablation import enumerate_laws,recurrence
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def jobs():
    result=[]
    for n in (2,4,6,7,8,10):
        for a in (F(1,16),F(1,4),F(1),F(4),F(16)):result.append(dict(active=n,intensity=a,weights=[F(1)]*n,order=list(range(n)),cutoff=2,family='equal'))
    for n in (4,8):
        w=[F(1,1<<(j%4)) for j in range(n)]
        for a in (F(1,4),F(4)):
            for order in (list(range(n)),list(reversed(range(n)))):result.append(dict(active=n,intensity=a,weights=[w[j] for j in order],order=order,cutoff=2,family='weighted'))
    for n in (4,8,10):
        for a in (F(1,4),F(4)):result.append(dict(active=n,intensity=a,weights=[F(1)]*n,order=list(range(n)),cutoff=4,family='equal'))
    return result

def main():
    out=ROOT/'results/iteration41_panel';out.mkdir(exist_ok=False)
    names=['research/iteration41_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration41_run','iteration41_ablation','iteration41_physical','iteration22_external','iteration33_run','iteration39_independent')]]
    hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    manifests={i:json.loads((ROOT/f'results/iteration{i}_manifest.json').read_text()) for i in (39,40)}
    def reuse(n,it):
        sha=digest(ROOT/n);assert sha==manifests[it]['artifacts'][n];deps[n]=sha;return ROOT/n
    exe=reuse('results/iteration39_build/frugal_extended.exe',39);dump=reuse('results/iteration40_target_build/frugal_target_dump.exe',40)
    for tool,it in ((exe,39),(dump,40)):
        reuse((tool.parent/'build.json').relative_to(ROOT).as_posix(),it)
        for p in tool.parent.glob('*.dll'):reuse(p.relative_to(ROOT).as_posix(),it)
    tasks=jobs();save(out/'plan.json',encode(dict(jobs=tasks,source_hashes=hashes,dependencies=deps,scope='Exact finite laws and exploratory-selected examples; no statistical confidence claim')));rows=[]
    for i,job in enumerate(tasks):
        start=time.perf_counter();row=dict(sequence=i,active=job['active'],intensity=str(job['intensity']),family=job['family'],order=job['order'],cutoff=job['cutoff']);print('start',i,row,flush=True)
        try:
            V=covariance(job['weights'],job['intensity']);csv=out/f'case_{i:02d}.csv'
            csv.write_text('\n'.join(','.join(format(float(x),'.17g') for x in line) for line in V)+'\n')
            assert all(F(float(x))==x for line in V for x in line)
            cmd=[str(exe),'--input',str(csv),'--modes','10','--cutoff',str(job['cutoff']),'--samples','65','--seed',str(411001+i),'--threads','1','--audit','--prefix-tree']
            for label,command in (('stdout',cmd),('reverse',[str(exe),'--input',str(csv),'--modes','10','--cutoff',str(job['cutoff']),'--samples','0','--seed',str(411001+i),'--threads','1','--prefix-tree','--reverse-prefixes'])):
                with (out/f'case_{i:02d}.{label}.json').open('w') as stdout,(out/f'case_{i:02d}.{label}.stderr.log').open('w') as stderr:r=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=600)
                assert r.returncode==0
            raw=json.loads((out/f'case_{i:02d}.stdout.json').read_text());reverse=json.loads((out/f'case_{i:02d}.reverse.json').read_text());assert {x['prefix']:x for x in raw['prefix_tree']}=={x['prefix']:x for x in reverse['prefix_tree']};steps=replay(raw)
            tree={r['prefix']:r['threshold_hex'] for r in raw['prefix_tree']};native={model:reconstruct(10,tree,model) for model in ('continuous','uint64')}
            core=target_law(job['weights'],job['intensity']);target=core+[F(0)]*((1<<10)-len(core));tv={k:half_l1(v,target) for k,v in native.items()};leak={k:sum(v[len(core):]) for k,v in native.items()}
            result=dict(job=job,command=cmd,native=native,target=target,TV=tv,vacuum_padding_leak=leak,replayed_steps=steps)
            row.update(TV_uint64=float(tv['uint64']),TV_continuous=float(tv['continuous']),vacuum_padding_leak={k:float(v) for k,v in leak.items()})
            if job['cutoff']==2:
                with (out/f'case_{i:02d}.coefficients.json').open('w') as stdout,(out/f'case_{i:02d}.coefficients.stderr.log').open('w') as stderr:r=subprocess.run([str(dump),str(csv),'10'],stdout=stdout,stderr=stderr,timeout=600)
                assert r.returncode==0;export=json.loads((out/f'case_{i:02d}.coefficients.json').read_text());p,c=coefficients(job['weights'],job['intensity']);controls=enumerate_laws(p,c)
                native_p=[(1+F(float.fromhex(x)))/2 for x in export['first_hex'][:job['active']]];native_c={(j,k):F(float.fromhex(x))/4 for j,k,x in export['pairs'] if k<job['active']}
                native_reference=[recurrence(''.join(str((mask>>j)&1) for j in range(job['active'])),native_p,native_c)['mass'] for mask in range(1<<job['active'])]
                negative={k:sum((-x for x in controls[k] if x<0),F(0)) for k in ('matching','unclamped','clamped')}
                result.update(physical_pvac=p,physical_covariance=c,controls=controls,native_coefficient_reference_signed=native_reference,negative_mass=negative,approximation_half_L1=half_l1(controls['matching'],controls['unclamped']),real_sampler_TV=half_l1(controls['sampler'],core),native_to_physical_real_sampler_TV=half_l1(native['uint64'],controls['sampler']+[F(0)]*((1<<10)-len(core))))
                row.update(negative_mass={k:float(v) for k,v in negative.items()},approximation_half_L1=float(result['approximation_half_L1']),real_sampler_TV=float(result['real_sampler_TV']),native_to_real_TV=float(result['native_to_physical_real_sampler_TV']),saturated_reachable=len(controls['saturated']))
            save(out/f'case_{i:02d}.audit.json',encode(result));row.update(status='completed',replayed_steps=steps)
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=len(tasks),completed=sum(x['status']=='completed' for x in rows),finished=len(rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
