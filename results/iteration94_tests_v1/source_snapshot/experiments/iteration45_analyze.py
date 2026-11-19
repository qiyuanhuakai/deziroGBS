"""Certified panel comparisons; no throughput or scaling extrapolation."""
import sys,json,shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration45_analysis';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('iteration45_analyze','iteration22_external','iteration33_run','verify_iteration34')];hashes={n:digest(ROOT/n) for n in names};deps={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    mps=read('results/iteration45_mps_recovery/comparisons.json');hier=read('results/iteration45_hierarchy_validation/rows.json');comparisons=read('results/iteration45_hierarchy_validation/comparisons.json');serialization=read('results/iteration45_serialization/summary.json');official=[]
    manifest=read('results/iteration44_manifest.json')
    for r in hier:
        if r['order']!=3 or r['source']['round']!=44:continue
        record=dict(source_case=r['source']['case'],degree3_true_maxent_TV=r['true_maxent_TV'])
        for k in (2,4):
            path=f"results/iteration44_native/case_{r['source']['case']:02d}_k{k}.audit.json";old=read(path);assert deps[path]==manifest['artifacts'][path];v=old['physical_TV_intervals']['uint64'];record[f'K{k}_TV']=v;record[f'better_than_K{k}']=r['true_maxent_TV'][1]<v[0];record[f'worse_than_K{k}']=r['true_maxent_TV'][0]>v[1]
        official.append(record)
    save(out/'official_comparisons.json',encode(official));improvements=[r['relative_midpoint_improvement'] for r in comparisons];summary=dict(mps_cases=len(mps),hierarchy_cases=len(hier),degree3_certified_improvements=len(comparisons),relative_midpoint_improvement_range=[min(improvements),max(improvements)],max_arithmetic_TV=max(float(r['arithmetic_TV']) for r in mps),max_quantization_TV=max(float(r['quantization_TV']) for r in mps),failure_mass=max(float(r['failure']) for r in mps),full_information_controls=sum(r['full_information'] for r in hier),official_comparisons=len(official),degree3_better_K2=sum(r['better_than_K2'] for r in official),degree3_worse_K2=sum(r['worse_than_K2'] for r in official),degree3_worse_K4=sum(r['worse_than_K4'] for r in official),serialization=serialization,source_hashes=hashes,dependencies=deps)
    plt.rcParams.update({'font.size':11});fig,axes=plt.subplots(1,3,figsize=(16,4.8),constrained_layout=True)
    for strength,style in [('moderate','-'),('strong','--')]:
        for rank,color in [(2,'#b55445'),(4,'#328779'),(8,'#395eaa')]:
            rows=[r for r in mps if r['job']['strength']==strength and r['job']['rank']==rank];axes[0].semilogy([r['job']['dimension'] for r in rows],[float(r['state_TV'][0]) for r in rows],style,marker='o',color=color,label=f'{strength}, bond {rank}')
    axes[0].set(title='Stored MPS vs physical target',xlabel='Local Fock dimension',ylabel='Total variation');axes[0].set_xticks([4,6,8]);axes[0].legend(fontsize=8)
    for order,color in [(2,'#a35250'),(3,'#337e99')]:
        rows=[r for r in hier if r['order']==order];axes[1].semilogy(range(12),[float(r['physical_TV'][0]) for r in rows],'o-',color=color,label=f'Degree {order}')
    axes[1].axvline(3.5,color='gray',linestyle=':');axes[1].set(title='Moment completion vs physical target',xlabel='Fixed input index (0–3 classical)',ylabel='Total variation');axes[1].legend()
    axes[2].semilogy(range(18),[float(r['arithmetic_TV']) for r in mps],'o-',label='Sampler vs stored state',color='#395eaa');axes[2].semilogy(range(18),[float(r['quantization_TV']) for r in mps],'x--',label='Uniform grid vs continuous CDF',color='#328779');axes[2].set(title='Arithmetic audit, including failure symbol',xlabel='MPS configuration index',ylabel='Total variation');axes[2].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.suptitle('Round 45: exact small-system audits; no large-device speed claim',fontsize=14);fig.savefig(out/'mps_and_hierarchy.png',dpi=180);plt.close(fig)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());summary.update(unchanged_sources=True,unchanged_dependencies=True);save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('source_hashes','dependencies')},indent=2))
if __name__=='__main__':main()
