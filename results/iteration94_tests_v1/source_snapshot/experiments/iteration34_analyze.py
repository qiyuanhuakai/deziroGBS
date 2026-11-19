import json,shutil,sys
from fractions import Fraction as F
from decimal import Decimal,localcontext
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from iteration22_external import digest,save
ROOT=Path(__file__).resolve().parents[1]

def main():
    sys.set_int_max_str_digits(0);out=ROOT/'results/iteration34_analysis';out.mkdir(exist_ok=False);source=Path(__file__);sha=digest(source);shutil.copy2(source,out/source.name);rows=json.loads((ROOT/'results/iteration34_panel/rows.json').read_text());original_seconds=sum(r['seconds'] for r in rows);recovered=rows[37]['status']!='completed'
    if recovered:rows[37]=json.loads((ROOT/'results/iteration34_recovery/case_37.json').read_text())
    assert len(rows)==42 and all(r['status']=='completed' for r in rows);save(out/'effective_rows.json',rows);native=[]
    for r in rows:
        if r['panel']=='reused':continue
        native.append(dict(case=r['sequence'],panel=r['panel'],modes=r['modes'],family=r.get('family'),exponent=r.get('exponent'),precision=r['precision'],target_scope='Exact stored covariance violates quantum uncertainty; determinant-function intervals only, NOT a physical TV certificate' if r['panel']=='original' else r['target_scope'],native_calls=r['native_calls'],unique_conditionals=r['unique_conditionals'],duplicate_prefix_checks=r['duplicate_prefix_checks'],failure_nodes=r['failure_nodes'],certificate_seconds=r['certificate_seconds'],native_seconds=r['native_seconds'],whitening_residual=r['whitening_residual'],maximum_certificate_atom_width=r['maximum_certificate_atom_width'],ideal_final=next(b for b in reversed(r['bounds']) if b['uniform_bits'] is None),grid_final=next(b for b in reversed(r['bounds']) if b['uniform_bits']==53)))
    original=json.loads((ROOT/'results/iteration34_panel/case_40.audit.json').read_text());b=next(a for a in reversed(original['audits']) if a['uniform_bits']==53);upper=F(b['tv_upper']['rational']);overlap=F(b['confirmed_overlap']['rational'])
    with localcontext() as context:
        context.prec=60;original_bound=str(Decimal(upper.numerator)/Decimal(upper.denominator));original_overlap=str(Decimal(overlap.numerator)/Decimal(overlap.denominator))
    reused=[]
    for k in (0,2,4,6,8):
        group=[next(b for b in r['bounds'] if b['max_clicks']==k and b['uniform_bits'] is None) for r in rows[9:33]];reused.append(dict(max_clicks=k,minimum_upper=min(b['tv_upper'] for b in group),maximum_upper=max(b['tv_upper'] for b in group)))
    physicality=json.loads((ROOT/'results/iteration34_physicality/rows.json').read_text());assert physicality[0]['strict_violation_proved'] and not any(r['strict_violation_proved'] for r in physicality[1:])
    summary=dict(analyzer_sha256=sha,effective_jobs=42,original_completed=41 if recovered else 42,serialization_recovery_jobs=int(recovered),native_cases=native,reused_layers=reused,original_functional_upper_decimal=original_bound,original_functional_overlap_decimal=original_overlap,physicality_witnesses=[dict(job=r['job'],floating_minimum_eigenvalue=r['floating_minimum_eigenvalue'],exact_rayleigh_float=r['exact_rayleigh_float'],strict_violation_proved=r['strict_violation_proved']) for r in physicality],total_native_calls=sum(r['native_calls'] for r in native),total_duplicate_prefix_checks=sum(r['duplicate_prefix_checks'] for r in native),effective_case_seconds=sum(r['seconds'] for r in rows),all_attempt_seconds=original_seconds+(rows[37]['seconds'] if recovered else 0),scope='Full-attempt bounds for specified cache_size=0 physical thermal/classical rank-one controls; target tables are never enumerated for large cases. The stored original 144-mode matrix has a strict exact uncertainty violation, so its determinant-function overlaps are NOT physical-TV certificates. Unselected output mass and unseen failures remain uncontrolled; default-cache or experimental-source guarantees do not follow.')
    summary['timing_scope']='Sum of per-case wall times, not elapsed makespan or isolated throughput; serialization recovery overlaps later original-panel work, and supplemental physicality checks overlap the main run.'
    fig,axes=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    labels={('thermal',8):'Thermal, excess 2^-8',('thermal',40):'Thermal, excess 2^-40',('rank_one',5):'Rank-one noise, a=2^-5',('rank_one',4):'Rank-one noise, a=2^-4'}
    for r in rows[33:36]:
        group=[b for b in r['bounds'] if b['uniform_bits']==53];axes[0].semilogy([b['max_clicks'] for b in group],[b['tv_upper'] for b in group],'o-',label=labels[(r['family'],r['exponent'])])
    axes[0].set(title='32 modes: selected full-output certificate',xlabel='Maximum selected clicks',ylabel='TV upper bound (grid53)');axes[0].legend(fontsize=7)
    for r in rows[36:40]:
        group=[b for b in r['bounds'] if b['uniform_bits']==53];axes[1].semilogy([b['max_clicks'] for b in group],[b['tv_upper'] for b in group],'o-',label=labels[(r['family'],r['exponent'])])
    axes[1].set(title='144 modes: structured physical controls',xlabel='Maximum selected clicks',ylabel='TV upper bound (grid53)');axes[1].set_xticks([0,1]);axes[1].legend(fontsize=7)
    for kind in (None,53):
        group=[b for b in rows[40]['bounds'] if b['uniform_bits']==kind];axes[2].semilogy([b['max_clicks'] for b in group],[b['confirmed_overlap'] for b in group],'o-',label='Ideal Bernoulli' if kind is None else 'Abstract grid53')
    axes[2].set(title='Existing 144-mode matrix: functions only',xlabel='Vacuum / vacuum + 16 selected singles',ylabel='Function overlap (NOT a physical-TV certificate)');axes[2].set_xticks([0,1]);axes[2].legend(fontsize=7);fig.suptitle('Iteration 34: complete-string overlap, exact residual enclosures, explicit scope');fig.savefig(out/'atom_certificates.png',dpi=140);plt.close(fig);summary['unchanged_source']=digest(source)==sha;save(out/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='native_cases'},indent=2))
if __name__=='__main__':main()
