"""Post-hoc formula substitution, not a sampler or runtime lower bound."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import shutil
import sys
import mpmath as mp


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output; out.mkdir(parents=True,exist_ok=False)
    source=Path(__file__); sha=hashlib.sha256(source.read_bytes()).hexdigest()
    shutil.copy2(source,out/source.name)
    prior=source.resolve().parents[1]/'results/iteration11_meixner/budgets.json'
    prior_bytes=prior.read_bytes(); prior_rows=json.loads(prior_bytes)
    (out/'iteration11_budgets.json').write_bytes(prior_bytes)
    plan=dict(kind='post-hoc literature formula substitution only',source='https://arxiv.org/html/2611.08444v1',equations=['G.4','G.28','G.32','G.40','G.76','G.101'],modes=[32,128],squeezing='0.9',eta=['0.2','0.4','0.65','0.85'],beta='0',epsilon_L1='0.1',delta='0.05',mpmath_dps=80,source_sha256=sha)
    plan['prior_budget_sha256']=hashlib.sha256(prior_bytes).hexdigest()
    (out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    mp.mp.dps=plan['mpmath_dps']; rows=[]; checks=[]
    for m in plan['modes']:
        for eta_text in plan['eta']:
            eta=mp.mpf(eta_text); n=mp.sinh(mp.mpf(plan['squeezing']))**2
            a=n+(1+n)/m; b=eta*a; q=eta**2*a*(1+a)/(b*(1+b)); C=mp.e*(1+(m+1)*n)
            eps=mp.mpf(plan['epsilon_L1']); delta=mp.mpf(plan['delta'])
            D=max(0,int(mp.ceil(mp.log((C-1)/(delta*eps**2))/mp.log(1/q)))-1)
            boundary=(C-1)*q**(D+1)/(eps**2); previous=(C-1)*q**D/(eps**2)
            nf=math.sinh(.9)**2; af=nf+(1+nf)/m; ef=float(eta_text); bf=ef*af
            qf=ef**2*af*(1+af)/(bf*(1+bf)); Cf=math.e*(1+(m+1)*nf)
            Df=max(0,math.ceil(math.log((Cf-1)/(.05*.1**2))/math.log(1/qf))-1)
            checks.extend([dict(name=f'{m}/{eta_text} minimum integer',passed=bool(boundary<=delta and (D==0 or previous>delta))),dict(name=f'{m}/{eta_text} float agreement',passed=bool(D==Df and abs(mp.mpf(qf)-q)<mp.mpf('2e-15')))])
            count=math.comb(m+D,D)
            if eta_text=='0.65':
                old=[row for row in prior_rows if row['modes']==m and row['profile']=='bright' and row['l1_error']==.1]
                checks.append(dict(name=f'{m} previous round budget consistency',passed=bool(len(old)==1 and old[0]['degree']==D and old[0]['naive_coefficient_count']==str(count))))
            rows.append(dict(modes=m,eta=eta_text,squeezing=plan['squeezing'],q=mp.nstr(q,55),C=mp.nstr(C,55),degree=D,deterministic_coefficient_count=str(count),log10_coefficient_count=mp.nstr(mp.log10(count),55),log10_variance_upper_bound=mp.nstr(mp.log10(3)+D*mp.log10(6*mp.e)+mp.log10(C),55),markov_failure_bound=mp.nstr(boundary,55),previous_degree_failure_bound=mp.nstr(previous,55),identity_ideal_log10_one_plus_chi2_lower_bound=mp.nstr(m*mp.log10((1+2*a)/(1+a)),55)))
    checks.append(dict(name='source unchanged',passed=hashlib.sha256(source.read_bytes()).hexdigest()==sha))
    summary=dict(rows=rows,checks=checks,passed=sum(c['passed'] for c in checks),count=len(checks),python=sys.version,platform=platform.platform(),mpmath=mp.__version__,scope='L1 signed approximation cutoff only; no implementation of the sampler and no runtime lower bound')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    table=['# 有符号截断粗界的参数代入','','ε_L1=0.1，δ=0.05，r=0.9，β=0。不是完整采样器性能。','','| m | η | 最小 D | log10 Q_D | log10 方差上界 |','|---:|---:|---:|---:|---:|']
    for r in rows: table.append(f"| {r['modes']} | {r['eta']} | {r['degree']} | {float(r['log10_coefficient_count']):.4f} | {float(r['log10_variance_upper_bound']):.4f} |")
    (out/'summary.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),passed=summary['passed'],count=summary['count'])))
    if summary['passed']!=summary['count']: raise AssertionError('preserve failed output')


if __name__=='__main__': main()
