"""Independent generating-function product, determinant and ideal cutoff laws."""
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode,exact_determinant
from iteration48_independent import polynomial_hermite,bisect_sqrt
from iteration45_independent import inverse,tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration52_box_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration52_box','iteration22_external','iteration33_run','verify_iteration34','iteration48_independent','iteration47_radicals','iteration45_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[]
    for n in names:
        p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration52_box';summary=read(base+'/summary.json');check('eight references complete',summary['planned']==summary['completed']==8)
    for n,s in summary['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    for i in range(8):
        rec=read(base+f'/case_{i:02d}.json');job=rec['job'];r=rec['reference'];source=read(f"results/iteration{job['source_round']}_inputs/case_{job['case']:02d}.json");p=source['physical'];re=p['pairing_real'];im=p['pairing_imag'];D=r['pairing_denominator'];check('integer pairing '+str(i),r['integer_pairing']==[[[re[a][b]*D,im[a][b]*D] for b in range(8)] for a in range(8)]);C=r['integer_pairing'];H=polynomial_hermite([[F(x[0]) for x in row] for row in C],[[F(x[1]) for x in row] for row in C],[job['dimension']]*8);check('independent generating product '+str(i),[[a,b] for a,b in H]==r['integer_hermite']);masses=[F(0)]*256
        for occ,(a,b) in zip(np.ndindex(*([job['dimension']]*8)),H):masses[sum(1<<j for j,x in enumerate(occ) if x)]+=(a*a+b*b)/(D**sum(occ)*math.prod(math.factorial(k) for k in occ))
        R=sum(masses);law=[x/R for x in masses];check('independent normalized law '+str(i),masses==r['unnormalized_threshold_masses'] and R==r['unnormalized_norm'] and law==r['distribution'] and sum(law)==1)
        S=[[re[a][b] for b in range(8)]+[im[a][b] for b in range(8)] for a in range(8)]+[[im[a][b] for b in range(8)]+[-re[a][b] for b in range(8)] for a in range(8)];A=[[F(a==b)-S[a][b] for b in range(16)] for a in range(16)];det=exact_determinant(A);inv=inverse(A);V=[[2*inv[a][b]-int(a==b) for b in range(16)] for a in range(16)];check('independent physical covariance '+str(i),V==p['covariance']==r['covariance'] and det==r['normalization_squared'] and all(x>0 for x in r['positive_pivots']));l,h=bisect_sqrt(det);a,b=r['normalization_interval'];check('independent normalization '+str(i),a<=l<=h<=b);a,b=r['projection_probability_interval'];check('projection mass '+str(i),a<=l*R<=h*R<=b);dl,dh=r['tail_trace_interval'];check('tail trace enclosure '+str(i),dl*dl<=1-b<=1-a<=dh*dh);value=tv(law,source['target']['target_intervals']);check('complete target TV '+str(i),value==tuple(rec['physical_TV']) and value[1]<=dh);rows.append(dict(case=i,job=job,physical_TV=value,projection=(a,b),tail_trace=(dl,dh)));print('verified ideal box',i,flush=True)
    check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=8,source_hashes=hashes,dependencies=deps,scope='Exact local physical projection, no MPS construction; generating-function product independent of Hermite recurrence'))
if __name__=='__main__':main()
