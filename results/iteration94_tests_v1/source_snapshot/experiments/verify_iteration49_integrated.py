"""Independent radical grouping of continuous Gaussian moment integrals."""
import sys,json,shutil,math
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration47_radicals import squarefree
from iteration48_independent import bisect_sqrt
from iteration45_independent import tv
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)


def main():
    out=ROOT/'results/iteration49_integrated_validation';out.mkdir(exist_ok=False);names=[f'experiments/{n}.py' for n in ('verify_iteration49_integrated','iteration22_external','iteration33_run','verify_iteration34','iteration47_radicals','iteration48_independent','iteration45_independent')];hashes={n:digest(ROOT/n) for n in names};deps={};checks=[];rows=[];references={}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    def read(n):deps[n]=digest(ROOT/n);return decode(json.loads((ROOT/n).read_text()))
    def check(n,ok):
        checks.append(dict(name=n,passed=bool(ok)))
        if not ok:save(out/'checks.failed.json',checks);raise AssertionError(n)
    base='results/iteration49_integrated';summary=read(base+'/summary.json');check('36 integrated controls',summary['planned']==summary['completed']==36)
    for n,s in summary['source_hashes'].items():check('source '+n,digest(ROOT/n)==s==digest(ROOT/base/'source_snapshot'/n))
    for n,s in summary['dependencies'].items():check('dependency '+n,digest(ROOT/n)==s);deps[n]=s
    validation=read('results/iteration49_noise_validation/validation.json');check('complete independent noisy program audit',validation['cases']==36 and validation['count']==validation['passed'])
    for index in range(36):
        rec=read(base+f'/case_{index:02d}.json');name=f'results/iteration49_noise/case_{index:02d}.json';old=read(name);check(f'{index} old audited record',deps[name]==validation['dependencies'][name] and rec['job']==old['job']);name=f'results/iteration49_noise/case_{index:02d}.npz';deps[name]=digest(ROOT/name);check(f'{index} audited tensor',deps[name]==validation['dependencies'][name]);t=np.load(ROOT/name)['tensor_0'];amps=[(F(float(z.real)),F(float(z.imag))) for z in t.ravel()];norm=sum(a*a+b*b for a,b in amps);sigma=rec['job']['sigma'];A=1+2*sigma*sigma;groups={}
        for a,(ar,ai) in enumerate(amps):
            for b,(br,bi) in enumerate(amps):
                p=a+b
                if p%2:continue
                # Incorporate the normal integral normalization into one radical.
                moment=F(math.prod(range(1,p,2)))*sigma**p/A**(p//2);den=math.factorial(a)*math.factorial(b)*A.numerator*A.denominator;sq,rad=squarefree(den);coefficient=(ar*br+ai*bi)*moment*A.denominator/(norm*sq);groups[rad]=groups.get(rad,F(0))+coefficient
        lo=hi=F(0)
        for rad,c in groups.items():
            l,h=bisect_sqrt(F(1,rad));lo+=c*(l if c>=0 else h);hi+=c*(h if c>=0 else l)
        integral=rec['integral'];a,b=integral['vacuum_interval'];check(f'{index} independent continuous density integral',a<=max(F(0),lo)<=min(F(1),hi)<=b and integral['law']==[[a,b],[1-b,1-a]]);save(out/f'case_{index:02d}.radicals.json',encode(dict(groups=groups,independent_vacuum_interval=(lo,hi))));key=(rec['job']['pairing'],sigma,rec['job']['dimension'])
        if key in references:check(f'{index} same continuous input across grids',references[key]==integral)
        else:references[key]=integral
        error=tv(old['mixture_law'],integral['law']+[(F(0),F(0))]);check(f'{index} discrete program versus finite continuous target',list(error)==rec['program_vs_finite_continuous_TV']);c,d=old['continuous_target'][0];representation=(a-d,b-c);numerics=(old['mixture_law'][0]-b,old['mixture_law'][0]-a);check(f'{index} signed decomposition',list(representation)==rec['finite_continuous_vs_Gaussian_signed_vacuum_bias'] and list(numerics)==rec['program_vs_finite_continuous_signed_vacuum_bias']);direct=(old['mixture_law'][0]-d,old['mixture_law'][0]-c);check(f'{index} complete signed bias inclusion',representation[0]+numerics[0]<=direct[0]<=direct[1]<=representation[1]+numerics[1]);check(f'{index} continuous representation contraction',max(abs(x) for x in representation)<=old['bridge']['full_trace_interval'][1]);rows.append(dict(case=index,job=rec['job'],representation_bias=representation,grid_and_numeric_bias=numerics,program_error=error,actual_TV=old['full_Gaussian_TV'],total_TV_upper=old['total_TV_upper']));print('verified integral',index,flush=True)
    check('twelve distinct continuous references',len(references)==12);check('sources unchanged',all(digest(ROOT/n)==s for n,s in hashes.items()));check('dependencies unchanged',all(digest(ROOT/n)==s for n,s in deps.items()));save(out/'rows.json',encode(rows));save(out/'checks.json',checks);save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),cases=36,unique_inputs=len(references),source_hashes=hashes,dependencies=deps))
if __name__=='__main__':main()
