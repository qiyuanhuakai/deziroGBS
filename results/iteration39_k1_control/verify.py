"""Independent binary64 rounding cells and exact induction checks."""
import sys,json,struct,hashlib
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.set_int_max_str_digits(0)
from verify_iteration34 import decode
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return decode(json.loads(p.read_text()))

def main():
    out=Path(__file__).parent;assert not (out/'verification.json').exists();r=read(out/'proof.json');checks=[]
    def check(n,b):checks.append(dict(name=n,passed=bool(b)));assert b,n
    check('proof source bound',sha(out/'prove.py')==r['source_sha256'])
    for n,v in r['dependencies'].items():check('dependency '+n,sha(ROOT/n)==v)
    raw=read(ROOT/'results/iteration39_global/case_00.stdout.json');u=F(1,2**53);eps=u+F(1,2**64);err=2*u+u*u;mu=F(0);var=F(0);coupling=F(0);check('144 records',len(r['records'])==144)
    for i,row in enumerate(r['records']):
        prev,z=map(float.fromhex,raw['audit_paths'][0][i][2:]);word=struct.unpack('>Q',struct.pack('>d',z))[0];below=struct.unpack('>d',struct.pack('>Q',word-1))[0];above=struct.unpack('>d',struct.pack('>Q',word+1))[0];lo=(F(below)+F(z))/(2*F(prev));hi=(F(above)+F(z))/(2*F(prev));p=(lo+hi)/2;radius=(hi-lo)/2;check(f'cell {i}',row['mode']==i and row['previous']==F(prev) and row['rounded_product']==F(z) and row['fixed_p_interval']==[lo,hi] and row['proxy_zero']==p);check(f'normal first path {i}',F(1,2**1022)<F(prev)<=1 and F(1,2**1022)<F(z)<1);check(f'all-history p domain {i}',F(1,8)<lo<=hi<F(7,8));check(f'local conditional bound {i}',row['threshold_error_upper']==radius+err and row['finite_word_error_upper']==radius+err+eps);mu+=1-p;var+=p*(1-p);coupling+=radius+err+eps
    # Explicit interval arithmetic induction, 144 times, for both branches.
    previous_lower=F(1);previous_upper=F(1)
    for j in range(144):
        lower_product=previous_lower/8;upper_ratio=F(7,8)*(1+u);zero_lower=previous_lower*F(1,8)*(1-u);one_lower=previous_lower*(1-upper_ratio)*(1-u);one_upper=previous_upper*(1-F(1,8)*(1-u))*(1+u);check(f'induction {j}',lower_product>F(1,2**1022) and zero_lower>previous_lower/16 and one_lower>previous_lower/16 and one_upper<previous_upper and upper_ratio<1);previous_lower/=16
    check('normal invariant endpoint',previous_lower==r['all_history_previous_lower']==F(1,2**576) and r['minimum_intermediate_lower']==F(1,2**579)>F(1,2**1022));check('constant error bounds',r['unit_roundoff']==u and r['uniform_CDF_error_upper']==eps and r['multiply_divide_error_upper']==err);check('proxy moments',mu==r['proxy_mean'] and var==r['proxy_variance'] and coupling==r['full_law_coupling_upper']);check('bounded observable',all(0<=F((n-72)**2,5184)<=1 for n in range(145)) and r['center']==72 and r['scale']==5184);proxy=(var+(mu-72)**2)/5184;mom=read(ROOT/'results/iteration36_recovery/case_11.audit.json')['moments'];a,b=mom['mean_interval'];c,d=mom['second_moment_interval'];target=[(c-144*b+5184)/5184,(d-144*a+5184)/5184];lower=max(F(0),target[0]-proxy-coupling,proxy-target[1]-coupling);transfer=read(ROOT/'results/iteration35_panel/case_07.audit.json')['transfers']['-40']['all_measurement_TV_transfer_upper'];check('deterministic TV witness',target==r['target_observable_interval'] and proxy==r['proxy_observable_mean'] and lower==r['full_TV_deterministic_lower'] and max(F(0),lower-transfer)==r['shifted_target_TV_deterministic_lower']);(out/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');(out/'verification.json').write_text(json.dumps(dict(count=len(checks),passed=sum(c['passed'] for c in checks),verifier_sha256=sha(Path(__file__)),proof_sha256=sha(out/'proof.json'),scope='Exact rounding-cell, induction-inequality and moment checks; IEEE/source semantics stated in accompanying proof'),indent=2)+'\n');print('verified',len(checks))
if __name__=='__main__':main()
