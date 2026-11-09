"""Check the analytic all-history enclosure against every recorded K=1 step."""
import sys,json,hashlib
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'experiments'));sys.set_int_max_str_digits(0)
from verify_iteration34 import decode
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    out=Path(__file__).parent;dest=out/'observed_validation.json';assert not dest.exists();proof=decode(json.loads((out/'proof.json').read_text()));raw=json.loads((ROOT/'results/iteration39_global/case_00.stdout.json').read_text());steps=0
    for path in raw['audit_paths']:
        for j,a in enumerate(path):
            q,v,z=map(lambda h:F(float.fromhex(h)),a[1:]);r=proof['records'][j];assert F(1,1<<(4*j))<=v<=1 and F(1,1<<(4*(j+1)))<=z<=v and abs(q-r['proxy_zero'])<=r['threshold_error_upper'];steps+=1
    dest.write_text(json.dumps(dict(steps_checked=steps,passed=steps,source_sha256=sha(Path(__file__)),dependencies={'results/iteration39_k1_control/proof.json':sha(out/'proof.json'),'results/iteration39_global/case_00.stdout.json':sha(ROOT/'results/iteration39_global/case_00.stdout.json')},scope='Observed-path consistency only; the separate induction supplies the all-history argument'),indent=2)+'\n');print('observed consistency',steps)
if __name__=='__main__':main()
