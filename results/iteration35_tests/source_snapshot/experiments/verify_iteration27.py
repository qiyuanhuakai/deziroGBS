"""Independent artifact checks, probability bookkeeping, and source-byte audit."""
import argparse
import json
import math
from pathlib import Path
import shutil
import sys
import zlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from iteration23_support import digest,save


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--external',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False); src=Path(__file__); initial=digest(src); shutil.copy2(src,out/src.name)
    checks=[]
    def check(name,value): checks.append(dict(name=name,passed=bool(value)))
    def read(path): return json.loads(Path(path).read_text(encoding='utf8'))
    panel=ROOT/'results/iteration27_panel'; plan=read(panel/'plan.json'); rows=read(panel/'rows.json'); summary=read(panel/'summary.json')
    check('44 planned and passed',summary['planned']==summary['finished']==summary['passed']==len(rows)==44)
    for name,sha in plan['source_hashes'].items():
        check('current and frozen source '+name,digest(ROOT/name)==digest(panel/'source_snapshot'/name)==sha)
    for name,sha in plan['dependencies'].items(): check('historical dependency '+name,digest(ROOT/name)==sha)
    for record in plan['external_members']:
        path=a.external/record['local_name']; data=path.read_bytes()
        check('external sha '+record['local_name'],digest(path)==record['sha256'])
        check('external crc '+record['local_name'],f'{zlib.crc32(data):08x}'==record['crc32'])
    error_full=[]; error_mp=[]; complement=[]
    for i,row in enumerate(rows):
        name=f'case_{i:02d}'; data=np.load(panel/(name+'.npz'),allow_pickle=False); ref=np.load(panel/(name+'.reference.npz'),allow_pickle=False)
        p=data['probabilities']; patterns=data['patterns']; m=row['modes']; total=math.fsum(p)
        check(name+' patterns',len(p)==m*(m+1)//2+1 and len({tuple(x) for x in patterns})==len(p))
        check(name+' nonnegative',np.all(np.isfinite(p)) and np.all(p>=0))
        check(name+' normalization bookkeeping',abs(total-row['selected_mass'])<1e-14 and abs(total+row['remainder_probability']-1)<1e-12)
        check(name+' masses',abs(p[0]-row['vacuum'])<1e-14 and abs(sum(p[1:1+m])-row['one_click_mass'])<1e-14 and abs(math.fsum(p[1+m:])-row['two_click_mass'])<1e-12)
        check(name+' no correction',row['probability_corrections']==0)
        if row['panel']=='small':
            ids=[sum(1<<int(k) for k in pair if k>=0) for pair in patterns]
            ef=float(np.max(abs(p-ref['full_threshold'][ids]))); eh=float(np.max(abs(p-ref['high_precision'])))
            er=float(np.max(abs(ref['high_precision']-ref['independent_high_precision'])/np.maximum(abs(ref['independent_high_precision']),1e-70)))
            check(name+' full reference',ef<2e-12 and abs(ef-row['float_vs_full_max_absolute'])<1e-25)
            check(name+' high precision reference',eh<2e-12 and er<2e-12)
            error_full.append(ef); error_mp.append(eh)
        else:
            err=float(np.max(abs(p[ref['indices']]-ref['independent_complement']))/p[0]); complement.append(err)
            check(name+' complement reference',len(ref['indices'])==49 and err<2e-8 and abs(err-row['complement_max_error_divided_by_vacuum'])<1e-25)
    audit=read(panel/'covariance_audit.json'); witness=np.load(panel/'negative_witnesses.npz',allow_pickle=False)
    for name in ('author_pure','candidate_pure'):
        value=audit['mixed_minus_'+name]
        check(name+' eigenvalue agreement',abs(value['minimum_eigenvalue']-value['scipy_minimum_eigenvalue'])<2e-11)
        check(name+' negative witness',float(value['rayleigh_80_digit'])<-1e-5 and abs(np.linalg.norm(witness[name])-1)<1e-12)
    source=read(ROOT/'results/iteration27_source/summary.json'); check('range audit passed',source['status']=='passed' and source['selected']==5 and not source['whole_archive_md5_verified'])
    start=read(ROOT/'results/iteration27_source/plan.json')['source_sha256']; check('range source snapshot',digest(ROOT/'experiments/iteration27_source.py')==digest(ROOT/'results/iteration27_source/iteration27_source.py')==start)
    check('verifier unchanged',digest(src)==initial)
    save(out/'validation.json',dict(checks=checks,count=len(checks),passed=sum(x['passed'] for x in checks),verifier_sha256=initial,dependencies=plan['dependencies'],maximum_small_full_error=max(error_full),maximum_small_high_precision_error=max(error_mp),maximum_large_complement_error_divided_by_vacuum=max(complement)))
    print(json.dumps(dict(count=len(checks),passed=sum(x['passed'] for x in checks),maximum_small_full_error=max(error_full),maximum_small_high_precision_error=max(error_mp),maximum_large_complement_error_divided_by_vacuum=max(complement))))
    assert all(x['passed'] for x in checks)


if __name__=='__main__': main()
