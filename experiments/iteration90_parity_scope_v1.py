"""Correct one overbroad metadata sentence without changing executed evidence.
The specified two-mode squeezing transformation preserves N1-N2. Arbitrary
Gaussian unitaries need not do so; the exact PNR-TV comparison is only for the
stated squeezed-|k,0> conditional family. All numerical results are unchanged.
"""
from pathlib import Path
import argparse,json,hashlib,copy
ROOT=Path(__file__).resolve().parents[1]
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);source=ROOT/'results/iteration90_parity_gates_v1/summary.json';raw=json.loads(source.read_text());corrected=copy.deepcopy(raw)
for row in corrected['fock_comparisons']:
 assert 'Gaussian unitaries preserve charge' in row['scope'];row['scope']='Conditional squeezed-|k,0> family only: the specified two-mode squeezing preserves N1-N2, so the complete two-port PNR TV equals one minus the correct input number weight. Arbitrary Gaussian unitaries need not preserve this charge. This is not a full sampler result.'
restored=copy.deepcopy(corrected)
for old,new in zip(raw['fock_comparisons'],restored['fock_comparisons']):new['scope']=old['scope']
assert restored==raw
corrected['metadata_correction']=dict(scope=__doc__,original_summary=source.relative_to(ROOT).as_posix(),original_sha256=H(source),numerical_results_unchanged=True)
(out/'plan.json').write_text(json.dumps(dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in (Path(__file__),source)}),indent=2)+'\n',encoding='utf-8');(out/'summary.json').write_text(json.dumps(corrected,indent=2)+'\n',encoding='utf-8');(out/'manifest.json').write_text(json.dumps(dict(artifacts={p.name:H(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n',encoding='utf-8');print('scope corrected; all numerical results unchanged')
