"""Exact enumeration of legacy PNR stream identifiers; no sampling claims.
Duplicate inputs to mix imply identical mt19937_64 engines. The old mixed
baseline has zero added noise, so its noise engine is instantiated but unused.
Waiting/detector overlap alone invalidates sector/record stream separation.
"""
from pathlib import Path
import json,hashlib,collections,argparse
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,x):
 with (OUT/n).open('xb') as f:f.write((json.dumps(x,indent=2)+'\n').encode())
sources=[Path(__file__),ROOT/'experiments/iteration83_full_followup_quality_v1.py',ROOT/'native/gaussian_pnr_polynomial73_v1.cpp',ROOT/'experiments/iteration84_correlated_pnr_pilot_v1.py',ROOT/'native/gaussian_pnr_correlated84_v1.cpp',ROOT/'experiments/iteration84_full_quality_v1.py'];save('plan.json',dict(scope=__doc__,hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources}));groups=collections.defaultdict(list)
for sector in range(2):
 for row in range(1024):
  seed=832100+sector
  for purpose,argument in [('waiting',seed+2*row),('detector',seed+2*row+1),('noise',seed+0x81c6f2d54a970e31+row)]:groups[argument%(1<<64)].append(dict(sector=sector,row=row,purpose=purpose,consumed=(purpose!='noise')))
overlaps=[dict(mix_input=str(k),uses=v) for k,v in groups.items() if len(v)>1];consumed=[r for r in overlaps if any(v['consumed'] for v in r['uses'])];save('overlaps.json',overlaps);save('summary.json',dict(confirmed=True,records_per_sector=1024,total_engine_initializations=6144,unique_engine_inputs=len(groups),duplicate_groups=len(overlaps),consumed_wait_detector_duplicate_groups=len(consumed),noise_duplicate_groups=len(overlaps)-len(consumed),example_consumed=consumed[:6],consequence='Adjacent combined records share consumed engines; spectra also share waiting/detector engines. Original iid DKW coverage and independently combined spectral law are unsupported.',withdrawn_iteration83_claims=['PNR complete click TV lower 0.2950406587837476 at nominal joint99','PNR complete photon TV lower 0.3072123538158624 at nominal joint99','PNR experimental full-bit TV lower 0.4469147795128255 at nominal joint99'],unaffected_scope='Deterministic Gaussian moments/core decompositions and scalar arithmetic checks do not depend on this RNG. GPU uses Philox(seed,row), not the legacy PNR affine engine assignment. Separate direct-click claims remain subject to their own stated assumptions.'))
save('manifest.json',dict(artifacts={p.name:H(p) for p in OUT.iterdir() if p.is_file()}));print('Confirmed consumed duplicate groups',len(consumed),'unused-noise duplicate groups',len(overlaps)-len(consumed),flush=True)
