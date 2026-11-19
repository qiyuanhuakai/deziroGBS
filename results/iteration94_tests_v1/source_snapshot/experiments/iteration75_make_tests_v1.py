"""Enable new deep/resplit interfaces in the full historical suite."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'experiments/iteration74_tests_v3.py').read_text(encoding='utf-8').replace('iteration71/72/73/74 APIs','iteration71/72/73/74/75 APIs')
s=s.replace('sources=sorted(',"os.environ['GBSLAB_DEEP_PNR_LIBRARY']=str(ROOT/'results/iteration75_deep_pilot_v1/deep.dll')\nos.environ['GBSLAB_RESPLIT_PNR_LIBRARY']=str(ROOT/'results/iteration75_resplit_pilot_v1/resplit.dll')\nsources=sorted(")
s=s.replace('skipped_cases=sum(',"deep_pnr_passed=sum(s.startswith('test_native_deep_pnr.') for s in result.successes),resplit_pnr_passed=sum(s.startswith('test_native_resplit_pnr.') for s in result.successes),skipped_cases=sum(")
with (ROOT/'experiments/iteration75_tests_v1.py').open('xb') as f:f.write(s.encode())
