"""Predeclare twelve independent quality laws including the validated Hermite prefix."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'experiments/iteration75_pnr_quality_v1.py').read_text(encoding='utf-8').replace('Eight independent','Twelve independent').replace('32 one-sided','48 one-sided')
s=s.replace('from gbslab.native_deep_pnr import NativeDeepPNRSampler','from gbslab.native_prefix_pnr import NativePrefixPNRSampler\nfrom gbslab.native_deep_pnr import NativeDeepPNRSampler')
s=s.replace("runtime=[Path(shutil.which('g++')).parent]","models['prefix']=(NativePrefixPNRSampler,ROOT/'results/iteration75_prefix_pilot_v1/prefix.dll');runtime=[Path(shutil.which('g++')).parent]")
s=s.replace('math.log(3200)','math.log(4800)').replace('panels=[(0,kind,7) for kind in models]+[(1,kind,k) for kind in models for k in (3,5,7)]','panels=[(0,kind,7) for kind in ("deep","resplit")]+[(1,kind,k) for kind in ("deep","resplit") for k in (3,5,7)]+[(0,"prefix",31)]+[(1,"prefix",k) for k in (15,23,31)]').replace('0x7a000000000000','0x7c000000000000').replace('0x79000000000000','0x7d000000000000').replace('one_sided_bounds=32','one_sided_bounds=48')
compile(s,'quality75v2','exec')
with (ROOT/'experiments/iteration75_pnr_quality_v2.py').open('xb') as f:f.write(s.encode())
