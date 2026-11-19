"""Raise only MSVC's constant-evaluation step budget for N=384 templates."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'experiments/iteration75_frugal_gpu_build_v1.py').read_text(encoding='utf-8').replace('Build expanded-dimension FP64 v1:','Build expanded-dimension FP64 v2, higher constexpr evaluation budget:').replace("'/MD,/utf-8,/openmp:llvm'","'/MD,/utf-8,/openmp:llvm,/constexpr:steps100000000'")
with (ROOT/'experiments/iteration75_frugal_gpu_build_v2.py').open('xb') as f:f.write(s.encode())
s=(ROOT/'experiments/iteration75_scale_click_v2.py').read_text(encoding='utf-8').replace('iteration75_frugal_gpu_build_v1/frugal_fp64.dll','iteration75_frugal_gpu_build_v2/frugal_fp64.dll')
# v2 scale script has never been executed; finalize its binary path before use.
(ROOT/'experiments/iteration75_scale_click_v2.py').write_bytes(s.encode())
