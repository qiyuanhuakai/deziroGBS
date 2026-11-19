"""Prepare fresh best-CPU Frugal comparisons; retain earlier GPU results."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'experiments/iteration73_click_benchmark_v1.py').read_text().replace('results/iteration73_philox_pilot_v1/philox.dll','results/iteration74_fused_pilot_v2/philox.dll')
s=s.replace("large_methods=[('philox',1),('philox',4),('philox',8),('cpu64',2),('cpu64',3),('gpu32',2),('gpu32',3)]","large_methods=[('philox',1),('philox',4),('philox',8),('cpu64',2),('cpu64',3)]")
start=s.index("for kind,budget in [('cpu32',2),('cpu32',3)]:")
end=s.index('for count in (512,4096):',start)
s=s[:start]+s[end:];s=s.replace('for count in (512,4096):','for count in (4096,):').replace('large_sizes=[512,4096]','large_sizes=[4096]').replace('cpu32_diagnostic_count=512','cpu32_diagnostic_count=0')
s=s.replace('737001','745001').replace('737501','745501').replace('737000','745000')
s=s.replace('CPU32 large512 diagnostics are single runs, not median claims.','Large comparisons remeasure the best tested AVX2 FP64 CPU; prior GPU timings remain explicitly historical.')
with (ROOT/'experiments/iteration74_click_benchmark_v1.py').open('xb') as f:f.write(s.encode())
