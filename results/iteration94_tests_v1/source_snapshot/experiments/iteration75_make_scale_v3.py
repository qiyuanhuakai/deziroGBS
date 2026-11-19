"""Separate complete CPU/fused comparisons from bounded slow GPU baseline attempts."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'experiments/iteration75_scale_click_v2.py').read_text(encoding='utf-8').replace("('cpu64',3),('gpu64',2)","('cpu64',3)").replace('Complete rerun after adapter dimension extension; no timings spliced from v1.','Complete CPU/fused rerun after GPU-budget interruption; no timings spliced from v1/v2.').replace('Frugal CPU and Frugal GPU.','Frugal CPU. GPU baselines are reported separately with an explicit budget.')
with (ROOT/'experiments/iteration75_scale_click_v3.py').open('xb') as f:f.write(s.encode())
