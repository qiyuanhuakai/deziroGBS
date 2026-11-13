"""Preserve the first run and make a complete uncontended timing rerun."""
from pathlib import Path
import json,hashlib
root=Path(__file__).resolve().parents[1]
old=root/'experiments/iteration75_lowrank_v1.py'
new=root/'experiments/iteration75_lowrank_v2.py'
text=old.read_text(encoding='utf-8').replace('Fixed two squeezed inputs, 144 outputs:', 'Complete uncontended rerun after possible final-panel/test overlap. Fixed two squeezed inputs, 144 outputs:',1)
with new.open('x',encoding='utf-8') as f:f.write(text)
out=root/'results/iteration75_lowrank_overlap_note_v1';out.mkdir(exist_ok=False)
with (out/'receipt.json').open('x',encoding='utf-8') as f:json.dump(dict(status='possible_final_panel_overlap',reason='Full suite was launched before observing the lowrank process termination. Preserve v1; replace every official timing by a complete v2 run after the suite terminates. No v1/v2 timing splicing.',original_driver_sha256=hashlib.sha256(old.read_bytes()).hexdigest(),rerun_driver_sha256=hashlib.sha256(new.read_bytes()).hexdigest()),f,indent=2)
