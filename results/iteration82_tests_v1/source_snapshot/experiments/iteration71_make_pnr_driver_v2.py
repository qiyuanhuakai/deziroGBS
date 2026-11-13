from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'experiments/iteration71_pnr_branch_pilot_v1.py').read_text(encoding='utf-8')
s=s.replace('gaussian_pnr_branches_v1.cpp','gaussian_pnr_branches_v2.cpp')
s=s.replace('pnr71_sample','pnr72_sample')
s=s.replace('C.c_int,C.c_double,DP,DP,UP','C.c_int,C.c_double,C.c_double,DP,DP,UP')
s=s.replace('16,budget,eps,ptr(mean)','16,budget,eps,1e6,ptr(mean)')
s=s.replace('20000','100000').replace('712001','713001').replace('712901','713901')
s=s.replace('epsilon=[.05,.01]','epsilon=[.01,.001],condition_limit=1e6')
s=s.replace('enumerate((.05,.01))','enumerate((.01,.001))')
s=s.replace('Finite-tap Gaussian difference approximates photon subtraction;', 'Condition-aware analytic moment compression at absolute signed weight limit 1e6. Finite-tap Gaussian difference approximates photon subtraction;')
with (ROOT/'experiments/iteration71_pnr_branch_pilot_v2.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
