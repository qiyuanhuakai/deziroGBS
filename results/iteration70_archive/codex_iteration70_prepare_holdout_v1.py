from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
s=(ROOT/'.git/codex_iteration70_noise_screen_v2.py').read_text(encoding='utf-8')
s=s.replace('noise_screen_v2_20261110','noise_holdout_v1_20261110').replace('flow_pilot_v1_20261110','parallel_pilot_v1_20261110')
s=s.replace('count=100000;theta_values=[0.,.1,.25,.5,.75,1.]','count=1000000;theta_values=[0.,.25]')
s=s.replace('702001','706001').replace("ROOT/'.git/iteration70_flow_v1.cpp'", "ROOT/'.git/iteration70_parallel_v1.cpp',ROOT/'.git/iteration70_flow_v2.cpp'")
s=s.replace("lib.g70_sample.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]", "lib.g70_sample_parallel.argtypes=[C.c_void_p,C.c_int,C.c_uint64,C.c_int,C.c_int,C.c_void_p,C.c_int,DP,DP,UP,DP,C.c_char_p,C.c_int]")
s=s.replace('stats=np.zeros(18)', 'stats=np.zeros(20)')
s=s.replace('lib.g70_sample(handle,count,706001+case*100,3,1,ptr(mean)', 'lib.g70_sample_parallel(handle,count,706001+case*100,3,1,None,16,ptr(mean)')
s=s.replace('Screen exact Gaussian input decompositions', 'Fresh independent seeds after the frozen 100k screen; validate centered, minimum-core theta=0, and theta=.25 input decompositions')
s=s.replace('Screening only; a fresh validation cohort is required before treating selected theta as improved.', 'Independent-seed validation of the previously screened settings; targets and isometries remain fixed, so no generalization claim to all inputs. Parallel sampling uses 16 CPU workers.')
p=ROOT/'.git/codex_iteration70_noise_holdout_v1.py';assert not p.exists();compile(s,str(p),'exec');p.write_bytes(s.encode())
