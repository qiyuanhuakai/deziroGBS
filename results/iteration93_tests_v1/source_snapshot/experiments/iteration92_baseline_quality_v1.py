"""Fresh upstream PNR / Frugal click baselines on the same weak/strong TMSV laws.
The joint family has60 events: previous48 + two upstream4-event cases + two
Frugal2-event click cases. All previous bounds are recomputed at delta=.01/60.
"""
import argparse
from iteration92_pe_common_v1 import *
from iteration92_frugal_common_v1 import *
from iteration92_quality_v1 import reference,labels_for,confidence
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);N=65536;pilot=4096;delta=.01/60;core,runtime=load();frugal=load_frugal();save(out,'plan.json',dict(scope=__doc__,records=N,pilot_records=pilot,photon_cap=128,events=60,joint_failure_budget=.01,pe_seeds=[0x92004000,0x92004001],frugal_seeds=[0x92005000,0x92005001],pe_device='cpu',pe_threads=1,frugal_threads=1,frugal_cutoff=4,hashes={str(p.relative_to(ROOT)):H(p) for p in (Path(__file__),ROOT/'experiments/iteration92_pe_common_v1.py',ROOT/'experiments/iteration92_frugal_common_v1.py',ROOT/'results/iteration75_frugal_cpu_build_v1/frugal_cpu_fp64.dll')}));rows=[]
for j,r in enumerate((.6,.85)):
    case=dict(reference88=j,detected_modes=2,eta=1);u,target,ctarget,tail,keys=reference(case);state,U=setup_pe(u,r);counts,stats,meta=run_pe(state,U,N,0x92004000+j);errors=np.asarray(stats.censored,np.int32);np.savez_compressed(out/f'pe_raw_{j}.npz',counts=counts,errors=errors);save(out,f'pe_stats_{j}.json',vars(stats));labels,cl=labels_for(counts,errors,case,keys,len(target));pnr,ph=confidence(labels,target,N,pilot,delta,tail);click,ch=confidence(cl,ctarget,N,pilot,delta,tail);rows.append(dict(method='photonic-engines CPU v2',r=r,reference88=j,pnr=pnr,click=click,**meta));print(rows[-1],flush=True)
    bits,meta=run_frugal(frugal,frugal_covariance(r),N,0x92005000+j);assert meta['returncode']==0,meta;assert not bits[:,2:].any();np.savez_compressed(out/f'frugal_raw_{j}.npz',clicks=bits);cl=bits[:,0].astype(int)+2*bits[:,1];click,_=confidence(cl,ctarget,N,pilot,delta,tail);rows.append(dict(method='Frugal CPU fp64 K4 (two vacuum pads)',r=r,reference88=j,click=click,**meta));print(rows[-1],flush=True)
save(out,'summary.json',dict(passed=True,rows=rows,events=60,joint_failure_budget=.01));manifest(out)
