"""Enumerate every upstream Frugal threshold on the exact small benchmark inputs."""
import argparse
import sys
from iteration93_baseline_common_v1 import ROOT,Path,C,H,save,manifest,read,np,os,shutil,Baseline93
sys.path.insert(0,str(ROOT/'src'))
from gbslab.frugal_law import full_law

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
out=p.parse_args().output.resolve();out.mkdir(parents=True,exist_ok=False)
prefix=ROOT/'results/iteration81_prefix_build_v1/prefix.dll'
base=ROOT/'results/iteration93_reference_v2'
panel=ROOT/'results/iteration95_ensemble_v2'
files=[Path(__file__),prefix,ROOT/'src/gbslab/frugal_law.py',ROOT/'native/frugal_prefix81_adapter_v1.cpp',
       panel/'summary.json',*[base/f'{name}_{j}.npz' for j in (0,2) for name in ('law','input')]]
save(out,'plan.json',dict(hashes={f.relative_to(ROOT).as_posix():H(f) for f in files},
    scope='entire ideal independent uint64-word law of pinned FP64 conditional sampler; prior single-thread timing measured for identical input, 2048 records',
    cases={'pure4':[2,4],'pure8':[2,5]},sample_call='Audited prefix DLL checks upstream official samples lane by lane.'))
runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
lib=C.CDLL(str(prefix));tree=lib.fc81_tree
D=np.ctypeslib.ndpointer(dtype=float,flags='C_CONTIGUOUS')
tree.argtypes=[C.c_int]*4+[D]*5+[C.c_char_p,C.c_int]
s=read(panel/'summary.json');rows=[]
for case,cutoffs in [(0,(2,4)),(2,(2,5))]:
    z=dict(np.load(base/f'input_{case}.npz'));m=len(z['u'])
    target=np.load(base/f'law_{case}.npz')['click']
    for cutoff in cutoffs:
        model=Baseline93(dict(family='frugal',case=case,cutoff=cutoff,threads=1),z)
        condition=np.empty(2**m-1);prev=condition.copy();proposed=condition.copy()
        stats=np.zeros(3);message=C.create_string_buffer(4096)
        assert tree(m,cutoff,m,1,model.cov,condition,prev,proposed,stats,message,len(message))==0,message.value
        model.close()
        nodes=[dict(prefix=''.join(str((mask>>i)&1) for i in range(depth)),
                    threshold_hex=float(condition[2**depth-1+mask]).hex())
               for depth in range(m) for mask in range(2**depth)]
        exact=full_law(m,nodes,model='uint64')
        distribution=np.array([float(v) for v in exact['distribution']])
        tv=float(abs(distribution-target).sum()/2)
        timing=[v['seconds']/v['n'] for v in s['timings'] if v['case']==case and v['variant']==16+cutoff]
        assert len(timing)==3
        rows.append(dict(case=case,cutoff=cutoff,full_tv=tv,seconds_per_record=float(np.median(timing)),
            nonfinite=len(exact['reachable_nonfinite_prefixes']),saturated=len(exact['reachable_saturated_prefixes']),
            prefix_checks=int(stats[2])))
        np.savez_compressed(out/f'case_{case}_k_{cutoff}.npz',distribution=distribution,target=target,
                            conditional=condition,previous=prev,proposed=proposed,statistics=stats)
save(out,'summary.json',rows);manifest(out)
