"""Three strictly serial rotations on identical four-port padded TMSV inputs.
One-thread resource comparisons and separate16-thread throughput. All include
model/core preparation, sampling, count-to-click conversion and destruction.
Imports and DLL loading are excluded for every method. Repetitions deliberately
replay fixed control records, never contribute additional quality N.
"""
import argparse,statistics
from iteration92_pe_common_v1 import *
from iteration92_frugal_common_v1 import *
from iteration89_common_v1 import NativeDirectClickSampler,small_input,tree,puretree
import iteration92_common_v2 as cc
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);core,runtime=cc.load();fg=load_frugal();oldlib=ROOT/'results/iteration89_gates_v1/puretree_probe.dll';preloaded=C.CDLL(str(oldlib));N=4096;methods=['core1','pe1','frugal1','r3_1','r5_1','core16','r3_16','r5_16'];save(out,'plan.json',dict(scope=__doc__,records=N,repetitions=3,methods=methods,r_values=[.6,.85],quality_data_not_reused=True,source_sha256=H(Path(__file__)),core_sha256=H(ROOT/'results/iteration92_build_v2/core.dll'),old_sha256=H(oldlib)));rows=[]
for case,r in enumerate((.6,.85)):
    u=np.vstack([np.eye(2,dtype=complex),np.zeros((2,2),complex)]);np.savez_compressed(out/f'input_{case}.npz',u=u,r=r)
    for rep in range(3):
        order=methods[rep:]+methods[:rep]
        for method in order:
            start=time.perf_counter();meta={};errors=None
            if method.startswith('core'):
                raw,meta=cc.sample(core,u.copy(),cc.fock_core(0),r,N,0x92006000+case,threads=int(method[4:]));counts=raw['counts'];errors=raw['errors'];bits=(counts>0).astype(np.uint8);assert not errors.any(),meta
            elif method=='pe1':
                state,U=setup_pe(u.copy(),r);counts,stats,meta=run_pe(state,U,N,0x92006100+case);assert meta['censored']==0,meta;bits=(counts>0).astype(np.uint8)
            elif method=='frugal1':
                bits,meta=run_frugal(fg,frugal_covariance(r),N,0x92006200+case,threads=1);assert meta['returncode']==0,meta;counts=None
            else:
                depth=int(method[1]);threads=int(method.split('_')[1]);q=small_input(dict(name='paired',squeezing=r,eta=1));q['v']=np.vstack([1j*q['v'],np.zeros((2,2),complex)])
                with NativeDirectClickSampler(q['v'],q['x'],oldlib,runtime_dirs=[Path(shutil.which('g++')).parent]) as model:
                    handle,destroy=tree(model)
                    try:raw,meta=puretree(model,handle,q,depth,N,0x92006300+16*case+depth,threads=threads)
                    finally:destroy(handle)
                counts=raw['counts'];errors=raw['errors'];bits=(counts>0).astype(np.uint8)
            elapsed=time.perf_counter()-start;data=dict(clicks=bits)
            if counts is not None:data['counts']=counts
            if errors is not None:data['errors']=errors
            np.savez_compressed(out/f'raw_{case}_{rep}_{method}.npz',**data);row=dict(case=case,r=r,rep=rep,method=method,records=N,seconds=elapsed,metadata=meta);rows.append(row);save(out,f'row_{case}_{rep}_{method}.json',row);print(case,rep,method,elapsed,flush=True)
medians=[dict(case=j,r=r,method=m,seconds=statistics.median(row['seconds'] for row in rows if row['case']==j and row['method']==m)) for j,r in enumerate((.6,.85)) for m in methods];save(out,'summary.json',dict(passed=True,rows=rows,medians=medians,scope=__doc__));manifest(out)
