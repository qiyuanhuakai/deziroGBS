"""Canonical stored layout; every MPS judged against the exact shared V."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import sys,json,shutil,time,traceback,gzip,hashlib
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from verify_iteration34 import decode
from iteration45_mps_trace import enumerate_trace,public_calibration
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.exact_mps_law import represented_threshold_law
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)

def main():
    out=ROOT/'results/iteration52_mps';out.mkdir(exist_ok=False);names=['research/iteration52_mps_plan.md',*[f'experiments/{n}.py' for n in ('iteration52_mps','iteration45_mps_trace','iteration22_external','iteration33_run','verify_iteration34')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names};deps={};sources=[]
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    sources={}
    for it in (46,47):
        manifest=json.loads((ROOT/f'results/iteration{it}_manifest.json').read_text());deps[f'results/iteration{it}_manifest.json']=digest(ROOT/f'results/iteration{it}_manifest.json')
        for index in range(4,8):
            name=f'results/iteration{it}_inputs/case_{index:02d}.json';deps[name]=digest(ROOT/name);assert deps[name]==manifest['artifacts'][name];sources[(it,index)]=decode(json.loads((ROOT/name).read_text()));name=f'results/iteration{it}_inputs/case_{index:02d}.npz';deps[name]=digest(ROOT/name);assert deps[name]==manifest['artifacts'][name]
    jobs=[dict(source_round=it,case=index,dimension=d,rank=32) for it in (46,47) for index in range(4,8) for d in (3,4)]+[dict(source_round=it,case=index,dimension=4,rank=16) for it in (46,47) for index in (4,6)];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies=deps,numpy_version=np.__version__,layout='C-contiguous before every audit',construction_seconds_cap=180,construction_elements_cap=1000000,fock_states_cap=65536,audit_nodes_cap=100000));rows=[]
    for i,job in enumerate(jobs):
        case_started=time.perf_counter();milestones={'start':case_started};row=dict(sequence=i,**job);print('start MPS',i,flush=True)
        try:
            source=sources[(job['source_round'],job['case'])];n=source['job']['modes'];B=np.load(ROOT/f"results/iteration{job['source_round']}_inputs/case_{job['case']:02d}.npz")['B'];built,info=build_schmidt_core(B,job['dimension'],job['rank'],max_seconds=180,max_elements=1000000);milestones['construction']=time.perf_counter();tensors=[np.ascontiguousarray(a) for a in built];np.savez_compressed(out/f'case_{i:02d}.npz',**{f'tensor_{j}':a for j,a in enumerate(tensors)});stored=np.load(out/f'case_{i:02d}.npz');assert all(np.array_equal(a,stored[f'tensor_{j}']) and a.strides==stored[f'tensor_{j}'].strides for j,a in enumerate(tensors));milestones['tensor_storage']=time.perf_counter();represented=represented_threshold_law(tensors,max_states=65536);milestones['exact_state']=time.perf_counter();sampling=enumerate_trace(tensors,max_nodes=100000);milestones['full_tree']=time.perf_counter();calibration=public_calibration(tensors,sampling['trace'],521000+i);milestones['public_replay']=time.perf_counter();target=source['target']['target_intervals']+[(F(0),F(0))];state=represented['distribution']+[F(0)];state_tv=physical_tv_interval(state,target);sampler_tv=physical_tv_interval(sampling['word_law'],target);arithmetic=sum(abs(a-b) for a,b in zip(state,sampling['word_law']))/2;quantization=sum(abs(a-b) for a,b in zip(sampling['real_law'],sampling['word_law']))/2;serial=json.loads(json.dumps(info,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError())))
            trace=sampling.pop('trace');shards=[];sizes=[]
            for offset in range(0,len(trace),2000):
                filename=f'traces/case_{i:02d}/chunk_{offset//2000:03d}.json.gz';path=out/filename;path.parent.mkdir(parents=True,exist_ok=True);raw=(json.dumps(encode(trace[offset:offset+2000]),indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8');packed=gzip.compress(raw,compresslevel=6,mtime=0);path.write_bytes(packed);shards.append(filename);sizes.append(dict(path=filename,raw_bytes=len(raw),compressed_bytes=len(packed),raw_sha256=hashlib.sha256(raw).hexdigest()))
            sampling['trace_shards']=shards;sampling['trace_storage']=sizes
            save(out/f'case_{i:02d}.json',encode(dict(job=job,construction=serial,stored_strides=[a.strides for a in tensors],represented=represented,sampling=sampling,calibration=calibration,state_physical_TV_interval=state_tv,sampler_physical_TV_interval=sampler_tv,arithmetic_TV=arithmetic,quantization_TV=quantization)));row.update(status='completed',state_TV=[float(x) for x in state_tv],sampler_TV=[float(x) for x in sampler_tv],arithmetic_TV=float(arithmetic),failure=float(sampling['word_law'][-1]),trace_nodes=sampling['nodes'],fock_states=represented['fock_states'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        milestones['stored_result']=time.perf_counter();times=list(milestones.items());row['seconds']=times[-1][1]-case_started;row['phase_seconds']={b[0]:b[1]-a[1] for a,b in zip(times,times[1:])};row['clock_milestones']=milestones;rows.append(row);save(out/'rows.json',rows);print('end MPS',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items()) and all(digest(ROOT/n)==s for n,s in deps.items());save(out/'summary.json',dict(planned=20,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True,unchanged_dependencies=True))
if __name__=='__main__':main()
