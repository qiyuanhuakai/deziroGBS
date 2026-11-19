"""Separate exact stored-tensor law, actual sampler law and physical target."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys,json,shutil,time,traceback
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from iteration22_external import digest,save
from iteration33_run import encode
from iteration35_run import certificate
from iteration36_run import envelope_records
from iteration45_mps_trace import enumerate_trace,public_calibration
from gbslab.gaussian_schmidt_mps import build_schmidt_core
from gbslab.certified_gaussian import ExactPairingModel
from gbslab.gaussian_envelope import GaussianEnvelope
from gbslab.exact_mps_law import represented_threshold_law
from gbslab.interval_completion import physical_tv_interval
ROOT=Path(__file__).resolve().parents[1];sys.set_int_max_str_digits(0)
def main():
    out=ROOT/'results/iteration45_mps';out.mkdir(exist_ok=False);names=['research/iteration45_panel_plan.md',*[f'experiments/{n}.py' for n in ('iteration45_mps','iteration45_mps_trace','iteration22_external','iteration33_run','iteration35_run','iteration36_run')],*[p.relative_to(ROOT).as_posix() for p in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:p=out/'source_snapshot'/n;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,p)
    jobs=[dict(strength=s,dimension=d,rank=r) for s in ('moderate','strong') for d in (4,6,8) for r in (2,4,8)];save(out/'plan.json',dict(jobs=jobs,source_hashes=hashes,dependencies={},numpy_version=np.__version__,budgets=dict(build_seconds=120,max_elements=1000000,max_fock_states=10000,max_trace_nodes=20000)));H=np.array([[(-1)**((i&j).bit_count()) for j in range(4)] for i in range(4)]);U=H@np.diag([1,1j,-1,1])@H/4;targets={};pairings={}
    for strength,values in (('moderate',[.25,.375,.125,.25]),('strong',[.5,.75,.375,.625])):
        B=(U*np.array(values))@U.T;assert np.array_equal(B,B.T);pairings[strength]=B;model=ExactPairingModel(B,np.zeros((8,0)));env=GaussianEnvelope(model);law=env.law(tuple(range(4)));targets[strength]=law['intervals'];np.savez_compressed(out/(strength+'.npz'),B=B,U=U,parameters=np.array(values));save(out/(strength+'.target.json'),encode(dict(law=law,enclosures=envelope_records(env),core=certificate(model.core),center_numerators=model.center_numerators.tolist(),center_denominator=model.center_denominator,operator_radius=model.operator_radius)))
    rows=[]
    for i,job in enumerate(jobs):
        start=time.perf_counter();row=dict(sequence=i,**job);print('start mps',i,flush=True)
        try:
            tensors,info=build_schmidt_core(pairings[job['strength']],job['dimension'],job['rank'],max_seconds=120,max_elements=1000000);np.savez_compressed(out/f'case_{i:02d}.npz',**{f'tensor_{j}':a for j,a in enumerate(tensors)});represented=represented_threshold_law(tensors,max_states=10000);sampling=enumerate_trace(tensors);calibration=public_calibration(tensors,sampling['trace'],451000+i);target=targets[job['strength']]+[(F(0),F(0))];state=represented['distribution']+[F(0)];state_tv=physical_tv_interval(state,target);sampling_tv=physical_tv_interval(sampling['word_law'],target);arithmetic=sum(abs(x-y) for x,y in zip(state,sampling['word_law']))/2;quantization=sum(abs(x-y) for x,y in zip(sampling['real_law'],sampling['word_law']))/2
            serial_info=json.loads(json.dumps(info,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else (_ for _ in ()).throw(TypeError(type(x).__name__))));save(out/f'case_{i:02d}.json',encode(dict(job=job,construction=serial_info,represented=represented,sampling=sampling,calibration=calibration,state_physical_TV_interval=state_tv,sampler_physical_TV_interval=sampling_tv,arithmetic_TV=arithmetic,quantization_TV=quantization)));row.update(status='completed',state_TV=[float(x) for x in state_tv],sampler_TV=[float(x) for x in sampling_tv],arithmetic_TV=float(arithmetic),quantization_TV=float(quantization),failure_probability=float(sampling['word_law'][-1]),trace_nodes=sampling['nodes'],fock_states=represented['fock_states'],construction_seconds=info['total_seconds'],stored_complex_elements=info['stored_complex_elements'],max_bond=info['max_bond_actual'])
        except Exception:row.update(status='failed',traceback=traceback.format_exc())
        row['seconds']=time.perf_counter()-start;rows.append(row);save(out/'rows.json',rows);print('end mps',json.dumps(row),flush=True)
    assert all(digest(ROOT/n)==s for n,s in hashes.items());save(out/'summary.json',dict(planned=18,finished=len(rows),completed=sum(r['status']=='completed' for r in rows),unchanged_sources=True))
if __name__=='__main__':main()
