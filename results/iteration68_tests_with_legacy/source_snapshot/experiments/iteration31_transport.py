"""Compare real kernel calls with independent native-process hexadecimal output."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import argparse,ctypes as ct,json,shutil,struct,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'tests')]
from gbslab.wire_forest import WireForestKernel,BYTE,_Conditional
from gbslab import batched_forest as legacy
from test_batched_forest_numerics import gaussian_fixture
from iteration28_codec import parse_hex
from iteration23_support import digest,save

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--library',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    names=['experiments/iteration31_transport.py','experiments/iteration28_codec.py','tests/test_batched_forest_numerics.py',*[x.relative_to(ROOT).as_posix() for x in (ROOT/'src').rglob('*.py')]];hashes={n:digest(ROOT/n) for n in names}
    for n in names:
        dest=out/'source_snapshot'/n;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/n,dest)
    shutil.copy2(ROOT/'research/iteration31_panel_plan.md',out/'frozen_plan.md');build=json.loads((a.library.parent/'build.json').read_text());env=dict(os.environ);env['PATH']=os.pathsep.join([str(Path(x['path']).parent) for x in build['runtime_dependencies']]+[env['PATH']])
    def case(name,V,z=(),r=(),mode=None,f=(),edges=(),policy=2,operation=1,expected=0):return dict(name=name,V=V,z=z,r=r,mode=V.shape[0]//2-1 if mode is None else mode,f=f,edges=edges,policy=policy,operation=operation,expected_status=expected)
    jobs=[case('thermal_'+str(policy),1.5*np.eye(8),f=(0,1,2),edges=((0,1),(1,2)),policy=policy) for policy in (0,1,2)]
    gaussian=gaussian_fixture(6).V
    jobs.extend([case('gaussian',gaussian,z=(3,),r=(4,1),f=(0,2),edges=((0,2),)),case('gaussian_raw_quad',gaussian,z=(3,),r=(4,1),f=(0,2),edges=((0,2),),policy=1,operation=0),case('vacuum',np.eye(8))])
    for name,b,policy,expected in [('thermal_refined',8,2,0),('thermal_ld_failure',8,0,2),('quad_exhausted',40,2,2)]:jobs.append(case(name,(1+2**-b)*np.eye(16),r=(3,4,5,6),f=(0,1,2),edges=((0,1),(1,2)),policy=policy,expected=expected))
    jobs.extend([case('endpoint_failure',1e17*np.eye(2),expected=2),case('argument_failure',1.5*np.eye(8),z=(0,),r=(0,),expected=1),case('common_zero_underflow',1e6*np.eye(200),z=tuple(range(99)))])
    save(out/'plan.json',dict(source_hashes=hashes,library_sha256=digest(a.library),reference_sha256=digest(a.reference),jobs=[{k:v for k,v in j.items() if k!='V'} for j in jobs]));records=[];checks=[]
    def check(name,condition):
        checks.append(dict(name=name,passed=bool(condition)))
        if not condition:save(out/'checks.failed.json',checks);raise AssertionError(name)
    for index,job in enumerate(jobs):
        V=job['V'];m=len(V)//2;order=tuple(range(m));np.savez_compressed(out/f'case_{index:02d}.npz',covariance=V)
        tokens=[m,job['mode'],len(job['z']),len(job['r']),len(job['f']),len(job['edges']),job['policy'],job['operation'],float(1e-6).hex(),*[float(x).hex() for x in V.ravel()],*order,*job['z'],*job['r'],*job['f'],*[x for edge in job['edges'] for x in edge]];payload=' '.join(map(str,tokens))+'\n';(out/f'case_{index:02d}.input.txt').write_text(payload)
        direct=json.loads(subprocess.check_output([str(a.reference.resolve()),'direct','0'],input=payload,text=True,env=env));record=dict(name=job['name'],direct=direct,wire=[])
        check(job['name']+' intended status',direct['status']==job['expected_status'])
        for controls in (0,1,2):
            wire=json.loads(subprocess.check_output([str(a.reference.resolve()),'wire',str(controls)],input=payload,text=True,env=env));record['wire'].append(wire)
            check(f'{index}/{controls} status',wire['status']==direct['status']);check(f'{index}/{controls} x87 restore',wire['control_before']==wire['control_after']);check(f'{index}/{controls} SSE restore',wire['mxcsr_controls_before']==wire['mxcsr_controls_after'])
            for key in ('longdouble_queries','quad_queries','auto_refinements'):check(f'{index}/{controls} {key}',wire[key]==direct[key])
            if direct['status']==0:
                for key in ('values','indicators')+ (('scalars',) if job['operation'] else ()):
                    check(f'{index}/{controls} exact {key}',wire[key]==[parse_hex(v).hex() for v in direct[key]])
                for key in ('exact_zero',)+ (('probability_bits','precision_used','refinement_attempted','zero_score_reasons','first_error') if job['operation'] else ()):check(f'{index}/{controls} {key}',wire[key]==direct[key])
        with WireForestKernel(V,library=a.library) as k:
            buffers,q,v,e,z=k._wire_request(job['z'],job['r'],job['mode'],job['f'],job['edges']);result=_Conditional();error=ct.create_string_buffer(512);before=(k._wire.scalar_wire_control_word(),k._wire.bfw_mxcsr()&0xffc0)
            if job['operation']:
                from gbslab.wire_forest import encode_dyadic
                guard=(ct.c_ubyte*24).from_buffer_copy(encode_dyadic(1e-6));status=k._wire.bfw_probability(k._handle,ct.byref(q),job['policy'],guard,24,v,e,z.ctypes.data_as(BYTE),len(z),ct.byref(result),error,len(error))
            else:status=k._wire.bfw_query(k._handle,ct.byref(q),job['policy'],0,v,e,z.ctypes.data_as(BYTE),len(z),error,len(error))
            after=(k._wire.scalar_wire_control_word(),k._wire.bfw_mxcsr()&0xffc0);check(f'{index} Python host controls',before==after);check(f'{index} Python status',status==direct['status']);record['python']=dict(status=status,error=error.value.decode(),controls_before=before,controls_after=after)
            if status==0:
                for name,data in (('values',bytes(v)),('indicators',bytes(e))):check(f'{index} Python {name}',[data[j*24:(j+1)*24].hex() for j in range(len(z))]==[parse_hex(x).hex() for x in direct[name]])
                if job['operation']:
                    check(f'{index} Python probability bits',format(struct.unpack('<Q',struct.pack('<d',result.probability))[0],'x')==direct['probability_bits']);check(f'{index} Python diagnostics',bytes(result.scalars).hex()==''.join(parse_hex(x).hex() for x in direct['scalars']))
        records.append(record);save(out/f'case_{index:02d}.json',record);print(index,job['name'],direct['status'],flush=True)
    check('unchanged sources',hashes=={n:digest(ROOT/n) for n in names});save(out/'validation.json',dict(count=len(checks),passed=sum(c['passed'] for c in checks),jobs=len(jobs),successful_jobs=sum(r['direct']['status']==0 for r in records),expected_rejections=sum(r['direct']['status']!=0 for r in records),checks=checks));print(len(checks))
if __name__=='__main__':main()
