"""AVX2 Frugal CPU builds, default FP32 and FP64, preserving all failures."""
from pathlib import Path
import os,json,hashlib,shutil,subprocess,time,argparse
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
upstream=ROOT/'third_party/FrugalGBS';command=json.loads((ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/plan.json').read_text())['frugal_command'];generated=OUT/'generated';(generated/'fgbs').mkdir(parents=True);(generated/'frugal_extended_modes.hpp').write_text('#define GBSLAB_FRUGAL_EXTENDED_MODES(M) M(4) M(8) M(144)\n',encoding='ascii')
hooks=ROOT/'results/iteration39_build/generated/frugal_extended_hooks.hpp';shutil.copy2(hooks,generated/hooks.name)
old_include=str(ROOT/'results/iteration39_build/generated');command=[str(generated) if v==old_include else v for v in command];command[1:1]=['-march=x86-64-v3','-mstackrealign'];sources=[Path(__file__),ROOT/'experiments/frugal_extended.cpp',hooks]+[p for p in upstream.rglob('*') if p.is_file()]+[p for p in (ROOT/'third_party/xsimd').rglob('*') if p.is_file()]
save('plan.json',dict(source_hashes={p.relative_to(ROOT).as_posix():H(p) for p in sources},cpu_flags=['-march=x86-64-v3','-mstackrealign'],precision_settings=[1,2],scope=__doc__,sampler_core='Unmodified official CPU sampling entry with existing optional audit instrumentation disabled. Generated config only; no host layout patch.'))
data=np.load(ROOT/'results/iteration71_branch_pilot_v1/input_3.npz');v=data['v'];o=np.block([[v.real,-v.imag],[v.imag,v.real]]);cov=np.eye(8)+2*o@data['x']@o.T;np.savetxt(OUT/'small_covariance.csv',cov,delimiter=',',fmt='%.17g');rows=[]
for precision in (1,2):
    bits=32 if precision==1 else 64;(generated/'fgbs/global.hpp').write_text('#ifndef FGBS_GLOBAL_HPP\n#define FGBS_GLOBAL_HPP\ninline constexpr auto FGBS_REAL_PRECISION = '+str(precision)+';\ninline constexpr auto FGBS_MAX_CUTOFF_ORDER = 5;\n#endif\n',encoding='ascii');shutil.copy2(generated/'fgbs/global.hpp',OUT/f'global_fp{bits}.hpp');exe=OUT/f'frugal_avx2_fp{bits}.exe';command[-1]=str(exe);start=time.perf_counter();r=subprocess.run(command,capture_output=True,text=True,errors='replace');save(f'build_{bits}.json',dict(command=command.copy(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start));print('AVX2 build',bits,r.returncode,flush=True)
    row=dict(precision=bits,build_returncode=r.returncode,smoke=[])
    if r.returncode==0:
        for modes,cutoff,count,path in ((4,4,100000,OUT/'small_covariance.csv'),(144,2,16,ROOT/'results/iteration70_archive/iteration70_frugal_large_v1_20261110/covariance.csv')):
            args2=[str(exe),'--input',str(path),'--modes',str(modes),'--cutoff',str(cutoff),'--samples',str(count),'--seed','735001','--threads','16'];start=time.perf_counter();q=subprocess.run(args2,capture_output=True,text=True,errors='replace');save(f'process_{bits}_{modes}.json',dict(command=args2,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,seconds=time.perf_counter()-start));entry=dict(modes=modes,returncode=q.returncode)
            if q.returncode==0:
                value=json.loads(q.stdout);save(f'result_{bits}_{modes}.json',value);entry['simd_lanes']=value['simd_lanes']
                if modes==4:
                    clicks=np.array([[int(c) for c in word] for word in value['outputs']],np.uint8);hist=np.bincount(clicks@np.array([1,2,4,8]),minlength=16)/len(clicks);entry['tv_to_exact']=float(abs(hist-data['target']).sum()/2);assert entry['tv_to_exact']<.015
            row['smoke'].append(entry);print(entry,flush=True)
            if q.returncode:break
        row['binary_sha256']=H(exe)
    rows.append(row)
save('summary.json',dict(passed=all(r['build_returncode']==0 and len(r['smoke'])==2 and all(t['returncode']==0 for t in r['smoke']) for r in rows),rows=rows));shutil.copy2(__file__,OUT/Path(__file__).name);save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():H(p) for p in OUT.rglob('*') if p.is_file()}))
