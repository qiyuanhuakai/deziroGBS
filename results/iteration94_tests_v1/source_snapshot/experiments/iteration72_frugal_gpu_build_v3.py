"""Build v3: host-only Eigen and an MSVC OpenMP task capture workaround.

CUDA kernels unchanged. Host layout recursion takes its bitstring by value
instead of a forwarding reference, enabling OpenMP firstprivate task copies.
Preserves failed builds v1/v2 and saves the exact host compatibility diff.
https://eigen.tuxfamily.org/dox/TopicCUDA.html
"""
from pathlib import Path
import os,json,hashlib,shutil,subprocess,time,uuid,argparse
ROOT=Path(__file__).resolve().parents[1];parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
upstream=ROOT/'third_party/FrugalGBS';nvcc=Path(shutil.which('nvcc'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';vcvars=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat';build=Path(os.environ['LOCALAPPDATA'])/'Temp'/('deziro_frugal_gpu72_'+uuid.uuid4().hex);build.mkdir(exist_ok=False)
shutil.copytree(upstream/'include',build/'include');shutil.copytree(upstream/'src',build/'src');shutil.copy2(ROOT/'native/frugal_gpu72_adapter.cu',build/'adapter.cu');(build/'environment.bat').write_text('@call "'+str(vcvars)+'" >nul\n@set\n',encoding='ascii');values=subprocess.check_output(['cmd','/d','/c','environment.bat'],cwd=build,text=True,errors='replace');env=os.environ.copy()
for line in values.splitlines():
    if '=' in line:key,value=line.split('=',1);env[key]=value
header=build/'include/fgbs/layout.hpp';original=header.read_text();patched=original.replace('BITSTRING&& bitstring','BITSTRING bitstring');assert patched!=original
header.write_text(patched,encoding='utf-8')
import difflib
(OUT/'host_layout_compatibility.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='official/layout.hpp',tofile='build/layout.hpp')),encoding='utf-8')
shutil.copy2(header,OUT/'layout_compatibility.hpp')
save('plan.json',dict(upstream_commit='fa8b796f2afebd4845a2e335e457735952617ef8',upstream_sha256={p.relative_to(upstream).as_posix():H(p) for p in upstream.rglob('*') if p.is_file()},adapter_sha256=H(ROOT/'native/frugal_gpu72_adapter.cu'),driver_sha256=H(Path(__file__)),build_directory=str(build),precision_settings=[1,2],gpu=0,scope='Official unmodified CUDA sampler and cumulant implementation, host layout bitstring-by-value compatibility patch, thin C ABI, default single and optional double precision. No CPU SIMD sampler included. OpenMP host layout/correlators plus CUDA target/sampling, sm_120.'))
shutil.copy2(__file__,OUT/Path(__file__).name);shutil.copy2(ROOT/'native/frugal_gpu72_adapter.cu',OUT/'frugal_gpu72_adapter.cu');rows=[]
core=['correlator_utils.cpp','cumulant_utils.cpp','matrix_utils.cpp','result.cpp','string_utils.cpp','thread_utils.cpp','time_utils.cpp']
for precision in (1,2):
    (build/'include/fgbs/global.hpp').write_text('#ifndef FGBS_GLOBAL_HPP\n#define FGBS_GLOBAL_HPP\ninline constexpr auto FGBS_REAL_PRECISION = '+str(precision)+';\ninline constexpr auto FGBS_MAX_CUTOFF_ORDER = 5;\n#endif\n',encoding='ascii')
    command=[str(nvcc),'-O3','-arch=sm_120','--fmad=false','-std=c++20','-shared','-DFGBS_GPU_ENABLED=1','-DEIGEN_NO_CUDA','-DEIGEN_DONT_VECTORIZE','-Xcompiler','/MD,/utf-8,/openmp:llvm','-I','include','-I',str(eigen),'adapter.cu','src/fgbs/gpu/gpu_utils.cu','src/fgbs/layout.cpp']+['src/fgbs/core/'+name for name in core]+['-o',f'frugal_fp{32 if precision==1 else 64}.dll']
    start=time.perf_counter();r=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True,errors='replace');row=dict(precision=precision,command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start);save(f'build_{precision}.json',row);rows.append(row);print('Frugal GPU build',precision,r.returncode,flush=True)
    if r.returncode==0:shutil.copy2(build/f'frugal_fp{32 if precision==1 else 64}.dll',OUT/f'frugal_fp{32 if precision==1 else 64}.dll')
save('summary.json',dict(passed=all(r['returncode']==0 for r in rows),rows=[dict(precision=r['precision'],returncode=r['returncode']) for r in rows],binaries={p.name:H(p) for p in OUT.glob('*.dll')}))
