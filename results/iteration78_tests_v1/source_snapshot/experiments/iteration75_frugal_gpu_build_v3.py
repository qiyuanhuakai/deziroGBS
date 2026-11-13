"""Build expanded-dimension FP64 v3, byte-verified literal index initialization: serialize only recursive host layout tasks; split host/CUDA translation units; host-only Eigen and an MSVC OpenMP task capture workaround.

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
upstream=ROOT/'third_party/FrugalGBS';nvcc=Path(shutil.which('nvcc'));eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0';vcvars=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat';build=Path(os.environ['LOCALAPPDATA'])/'Temp'/('deziro_frugal_gpu75_'+uuid.uuid4().hex);build.mkdir(exist_ok=False)
shutil.copytree(eigen,build/'eigen');eigen=build/'eigen'
shutil.copytree(upstream/'include',build/'include');shutil.copytree(upstream/'src',build/'src');shutil.copy2(ROOT/'native/frugal_gpu75_host_v1.cpp',build/'adapter.cpp');(build/'environment.bat').write_text('@call "'+str(vcvars)+'" >nul\n@set\n',encoding='ascii');values=subprocess.check_output(['cmd','/d','/c','environment.bat'],cwd=build,text=True,errors='replace');env=os.environ.copy()
for line in values.splitlines():
    if '=' in line:key,value=line.split('=',1);env[key]=value
header=build/'include/fgbs/layout.hpp';original=header.read_text();patched=original.replace('BITSTRING&& bitstring','BITSTRING bitstring').replace('#pragma omp task if (fork) shared(bitstring_layout)', '// Host compatibility: recurse serially; OpenMP correlator loop is retained.');assert patched!=original
header.write_text(patched,encoding='utf-8')
import difflib
(OUT/'host_layout_compatibility.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),patched.splitlines(True),fromfile='official/layout.hpp',tofile='build/layout.hpp')),encoding='utf-8')
shutil.copy2(header,OUT/'layout_compatibility.hpp')

# NVCC's constant evaluator rejects N=384. Evaluate the identical uint64
# recurrence as literals, and verify every byte against GCC's original header.
from array import array
reference_source=OUT/'index_reference.cpp'
reference_source.write_text('#include "fgbs/core/consts.hpp"\n#include <fstream>\n#include <string>\ntemplate<std::size_t N> void dump(const std::string& p){std::ofstream f(p+"_"+std::to_string(N)+".bin",std::ios::binary);const auto& a=fgbs::gbs_index_arrays<N>;f.write(reinterpret_cast<const char*>(a.data()),sizeof(a));}\nint main(int argc,char** argv){if(argc!=2)return 2;dump<192>(argv[1]);dump<288>(argv[1]);dump<384>(argv[1]);}\n',encoding='utf-8')
reference_binary=OUT/'index_reference.exe';gcc=Path(shutil.which('g++'))
reference_command=[str(gcc),'-O2','-std=c++20','-I',str(ROOT/'results/iteration75_frugal_cpu_build_v1/generated'),'-I',str(upstream/'include'),str(reference_source),'-o',str(reference_binary)]
refbuild=subprocess.run(reference_command,capture_output=True,text=True,errors='replace');save('index_reference_build.json',dict(command=reference_command,returncode=refbuild.returncode,stdout=refbuild.stdout,stderr=refbuild.stderr));assert refbuild.returncode==0,refbuild.stderr
refrun=subprocess.run([str(reference_binary),str(OUT/'original_index')],capture_output=True,text=True,errors='replace');save('index_reference_run.json',dict(returncode=refrun.returncode,stdout=refrun.stdout,stderr=refrun.stderr));assert refrun.returncode==0
mask=(1<<64)-1;literal='';index_rows=[]
for size in (192,288,384):
    table=[[0]*(size+1) for _ in range(size+1)]
    for i in range(size+1):
        table[0][i]=1;binom=1
        for j in range(1,size+1):
            binom=((binom*(i-j+1))&mask)//j if j<=i else 0
            table[j][i]=(table[j-1][i]+binom)&mask
    raw=array('Q',(value for row in table for value in row)).tobytes();native=(OUT/f'original_index_{size}.bin').read_bytes();assert raw==native,(size,len(raw),len(native));index_rows.append(dict(modes=size,entries=(size+1)**2,all_bytes_equal=True,sha256=hashlib.sha256(raw).hexdigest()))
    literal+=f'\ntemplate<> inline constexpr std::array<std::array<usize,{size+1}>,{size+1}> gbs_index_arrays<{size}> = {{{{\n'+',\n'.join('{{'+','.join(str(value)+'ULL' for value in row)+'}}' for row in table)+'\n}};\n'
consts=build/'include/fgbs/core/consts.hpp';original_constants=consts.read_text();patched_constants=original_constants.replace('\n}\n\n#endif',literal+'\n}\n\n#endif');assert patched_constants!=original_constants;consts.write_text(patched_constants,encoding='utf-8');shutil.copy2(consts,OUT/'literal_index_compatibility.hpp');save('index_equivalence.json',dict(passed=True,rows=index_rows,policy='Identical uint64 multiply-wrap/divide/add-wrap recurrence; all entries byte-equal to official GCC compile-time arrays; only constant initialization differs.'))

save('plan.json',dict(upstream_commit='fa8b796f2afebd4845a2e335e457735952617ef8',upstream_sha256={p.relative_to(upstream).as_posix():H(p) for p in upstream.rglob('*') if p.is_file()},adapter_sha256=H(ROOT/'native/frugal_gpu75_host_v1.cpp'),driver_sha256=H(Path(__file__)),build_directory=str(build),precision_settings=[2],gpu=0,scope='Official unmodified CUDA sampler and cumulant implementation, host layout bitstring-by-value and serialized recursive tasks compatibility patch, thin C ABI, FP64, byte-verified literal index initialization. No CPU SIMD sampler included. OpenMP host layout/correlators plus CUDA target/sampling, sm_120.'))
shutil.copy2(__file__,OUT/Path(__file__).name);shutil.copy2(ROOT/'native/frugal_gpu75_host_v1.cpp',OUT/'frugal_gpu75_host_v1.cpp');shutil.copy2(ROOT/'native/frugal_gpu75_kernels_v1.cu',build/'kernels.cu');shutil.copy2(ROOT/'native/frugal_gpu75_kernels_v1.cu',OUT/'frugal_gpu75_kernels_v1.cu');rows=[]
core=['correlator_utils.cpp','cumulant_utils.cpp','matrix_utils.cpp','result.cpp','string_utils.cpp','thread_utils.cpp','time_utils.cpp']
for precision in (2,):
    (build/'include/fgbs/global.hpp').write_text('#ifndef FGBS_GLOBAL_HPP\n#define FGBS_GLOBAL_HPP\ninline constexpr auto FGBS_REAL_PRECISION = '+str(precision)+';\ninline constexpr auto FGBS_MAX_CUTOFF_ORDER = 5;\n#endif\n',encoding='ascii')
    command=[str(nvcc),'-O3','-arch=sm_120','--fmad=false','-std=c++20','-shared','-DFGBS_GPU_ENABLED=1','-DEIGEN_NO_CUDA','-DEIGEN_DONT_VECTORIZE','-Xcompiler','/MD,/utf-8,/openmp:llvm,/constexpr:steps100000000','-I','include','-I',str(eigen),'adapter.cpp','kernels.cu','src/fgbs/gpu/gpu_utils.cu','src/fgbs/layout.cpp']+['src/fgbs/core/'+name for name in core]+['-o',f'frugal_fp{32 if precision==1 else 64}.dll']
    start=time.perf_counter();r=subprocess.run(command,cwd=build,env=env,capture_output=True,text=True,errors='replace');row=dict(precision=precision,command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,seconds=time.perf_counter()-start);save(f'build_{precision}.json',row);rows.append(row);print('Frugal GPU build',precision,r.returncode,flush=True)
    if r.returncode==0:shutil.copy2(build/f'frugal_fp{32 if precision==1 else 64}.dll',OUT/f'frugal_fp{32 if precision==1 else 64}.dll')
save('summary.json',dict(passed=all(r['returncode']==0 for r in rows),rows=[dict(precision=r['precision'],returncode=r['returncode']) for r in rows],binaries={p.name:H(p) for p in OUT.glob('*.dll')}))
