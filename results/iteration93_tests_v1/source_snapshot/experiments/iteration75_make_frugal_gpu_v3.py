"""Use byte-verified literal index tables to bypass NVCC constexpr complexity limits."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
insertion=r'''
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
'''
s=(ROOT/'experiments/iteration75_frugal_gpu_build_v2.py').read_text(encoding='utf-8');s=s.replace('Build expanded-dimension FP64 v2, higher constexpr evaluation budget:','Build expanded-dimension FP64 v3, byte-verified literal index initialization:');s=s.replace("save('plan.json',",insertion+"\nsave('plan.json',",1);s=s.replace('thin C ABI, default single and optional double precision.','thin C ABI, FP64, byte-verified literal index initialization.');compile(s,'gpu75v3','exec')
with (ROOT/'experiments/iteration75_frugal_gpu_build_v3.py').open('xb') as f:f.write(s.encode())
s=(ROOT/'experiments/iteration75_scale_click_v2.py').read_text(encoding='utf-8').replace('iteration75_frugal_gpu_build_v2/frugal_fp64.dll','iteration75_frugal_gpu_build_v3/frugal_fp64.dll');(ROOT/'experiments/iteration75_scale_click_v2.py').write_bytes(s.encode())
