"""Measure official CPU Frugal K2 on both 1152-port spectral covariances, with a runtime index table."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import ROOT,ORDER,H,bind,I,D,B,manifest,save
from iteration100_projection_v1 import physical


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'results/iteration75_frugal_cpu_build_v1'
    upstream=ROOT/'third_party/FrugalGBS'
    binary=out/'frugal.dll'
    with tempfile.TemporaryDirectory() as temp:
        tmp=Path(temp)
        hdr=tmp/'fgbs/core/consts.hpp'
        hdr.parent.mkdir(parents=True)
        orig=(old/'include/fgbs/core/consts.hpp').read_text()
        a=orig.index('    for (usize i')
        b=orig.index('    return arr;')
        begin=orig.index('inline constexpr std::array')
        end=orig.index(';',begin)+1
        runtime=('inline auto const& gbs_index_arrays = *[]{\n'
                 '    using Table = std::array<std::array<usize,N+1>,N+1>;\n'
                 '    auto* ptr=new Table{}; auto& arr=*ptr;\n'+orig[a:b]+'    return ptr;\n}();')
        hdr.write_text(orig[:begin]+runtime+orig[end:])
        src=tmp/'bench.cpp'
        adapter=(ROOT/'native/frugal_cpu75_adapter_v1.cpp').read_text()
        start=adapter.index('case 4:')
        end=adapter.index('default:',start)
        cases=''.join(f'case {n}:run72<{n}>(cutoff,count,seed,covariance,output,timings);break;' for n in (8,1152))
        src.write_text(adapter[:start]+cases+adapter[end:])
        eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
        core=upstream/'src/fgbs/core'
        command=[shutil.which('g++'),'-O3','-std=c++20','-shared','-fopenmp','-march=x86-64-v3','-mstackrealign','-ffp-contract=off','-fno-fast-math','-DFGBS_GPU_ENABLED=0']
        for inc in (tmp,old/'generated',old/'include',eigen,ROOT/'third_party/xsimd/include'):
            command+=['-I',str(inc)]
        command += [str(src),str(upstream/'src/fgbs/layout.cpp')]+[str(core/f'{x}.cpp') for x in ('correlator_utils','cumulant_utils','matrix_utils','result','string_utils','thread_utils','time_utils')]+['-o',str(binary)]
        subprocess.run(command,check=True)
        save(out,'build.json',dict(flags=command[1:10],input_adapter=H(ROOT/'native/frugal_cpu75_adapter_v1.cpp'),input_index=H(old/'include/fgbs/core/consts.hpp'),effective_index=H(hdr),effective_adapter=H(src)))
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    def sampler(binary):
        return bind(ct.CDLL(str(binary)),'fc73_sample',[I,I,I,ct.c_uint,I,D,B,D,ct.c_char_p,I])
    def draw(f,cov,n,seed):
        modes=len(cov)//2
        x=np.empty((n,modes),np.uint8)
        st=np.zeros(7)
        msg=ct.create_string_buffer(4096)
        tic=perf_counter()
        rc=f(modes,2,n,seed,1,cov,x,st,msg,len(msg))
        if rc:raise RuntimeError(msg.value)
        return x,st,perf_counter()-tic
    archived=sampler(old/'frugal_cpu_fp64.dll')
    full=sampler(binary)
    z=dict(np.load(ORDER))
    test=2*physical(z,np.arange(8))[0]-np.eye(16)
    assert np.array_equal(draw(archived,test,256,103080000)[0],draw(full,test,256,103080000)[0])
    cov=2*physical(z,np.arange(1152))-np.eye(2304)
    timings=[]
    batches=[]
    threading.stack_size(64*1024**2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for j,q in enumerate(cov):
            x,st,elapsed=pool.submit(draw,full,q,4,103081000+j).result()
            batches.append(x)
            timings.append(dict(total_s=elapsed,layout_s=float(st[0]),cumulant_layout_s=float(st[1]),target_s=float(st[2]),draw_s=float(st[3]),copy_s=float(st[4]),simd=int(st[6])))
            print(j,timings[-1],flush=True)
    np.savez_compressed(out/'samples.npz',spectra=np.array(batches),clicks=batches[0]|batches[1])
    save(out,'summary.json',dict(modes=1152,K=2,records=4,threads=1,spectra=timings,draw_us=sum(r['draw_s'] for r in timings)*1e6/4,total_us=sum(r['total_s'] for r in timings)*1e6/4,small_mode_seed_check=True))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):main(p.parse_args().output.resolve())
