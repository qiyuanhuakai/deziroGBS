"""Enumerate both complete causal 16-bit prefix laws with the actual 1,152-mode Frugal target."""
import argparse
import ctypes as ct
import json
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_baselines_v1 import independent_or
from iteration100_projection_v1 import physical
from iteration100_spatial_v1 import ROOT, ORDER, H, D, I, B, Native, bind, save, manifest
from iteration103_sample_v1 import BASE, DATA


def compile_audit(binary):
    old=DATA/'iteration75_frugal_cpu_build_v1'
    build=json.loads((DATA/'iteration103_frugal_full_v1/build.json').read_text())
    upstream=ROOT/'third_party/FrugalGBS/src/fgbs'
    with tempfile.TemporaryDirectory() as directory:
        tmp=Path(directory)
        header=tmp/'fgbs/core/consts.hpp'
        header.parent.mkdir(parents=True)
        original=(old/'include/fgbs/core/consts.hpp').read_text()
        start=original.index('inline constexpr std::array')
        loops=original[original.index('    for (usize i'):original.index('    return arr;')]
        runtime=('inline auto const& gbs_index_arrays = *[]{\n'
                 '    using Table = std::array<std::array<usize,N+1>,N+1>;\n'
                 '    auto* ptr=new Table{}; auto& arr=*ptr;\n'+loops+'    return ptr;\n}();')
        header.write_text(original[:start]+runtime+original[original.index(';',start)+1:])
        assert H(header)==build['effective_index']
        cmd=[shutil.which('g++'),*build['flags']]
        for inc in (tmp,old/'generated',old/'include',ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0',ROOT/'third_party/xsimd/include',DATA/'iteration81_prefix_build_v1'):
            cmd+=['-I',str(inc)]
        cmd+=[str(ROOT/'native/frugal_prefix105_v1.cpp'),str(upstream/'layout.cpp')]
        cmd+=[str(upstream/f'core/{part}.cpp') for part in ('correlator_utils','cumulant_utils','matrix_utils','result','string_utils','thread_utils','time_utils')]
        subprocess.run([*cmd,'-o',str(binary)],check=True)


def run(out):
    out.mkdir(parents=True,exist_ok=False)
    binary=out/'prefix.dll'
    compile_audit(binary)
    native=Native(BASE/'spatial.dll')
    trace=bind(ct.CDLL(str(binary)),'frugal105_tree',[D,D,ct.c_char_p,I])
    reduced=bind(ct.CDLL(str(DATA/'iteration96_frugal_build_v1/frugal.dll')),'fc96_sample16',[I,I,ct.c_uint,D,B,D,ct.c_char_p,I])
    cov=2*physical(dict(np.load(ORDER)),np.arange(1152))-np.eye(2304)
    msg=ct.create_string_buffer(2048)
    ids=np.r_[np.arange(16),1152+np.arange(16)]
    laws=[]
    replay=[]
    threading.stack_size(64*1024**2)
    with ThreadPoolExecutor(max_workers=1) as pool:
        for s,q in enumerate(cov):
            th=np.empty(65535)
            assert pool.submit(trace,q,th,msg,len(msg)).result()==0,msg.value
            p=np.ones(1,np.longdouble)
            for j in range(16):
                v=th[len(p)-1:2*len(p)-1]
                assert np.isfinite(v[p>0]).all()
                v=np.nan_to_num(v,nan=1).clip(0,1).astype(np.longdouble)
                p=np.r_[p*v,p*(1-v)]
            laws.append(p)
            x=np.empty((4,16),np.uint8)
            assert reduced(2,4,103081000+s,np.ascontiguousarray(q[np.ix_(ids,ids)]),x,np.zeros(7),msg,len(msg))==0,msg.value
            replay.append(x)
    both=independent_or(laws)
    truth=np.load(DATA/'iteration105_jiuzhang_v1/optical_panel_1.npz')['truth']
    tv=float(abs(both-truth).sum()/2)
    prev=json.loads((DATA/'iteration103_baselines_v1/summary.json').read_text())['click'][-1]['frugal'][0]['complete_tv']
    assert abs(tv-prev)<1e-9
    np.savez_compressed(out/'law.npz',frugal=both,optical=truth,seed_replay=replay)
    save(out,'summary.json',dict(modes=1152,depth=16,K=2,spectra=2,conditional_nodes=2*65535,full_target_prefix_tv=tv,projected_prefix_tv=prev))
    manifest(out)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('output',type=Path)
    with threadpool_limits(limits=1,user_api='blas'):run(parser.parse_args().output.resolve())
