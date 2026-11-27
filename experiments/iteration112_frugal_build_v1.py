"""Instantiate the archived Frugal GPU adapter at exactly 13 output bits, in FP32 and CPU FP64."""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from iteration100_spatial_v1 import ROOT,H,save,manifest

def build(out):
    if out.exists() and any(out.iterdir()):raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    previous=ROOT/'results/iteration72_frugal_gpu_build_v5'
    upstream=ROOT/'third_party/FrugalGBS'
    original=json.loads((previous/'plan.json').read_text())
    for name,sha in original['upstream_sha256'].items():
        if name.startswith(('include/','src/')):assert H(upstream/name)==sha
    host=(ROOT/'native/frugal_gpu72_host_v4.cpp').read_text()
    line=next(l for l in host.splitlines() if 'switch(modes)' in l)
    host=host.replace(line,'    if(modes!=13)throw std::runtime_error("Expected 13 modes"); run72<13>(cutoff,count,seed,covariance,output,timings);return 0;')
    kernels=(ROOT/'native/frugal_gpu72_kernels_v4.cu').read_text().splitlines()
    kernels='\n'.join([line.replace('<4>','<13>') for line in kernels if not line.startswith('template ') or '<4>' in line])+'\n'
    eigen=ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'
    vc=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    with tempfile.TemporaryDirectory(prefix='gbs112-frugal-') as tmp:
        tmp=Path(tmp);shutil.copytree(upstream/'include',tmp/'include');shutil.copytree(upstream/'src',tmp/'src');shutil.copytree(eigen,tmp/'eigen')
        shutil.copyfile(previous/'layout_compatibility.hpp',tmp/'include/fgbs/layout.hpp')
        (tmp/'include/fgbs/global.hpp').write_text('#pragma once\ninline constexpr auto FGBS_REAL_PRECISION = 1;\ninline constexpr auto FGBS_MAX_CUTOFF_ORDER = 5;\n')
        (tmp/'adapter.cpp').write_text(host);(tmp/'kernels.cu').write_text(kernels)
        (tmp/'env.bat').write_text(f'@call "{vc}" >nul\n@set\n',encoding='ascii')
        env=os.environ.copy()
        for line in subprocess.check_output(['cmd','/d','/c','env.bat'],cwd=tmp,text=True,errors='replace').splitlines():
            if '=' in line:
                k,v=line.split('=',1)
                if k:env[k]=v
        cmd=json.loads((previous/'build_1.json').read_text())['command']
        cmd[0]=shutil.which('nvcc');cmd[cmd.index('adapter.cpp')-1]='eigen'
        pr=subprocess.run(cmd,cwd=tmp,env=env,capture_output=True,text=True,errors='replace')
        save(out,'compiler.json',dict(command=cmd,returncode=pr.returncode,stdout=pr.stdout,stderr=pr.stderr,commit=original['upstream_commit']))
        pr.check_returncode()
        for name in ('frugal_fp32.dll','kernels.cu','adapter.cpp'):shutil.copyfile(tmp/name,out/name)
    cpu_host=(ROOT/'native/frugal_tree96_v1.cpp').read_text().replace('fc96_full16','fc112_full13').replace('fc96_sample16','fc112_sample13').replace('prefix81<16>(cutoff,16','prefix81<13>(cutoff,13').replace('run72<16>','run72<13>')
    cpp=out/'cpu.cpp';cpp.write_text(cpu_host)
    cmd=json.loads((ROOT/'results/iteration96_frugal_build_v1/build.json').read_text())['command']
    cmd[0]=shutil.which('g++')
    for k,arg in enumerate(cmd):
        if Path(arg).name=='frugal_tree96_v1.cpp':cmd[k]=str(cpp)
    cmd[-1]=str(out/'cpu.dll')
    cmd[1:1]=['-I',str(ROOT/'native')]
    pr=subprocess.run(cmd,capture_output=True,text=True,errors='replace')
    save(out,'cpu_build.json',dict(command=cmd,returncode=pr.returncode,stderr=pr.stderr[-3000:]))
    pr.check_returncode()
    manifest(out)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);build(ap.parse_args().output.resolve())
