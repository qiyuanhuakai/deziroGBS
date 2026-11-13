"""Explicit Windows build and safe ABI probe; never invoke mismatched scalars."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from time import perf_counter
import traceback
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from gbslab.batched_forest import _Formats,BatchedForestKernel
from iteration23_support import digest,save


def main():
    p=argparse.ArgumentParser(); p.add_argument('--compiler',type=Path,required=True); p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False); compiler=args.compiler.resolve()
    sources=sorted({'experiments/iteration26_native.py','experiments/build_batched_forest.py','experiments/batched_forest_kernel.cpp','experiments/batched_forest_kernel.h','experiments/iteration23_support.py','experiments/iteration25_worker.py','experiments/iteration24_reference.py',*(f.relative_to(ROOT).as_posix() for f in (ROOT/'src').rglob('*.py'))}); hashes={f:digest(ROOT/f) for f in sources}
    for f in sources:
        dst=out/'source_snapshot'/f; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/f,dst)
    def run(label,command,timeout=180):
        start=perf_counter()
        try:
            r=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
            (out/(label+'.stdout.log')).write_text(r.stdout,encoding='utf-8'); (out/(label+'.stderr.log')).write_text(r.stderr,encoding='utf-8')
            return dict(command=command,returncode=r.returncode,seconds=perf_counter()-start)
        except Exception as e: return dict(command=command,error=repr(e),seconds=perf_counter()-start)
    dependencies={}
    for name in ('libquadmath.a','libquadmath.dll.a','libquadmath-0.dll','include'):
        value=subprocess.check_output([str(compiler),'-print-file-name='+name],text=True).strip(); path=Path(value).resolve()
        if name=='include': path=path/'quadmath.h'
        dependencies[name]=dict(path=str(path),exists=path.is_file(),sha256=digest(path) if path.is_file() else None)
    save(out/'plan.json',dict(source_hashes=hashes,compiler=str(compiler),compiler_sha256=digest(compiler),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True),dependencies=dependencies,platform=os.name,interface_policy='Discover integer-only formats first; incompatible long-double ABI must be rejected before kernel creation'))
    shutil.copy2(ROOT/'research/iteration26_panel_plan.md',out/'frozen_plan.md')
    legacy=run('legacy',[sys.executable,'experiments/build_batched_forest.py','--compiler',str(compiler),'--build-dir',str(out/'legacy')]); save(out/'legacy.json',legacy)
    dll=out/'libbatched_forest.dll'; command=[str(compiler),'-O2','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','experiments/batched_forest_kernel.cpp','-o',dll.relative_to(ROOT).as_posix(),'-lquadmath']
    build=run('portable_windows',command); save(out/'compile.json',build); probe=dict(status='build_failed',kernel_created=False)
    if build.get('returncode')==0:
        save(out/'build.json',dict(library_sha256=digest(dll),source_hashes=hashes,command=command,compiler_sha256=digest(compiler)))
        try:
            search_dirs={compiler.parent,Path(dependencies['libquadmath-0.dll']['path']).parent}
            dll_search_handles=[os.add_dll_directory(str(d)) for d in search_dirs] if os.name=='nt' else []
            lib=ct.CDLL(str(dll)); lib.bf_formats.argtypes=[ct.POINTER(_Formats)]; lib.bf_formats.restype=ct.c_int; formats=_Formats(); assert lib.bf_formats(ct.byref(formats))==0
            info={name:int(getattr(formats,name)) for name,_ in formats._fields_}
            compatible=info['long_double_bytes']==ct.sizeof(ct.c_longdouble)==np.dtype(np.longdouble).itemsize and info['long_double_mantissa_bits']==np.finfo(np.longdouble).nmant+1
            probe.update(native_formats=info,ctypes_long_double_bytes=ct.sizeof(ct.c_longdouble),numpy_long_double_bytes=np.dtype(np.longdouble).itemsize,numpy_long_double_mantissa_bits=np.finfo(np.longdouble).nmant+1,scalar_format_compatible=compatible)
            try:
                kernel=BatchedForestKernel(np.eye(4),library=dll); probe['kernel_created']=True; kernel.close(); probe['status']='compatible_constructor_passed'
                assert compatible
            except RuntimeError as error:
                if compatible: raise
                assert 'format mismatch' in str(error); probe.update(status='abi_mismatch_rejected',adapter_error=str(error))
        except Exception: probe.update(status='probe_failed',traceback=traceback.format_exc())
    save(out/'probe.json',probe)
    # Exercise the repaired relative-path worker through its real CLI.
    job=out/'path_job.json'; target=out/'path_result.json'; save(job,dict(panel='quality',modes=2,seed=260901,dimension=4,rank=4,max_seconds=120))
    relative=run('relative_path',[sys.executable,'experiments/iteration25_worker.py',job.relative_to(ROOT).as_posix(),target.relative_to(ROOT).as_posix()])
    if target.exists():
        record=json.loads(target.read_text(encoding='utf-8')); relative.update(status=record['status'],validated=record.get('validation_status')=='passed',state_exists=(ROOT/record.get('state',{}).get('path','missing')).is_file())
    save(out/'relative_path.json',relative)
    assert hashes=={f:digest(ROOT/f) for f in sources}
    summary=dict(legacy_build_returncode=legacy.get('returncode'),windows_compile_returncode=build.get('returncode'),native_status=probe['status'],kernel_created=probe['kernel_created'],relative_path_validated=relative.get('validated',False),unchanged_sources=True)
    save(out/'summary.json',summary); print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
