"""Build official FrugalGBS for native/portable, non-exponential-output probes.

This uses a separate binary from the original archived integration. The only
upstream sampling API remains generate_samples; instrumentation is restricted
to small modes and excluded from timed sampling.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time

from build_frugal import DEFAULT_EIGEN, UPSTREAM_COMMIT, make_trace_header, tree_fingerprint

ROOT=Path(__file__).resolve().parents[1]


def build(build_dir,eigen,compiler,modes,native=False,force=False):
    upstream=ROOT/"third_party/FrugalGBS"
    xsimd=ROOT/"third_party/xsimd/include"
    compiler_path=shutil.which(compiler)
    if compiler_path is None:raise FileNotFoundError(compiler)
    if not (eigen/"Eigen/Core").exists():raise FileNotFoundError(eigen)
    if not modes or any(m<2 or m>128 for m in modes):raise ValueError("supported modes are 2..128")
    build_dir=build_dir.resolve();generated=build_dir/"generated"
    (generated/"fgbs").mkdir(parents=True,exist_ok=True)
    global_header=(upstream/"include/fgbs/global.hpp.in").read_text().replace("@FGBS_REAL_PRECISION@","2").replace("@FGBS_MAX_CUTOFF_ORDER@","5")
    (generated/"fgbs/global.hpp").write_text(global_header)
    trace=make_trace_header(upstream)
    (generated/"frugal_trace.hpp").write_text(trace)
    modes_header="#define GBSLAB_FRUGAL_MODES(M) "+" ".join(f"M({n})" for n in modes)+"\n"
    (generated/"frugal_modes.hpp").write_text(modes_header)
    sources=[ROOT/"experiments/frugal_sampling_adapter.cpp",*sorted((upstream/"src/fgbs/core").glob("*.cpp")),upstream/"src/fgbs/layout.cpp",upstream/"src/fgbs/cpu/cpu_sampler.cpp"]
    executable=build_dir/"frugal_sampling_adapter";staged=build_dir/"frugal_sampling_adapter.building"
    flags=["-O3","-std=c++20","-ffp-contract=off","-fopenmp","-DFGBS_GPU_ENABLED=0"]
    if native:flags.append("-march=native")
    command=[compiler_path,*flags,"-I",str(generated),"-I",str(upstream/"include"),"-I",str(eigen),"-I",str(xsimd),*map(str,sources),"-o",str(staged)]
    version=subprocess.check_output([compiler_path,"--version"],text=True)
    compiler_sha256=hashlib.sha256(Path(compiler_path).read_bytes()).hexdigest()
    dependencies={"eigen":tree_fingerprint(eigen),"xsimd":tree_fingerprint(xsimd)}
    h=hashlib.sha256()
    h.update(json.dumps({"command":command,"compiler":version,"compiler_sha256":compiler_sha256,"dependencies":dependencies},sort_keys=True).encode())
    for content in [trace,global_header,modes_header]:h.update(content.encode())
    for path in [Path(__file__),*sources,*sorted((upstream/"include").rglob("*.hpp"))]:h.update(path.read_bytes())
    cache_key=h.hexdigest();metadata_path=build_dir/"build.json"
    if not force and executable.exists() and metadata_path.exists():
        old=json.loads(metadata_path.read_text())
        if old.get("cache_key")==cache_key and old.get("executable_sha256")==hashlib.sha256(executable.read_bytes()).hexdigest():return executable
    start=time.perf_counter()
    with (build_dir/"compile.log").open("w") as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f"Compile failed; see {build_dir/'compile.log'}")
    staged.replace(executable)
    metadata={"upstream_commit":UPSTREAM_COMMIT,"native":native,"modes":modes,"flags":flags,"command":command,"compiler":version.splitlines()[0],"compiler_sha256":compiler_sha256,"dependencies":dependencies,"cache_key":cache_key,"executable_sha256":hashlib.sha256(executable.read_bytes()).hexdigest(),"build_seconds":time.perf_counter()-start,"precision":"float64","full_law_max_modes":14,"sampling_api":"unmodified official generate_samples","fast_math":False}
    metadata_path.write_text(json.dumps(metadata,indent=2)+"\n")
    return executable


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir",type=Path,default=ROOT/".cache/frugal-native")
    parser.add_argument("--eigen",type=Path,default=Path(os.environ.get("EIGEN3_INCLUDE_DIR",DEFAULT_EIGEN)))
    parser.add_argument("--compiler",default=os.environ.get("CXX","c++"))
    parser.add_argument("--modes",default="8,10,12,16,32,64")
    parser.add_argument("--native",action="store_true")
    parser.add_argument("--force",action="store_true")
    args=parser.parse_args()
    print(build(args.build_dir,args.eigen,args.compiler,tuple(sorted(set(map(int,args.modes.split(','))))),args.native,args.force))
