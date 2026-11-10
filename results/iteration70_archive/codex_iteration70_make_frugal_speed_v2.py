from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'.git/codex_iteration70_frugal_speed_v1.py'
s=p.read_text(encoding='utf-8').replace('frugal_speed_v1_20261110','frugal_speed_v2_20261110')
s=s.replace("['-march=native','-ffp-contract=off','-fno-fast-math']", "['-ffp-contract=off','-fno-fast-math']")
s=s.replace("'-O3','-march=native','-std=c++20'", "'-O3','-std=c++20'")
anchor="gcc=Path(shutil.which('g++'));runtime=os.add_dll_directory(str(gcc.parent));dll=ROOT/'.git/iteration70_flow_pilot_v1_20261110/native.dll';lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)"
replacement="""gcc=Path(shutil.which('g++'));runtime=os.add_dll_directory(str(gcc.parent));dll=OUT/'native.dll'
native_source=ROOT/'.git/iteration70_flow_v1.cpp';shutil.copy2(native_source,OUT/native_source.name)
native_command=[str(gcc)]+flags+['-shared','-DEIGEN_DONT_PARALLELIZE','-I',str(ROOT.parent/'external_sources/eigen_3_4_0_iteration38/eigen-3.4.0'),str(native_source),'-o',str(dll)]
start=time.perf_counter();nb=subprocess.run(native_command,capture_output=True,text=True,errors='replace');save('native_build.json',dict(command=native_command,returncode=nb.returncode,stdout=nb.stdout,stderr=nb.stderr,seconds=time.perf_counter()-start,source_sha256=H(native_source),included_source_sha256=H(ROOT/'native/gaussian_reset_cpu.cpp')));print('native build',nb.returncode,flush=True);assert nb.returncode==0
lib=C.CDLL(str(dll));DP=C.POINTER(C.c_double);UP=C.POINTER(C.c_uint32)"""
assert anchor in s;s=s.replace(anchor,replacement)
new=ROOT/'.git/codex_iteration70_frugal_speed_v2.py';assert not new.exists();compile(s,str(new),'exec');new.write_bytes(s.encode())
