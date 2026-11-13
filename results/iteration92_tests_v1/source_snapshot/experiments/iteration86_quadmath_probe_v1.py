"""Isolate SSE rounding changes in quadmath operations, no sampling."""
from pathlib import Path
import os,ctypes as C,json,argparse,subprocess,shutil,hashlib
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();ROOT=Path(__file__).resolve().parents[1];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
def save(n,x):(out/n).write_text(json.dumps(x,indent=2)+'\n',encoding='utf-8')
source=ROOT/'native/quadmath_environment86_v1.cpp';gcc=Path(shutil.which('g++'));command=[str(gcc),'-O3','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math',str(source),'-o',str(out/'probe.dll'),'-lquadmath'];r=subprocess.run(command,capture_output=True,text=True,errors='replace');save('build.json',dict(command=command,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr
runtime=os.add_dll_directory(str(gcc.parent));lib=C.CDLL(str(out/'probe.dll'));fn=lib.quad86_probe;fn.argtypes=[C.c_int,C.c_char_p,C.c_char_p,C.c_char_p,C.POINTER(C.c_uint),C.c_char_p,C.c_int];rows=[]
for op in range(5):
 for aa,bb,cc in [('0.1','0.2','-0.02'),('1.0000000000000000000000000000000001','1.0000000000000000000000000000000001','-1'),('1e-50','1e-50','-1e-100'),('1','2','0'),('0','1','0'),('1e-4930','1e-10','0')]:
  state=(C.c_uint*4)();buf=C.create_string_buffer(512);rc=fn(op,aa.encode(),bb.encode(),cc.encode(),state,buf,len(buf));assert rc==0;rows.append(dict(op=op,a=aa,b=bb,c=cc,state=list(state),sse_rounding_changed=(state[1]&0x6000)!=(state[3]&0x6000),result=buf.value.decode()))
save('summary.json',dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}));print(json.dumps([r for r in rows if r['sse_rounding_changed']],indent=2),flush=True)
