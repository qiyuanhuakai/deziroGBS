"""Compile isolated scalar transport; compare every bit with Python integers."""
import argparse
import ctypes as ct
import json
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from iteration23_support import save,digest
ROOT=Path(__file__).resolve().parents[1]


def canonical(mantissa,exponent,sign=0):
    if mantissa==0: return bytes(24)
    while mantissa%2==0: mantissa//=2; exponent+=1
    return bytes([1,sign,0,0])+int(exponent).to_bytes(4,'little',signed=True)+mantissa.to_bytes(16,'little')


def parse_hex(value):
    match=re.fullmatch(r'(-?)0x([0-9a-f]+)(?:\.([0-9a-f]+))?p([+-]?[0-9]+)',value)
    if not match: raise ValueError(value)
    sign,whole,frac,exponent=match.groups(); frac=frac or ''
    return canonical(int(whole+frac,16),int(exponent)-4*len(frac),int(bool(sign)))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--compiler',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    out=a.output.resolve(); out.mkdir(parents=True,exist_ok=False); compiler=a.compiler.resolve()
    sources=['experiments/iteration28_codec.py','experiments/portable_scalar_codec.cpp','experiments/iteration23_support.py']; hashes={s:digest(ROOT/s) for s in sources}
    for name in sources:
        dest=out/'source_snapshot'/name; dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(ROOT/name,dest)
    quad=Path(subprocess.check_output([str(compiler),'-print-file-name=libquadmath-0.dll'],text=True).strip()).resolve()
    save(out/'plan.json',dict(source_hashes=hashes,compiler_sha256=digest(compiler),compiler=str(compiler),compiler_version=subprocess.check_output([str(compiler),'--version'],text=True),quadmath_dll=dict(path=str(quad),sha256=digest(quad)),random_seed=281401,policy='Isolated codec only. Exact canonical wire comparisons; original forest ABI remains unchanged and rejected.'))
    dll=out/'scalar_codec.dll'; command=[str(compiler),'-O2','-std=c++20','-shared','-ffp-contract=off','-fno-fast-math','experiments/portable_scalar_codec.cpp','-o',str(dll),'-lquadmath']
    build=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace'); (out/'compile.stdout.log').write_text(build.stdout); (out/'compile.stderr.log').write_text(build.stderr); save(out/'build.json',dict(command=command,returncode=build.returncode))
    if build.returncode: raise RuntimeError('Codec build failed; logs preserved')
    handles=[os.add_dll_directory(str(d)) for d in {compiler.parent,quad.parent}] if os.name=='nt' else []
    lib=ct.CDLL(str(dll)); buf_type=ct.c_ubyte*24
    lib.scalar_wire_format.argtypes=[ct.POINTER(ct.c_int),ct.c_size_t]; lib.scalar_wire_format.restype=ct.c_int
    lib.scalar_wire_from_text.argtypes=[ct.c_char_p,ct.c_int,ct.POINTER(ct.c_ubyte),ct.c_size_t,ct.c_char_p,ct.c_size_t]; lib.scalar_wire_from_text.restype=ct.c_int
    lib.scalar_wire_to_hex.argtypes=[ct.POINTER(ct.c_ubyte),ct.c_size_t,ct.c_int,ct.c_char_p,ct.c_size_t,ct.c_char_p,ct.c_size_t]; lib.scalar_wire_to_hex.restype=ct.c_int
    fmt=(ct.c_int*4)(); assert lib.scalar_wire_format(fmt,4)==0; formats=list(fmt); assert formats[:2]==[1,24]
    records=[]; rng=random.Random(281401)
    for precision,bits in enumerate(formats[2:]):
        assert bits in (64,113)
        cases=[(0,0,0),(1,0,1),(1,-16445 if bits==64 else -16494,0),((1<<bits)-1,16384-bits,0),(1,-16382,1),((1<<bits)-1,-bits,1)]
        for _ in range(128):
            length=rng.randint(1,bits); cases.append((rng.getrandbits(length)|1,rng.randint(-16000,16000-bits),rng.randrange(2)))
        for i,(mantissa,exponent,sign) in enumerate(cases):
            text=('-' if sign else '')+f'0x{mantissa:x}p{exponent:+d}'; expected=canonical(mantissa,exponent,sign); buffer=buf_type(); error=ct.create_string_buffer(256); output=ct.create_string_buffer(256)
            code=lib.scalar_wire_from_text(text.encode(),precision,buffer,24,error,256); encoded=bytes(buffer)
            decoded=lib.scalar_wire_to_hex(buffer,24,precision,output,256,error,256) if code==0 else -1
            passed=code==decoded==0 and encoded==expected and parse_hex(output.value.decode())==expected
            records.append(dict(kind='exact_roundtrip',precision=precision,case=i,input=text,wire=encoded.hex(),expected=expected.hex(),decoded=output.value.decode(),passed=passed))
        for text in ('nan','inf','-inf','0x1p-99999','0x1p+99999','not-a-number','1 trailing'):
            buffer=buf_type(); error=ct.create_string_buffer(256); code=lib.scalar_wire_from_text(text.encode(),precision,buffer,24,error,256)
            records.append(dict(kind='invalid_text',precision=precision,input=text,passed=code!=0,error=error.value.decode()))
        malformed=[bytes([2])+bytes(23),bytes([0,1])+bytes(22),bytes([1,2])+bytes(22),bytes([1,0,1])+bytes(21),bytes([1])+bytes(23),bytes([1,0,0,0])+bytes(4)+(2).to_bytes(16,'little'),canonical(1,20001),canonical(1,-20001),canonical((1<<(bits+1))-1,0)]
        for i,data in enumerate(malformed):
            output=ct.create_string_buffer(256); error=ct.create_string_buffer(256); code=lib.scalar_wire_to_hex(buf_type.from_buffer_copy(data),24,precision,output,256,error,256)
            records.append(dict(kind='invalid_wire',precision=precision,case=i,wire=data.hex(),passed=code!=0,error=error.value.decode()))
    save(out/'checks.json',records); unchanged=hashes=={name:digest(ROOT/name) for name in sources}
    save(out/'summary.json',dict(formats=formats,count=len(records),passed=sum(r['passed'] for r in records),unchanged_sources=unchanged,library_sha256=digest(dll),forest_abi_enabled=False))
    print(json.dumps(dict(formats=formats,count=len(records),passed=sum(r['passed'] for r in records),unchanged_sources=unchanged)))
    assert unchanged and all(r['passed'] for r in records)


if __name__=='__main__': main()
