"""Retrieve the two other published Jiuzhang 3 calibrations using the archived RAR headers."""
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import zlib
import requests
import numpy as np
from openpyxl import load_workbook
from iteration115_visible_v1 import ROOT,hardware
from iteration117_quartic import OUT

CACHE=ROOT.parent/"external_sources"/"jiuzhang3_iteration117_power"
INVENTORY="results/iteration81_jiuzhang_inventory_v1/volume10.json"


def header(n):
    data=bytes([3,n,0,0])
    return struct.pack("<I",zlib.crc32(data))+data


def parse(matrix,book):
    data=np.loadtxt(matrix,skiprows=3)
    raw=(data[:,0]+1j*data[:,1]).reshape((1152,50),order="F")
    workbook=load_workbook(book,read_only=True,data_only=True)
    values=np.array(list(workbook.active.values),dtype=float)
    workbook.close()
    assert values.shape==(25,2) and np.all(values>0)
    t=np.empty_like(raw);t[:,::2]=1j*(raw[:,::2]+raw[:,1::2])/np.sqrt(2)
    t[:,1::2]=(raw[:,::2]-raw[:,1::2])/np.sqrt(2)
    return t,values.T.repeat(2,axis=1)


def load_power(label):
    root=CACHE/"jiuzhang3"
    return parse(root/"matrix"/(label+" power.mtx"),root/"sqz parameter"/(label+" power.xlsx"))


def run():
    OUT.mkdir(exist_ok=True,parents=True)
    CACHE.mkdir(exist_ok=True,parents=True)
    inventory=json.loads(subprocess.check_output(["git","show","c679bf9a:"+INVENTORY],cwd=ROOT))
    rows=[]
    for f in inventory["files"]:
        name=f["filename"]
        if not any(name.endswith(power+" power"+ext) for power in ("low","median","high") for ext in (".mtx",".xlsx")):continue
        target=CACHE/name;assert target.resolve().is_relative_to(CACHE.resolve())
        start=f["header_offset"];end=f["data_offset"]+f["compress_size"]-1
        if not target.is_file():
            url=inventory["url"]
            response=requests.get(url,headers={"Range":f"bytes={start}-{end}"},timeout=90)
            assert response.status_code==206 and response.headers["Content-Range"].startswith(f"bytes {start}-{end}/")
            assert len(response.content)==end-start+1
            rar=CACHE/(str(start)+".rar")
            rar.write_bytes(b"Rar!\x1a\x07\x01\x00"+header(1)+response.content+header(5))
            subprocess.run(["tar","-xf",str(rar),"-C",str(CACHE)],check=True)
        data=target.read_bytes()
        assert len(data)==f["file_size"] and zlib.crc32(data)==f["CRC"]
        rows.append(dict(name=name,url=inventory["url"],byte_range=[start,end],bytes=len(data),
                         crc32=f["CRC"],sha256=hashlib.sha256(data).hexdigest()))
    assert len(rows)==6
    # End-to-end convention check against the archived high-power input actually used in the paper.
    high=load_power("high")
    for a,b in zip(high,hardware()):np.testing.assert_allclose(a,b,atol=2e-14,rtol=0)
    (OUT/"power_sources.json").write_text(json.dumps(rows,indent=2)+"\n",encoding="utf8")
    print([r["name"] for r in rows],flush=True)


if __name__=="__main__":run()
