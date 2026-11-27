"""Restore two untouched high-power captures from the official per-member RAR index."""
import argparse
import hashlib
import json
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]


def header(kind):
    data=bytes([3,kind,0,0])
    return struct.pack('<I',zlib.crc32(data))+data


def run(out,numbers=(10,11)):
    out.mkdir(parents=True,exist_ok=True)
    path='results/iteration81_jiuzhang_inventory_v1/volume01.json'
    raw_index=subprocess.check_output(['git','cat-file','blob','HEAD:'+path],cwd=ROOT)
    if hashlib.sha256(raw_index).hexdigest()!='808e16750dfaf5f176cb9966fab2e38ced7507718f99aab8a1b4e6733dda70d0':
        raise ValueError('Archived volume index digest differs')
    index=json.loads(raw_index)
    rows=[]
    for number in numbers:
        name=f'jiuzhang3/data/high power/high power ({number}).bin'
        entry=next(a for a in index['files'] if a['filename']==name)
        target=out/name
        if not target.exists():
            start=entry['header_offset'];end=entry['data_offset']+entry['compress_size']-1
            with tempfile.TemporaryDirectory(dir=out) as tmp:
                rar=Path(tmp)/'capture.rar'
                with rar.open('wb') as f:
                    f.write(b'Rar!\x1a\x07\x01\x00'+header(1))
                    for offset in range(start,end+1,8*1024**2):
                        last=min(end,offset+8*1024**2-1)
                        for attempt in range(4):
                            try:
                                response=requests.get(index['url'],headers={'Range':f'bytes={offset}-{last}'},timeout=(20,65))
                                if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {offset}-{last}/') or len(response.content)!=last-offset+1:
                                    raise OSError(f'Unexpected HTTP range {response.status_code} {response.headers.get("Content-Range")}')
                                break
                            except (requests.RequestException,OSError):
                                if attempt==3:raise
                        f.write(response.content)
                    f.write(header(5))
                subprocess.run(['tar','-xf',str(rar),'-C',tmp],check=True)
                raw=Path(tmp)/name
                b=raw.read_bytes()
                assert len(b)==entry['file_size'] and zlib.crc32(b)==entry['CRC']
                target.parent.mkdir(parents=True,exist_ok=True)
                raw.replace(target)
        b=target.read_bytes()
        assert len(b)==entry['file_size'] and zlib.crc32(b)==entry['CRC']
        rows.append(dict(file=number,source=index['url'],member=name,rows=len(b)//144,bytes=len(b),
                         crc32=entry['CRC'],sha256=hashlib.sha256(b).hexdigest()))
        (out/'files.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf8')
        print(rows[-1],flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('output',type=Path)
    run(ap.parse_args().output.resolve())
