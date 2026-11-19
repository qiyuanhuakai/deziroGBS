"""Extract selected independent official RAR5 entries by verified byte ranges.
Preserve original compressed headers/payloads; wrap only with standalone RAR5
main/end headers. Verify extracted lengths and original per-file CRC32 values.
"""
from pathlib import Path
import requests,json,hashlib,struct,zlib,subprocess,argparse
ROOT=Path(__file__).resolve().parents[1];p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--sample',action='store_true');a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
def header(kind):
    data=bytes([3,kind,0,0]);return struct.pack('<I',zlib.crc32(data))+data
sources=[];rows=[]
for volume in (10,1):
    path=ROOT/f'results/iteration81_jiuzhang_inventory_v1/volume{volume:02d}.json';d=json.loads(path.read_text());sources.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for f in d['files']:
        name=f['filename']
        wanted=('README.md' in name or name.endswith('matrix/high power.mtx') or name.endswith('sqz parameter/high power.xlsx')) if volume==10 else (a.sample and name.endswith('high power (1).bin'))
        if wanted:rows.append(dict(volume=volume,url=d['url'],entry=f))
save('plan.json',dict(scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),sources=sources,entries=rows,sample=a.sample));results=[]
for index,row in enumerate(rows):
    f=row['entry'];start=f['header_offset'];end=f['data_offset']+f['compress_size']-1;r=requests.get(row['url'],headers={'Range':f'bytes={start}-{end}'},timeout=90);assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/');payload=r.content;assert len(payload)==end-start+1
    raw=OUT/f'entry_{index}.bin';raw.write_bytes(payload);archive=OUT/f'entry_{index}.rar';archive.write_bytes(b'Rar!\x1a\x07\x01\x00'+header(1)+payload+header(5));extract=OUT/f'extracted_{index}';extract.mkdir();run=subprocess.run(['tar','-xf',str(archive),'-C',str(extract)],capture_output=True,text=True,errors='replace');target=extract/f['filename'];assert target.is_file(),run.stderr;data=target.read_bytes();assert len(data)==f['file_size'] and zlib.crc32(data)==f['CRC'],f['filename'];result=dict(filename=f['filename'],returncode=run.returncode,stdout=run.stdout,stderr=run.stderr,original_crc32=f['CRC'],uncompressed_bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),compressed_range_sha256=hashlib.sha256(payload).hexdigest(),relative_path=target.relative_to(OUT).as_posix());results.append(result);save(f'entry_{index}.json',result);print(result,flush=True)
save('summary.json',dict(passed=True,rows=results));save('manifest.json',dict(artifacts={p.relative_to(OUT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.rglob('*') if p.is_file()}))
