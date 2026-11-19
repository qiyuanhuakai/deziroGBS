"""Read selected author ZIP members with bounded HTTP ranges; never execute them."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import zipfile
import requests

ROOT=Path(__file__).resolve().parents[1]
URL='https://zenodo.org/records/18045305/files/filtration%20of%20quantum%20channel.zip?download=1'
SIZE=1950834739


class RemoteZip(io.RawIOBase):
    def __init__(self): self.pos=0; self.requests=[]
    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else SIZE+offset
        if not 0<=self.pos<=SIZE: raise ValueError('Out-of-range ZIP seek')
        return self.pos
    def read(self,n=-1):
        if n<0: n=SIZE-self.pos
        n=min(n,SIZE-self.pos)
        if not n: return b''
        if n>2_000_000: raise ValueError('Read exceeds bounded source-audit limit')
        start=self.pos; end=start+n-1
        with requests.get(URL,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},stream=True,timeout=60) as r:
            expected=f'bytes {start}-{end}/{SIZE}'
            if r.status_code!=206 or r.headers.get('Content-Range')!=expected:
                raise IOError(f'Range not honored: {r.status_code}, {r.headers.get("Content-Range")}')
            data=r.raw.read(n+1)
            if len(data)!=n: raise IOError('Wrong range byte count')
        self.requests.append(dict(start=start,end=end,bytes=n,sha256=hashlib.sha256(data).hexdigest()))
        self.pos+=n
        return data


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True); p.add_argument('--cache',type=Path,required=True); args=p.parse_args()
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=False)
    cache=args.cache.resolve(); cache.mkdir(parents=True,exist_ok=False)
    source=Path(__file__); initial=source.read_bytes(); (out/source.name).write_bytes(initial)
    def save(name,data): (out/name).write_text(json.dumps(data,indent=2)+'\n',encoding='utf8')
    save('plan.json',dict(url=URL,archive_bytes=SIZE,published_md5='8cbbb641abc55c1859fe1908ed29dcf8',whole_archive_md5_verified=False,source_sha256=hashlib.sha256(initial).hexdigest(),policy='Read metadata and selected small members only. ZIP CRC checked by zipfile; no author code execution or redistribution.'))
    remote=RemoteZip(); selected=[]
    try:
        with zipfile.ZipFile(remote) as z:
            inventory=[dict(name=x.filename,bytes=x.file_size,compressed_bytes=x.compress_size,crc32=f'{x.CRC:08x}',header_offset=x.header_offset) for x in z.infolist()]
            save('inventory.json',inventory)
            for item in z.infolist():
                name=item.filename
                if not name.startswith('filtration of quantum channel/') or name.endswith('/') or '/._' in name: continue
                if not ('jiuzhang2' in name.lower() or name.endswith('.py')): continue
                if item.file_size>2_000_000: raise ValueError('Selected member exceeds audit limit')
                data=z.read(item); target=cache/Path(name).name
                if target.exists(): raise FileExistsError(target)
                target.write_bytes(data)
                row=dict(name=name,local_name=target.name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32=f'{item.CRC:08x}',zip_crc_verified=True)
                selected.append(row); save('selected.json',selected)
        save('summary.json',dict(status='passed',members=len(inventory),selected=len(selected),downloaded_range_bytes=sum(r['bytes'] for r in remote.requests),source_unchanged=source.read_bytes()==initial,whole_archive_md5_verified=False))
    finally:
        save('ranges.json',remote.requests)
    assert source.read_bytes()==initial
    print(json.dumps(json.loads((out/'summary.json').read_text())))


if __name__=='__main__': main()
