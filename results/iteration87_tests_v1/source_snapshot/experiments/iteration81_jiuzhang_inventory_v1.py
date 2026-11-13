"""Read official Jiuzhang3 RAR volume headers using verified HTTP byte ranges.
No archive code is executed. Range bytes and hashes are preserved for provenance.
"""
from pathlib import Path
import sys,io,requests,json,hashlib,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'.cache/iteration81_tools'));import rarfile
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
save('plan.json',dict(scope=__doc__,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),source_page='https://quantum.ustc.edu.cn/node/1121',volumes=[10,1],maximum_fetched_per_volume=16777216,rarfile_version=rarfile.__version__))
class Remote(io.RawIOBase):
    def __init__(self,volume):
        self.volume=volume;self.url=f'https://quantum.ustc.edu.cn/sites/default/files/2023-05/jiuzhang3.part{volume:02d}.rar';self.pos=0;self.cache={};self.session=requests.Session();self.size=None;self.total=0;self.ranges=[];self.block(0)
    def block(self,start):
        if start in self.cache:return self.cache[start]
        end=start+65535
        if self.size is not None:end=min(end,self.size-1)
        assert self.total+end-start+1<=16777216
        r=self.session.get(self.url,headers={'Range':f'bytes={start}-{end}'},timeout=30);assert r.status_code==206,(r.status_code,r.headers);cr=r.headers['Content-Range'];assert cr.startswith(f'bytes {start}-{end}/'),cr;self.size=int(cr.split('/')[-1]);data=r.content;assert len(data)==end-start+1
        name=f'volume{self.volume:02d}_{start}.bin';(OUT/name).write_bytes(data);self.ranges.append(dict(start=start,end=end,bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),file=name,etag=r.headers.get('ETag')));self.total+=len(data);self.cache[start]=data;return data
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos);assert n<=1048576;parts=[]
        while n>0:
            start=self.pos//65536*65536;data=self.block(start);offset=self.pos-start;take=min(n,len(data)-offset);assert take>0;parts.append(data[offset:offset+take]);self.pos+=take;n-=take
        return b''.join(parts)
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset;return self.pos
    def tell(self):return self.pos
    def seekable(self):return True
    def readable(self):return True
rows=[]
for volume in (10,1):
    remote=Remote(volume)
    try:
        archive=rarfile.RarFile(remote,part_only=True,errors='strict');files=[]
        for f in archive.infolist():
            fields={k:v for k,v in vars(f).items() if isinstance(v,(str,int,float,bool,type(None)))};files.append(fields)
        row=dict(volume=volume,url=remote.url,size=remote.size,downloaded=remote.total,solid=archive.is_solid(),files=files,ranges=remote.ranges,success=True);archive.close()
    except Exception as e:row=dict(volume=volume,url=remote.url,size=remote.size,downloaded=remote.total,ranges=remote.ranges,success=False,error=repr(e))
    rows.append(row);save(f'volume{volume:02d}.json',row);print(json.dumps({k:row[k] for k in ('volume','size','downloaded','success')}),flush=True)
    if row['success']:print([(f.get('filename'),f.get('file_size'),f.get('compress_size'),f.get('header_offset')) for f in row['files']],flush=True)
save('summary.json',dict(rows=rows));save('manifest.json',dict(artifacts={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()}))
