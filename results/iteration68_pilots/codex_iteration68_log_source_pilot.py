"""Preserve bounded official author-log downloads and inventory; no code execution."""
from pathlib import Path,PurePosixPath
import hashlib,json,datetime,urllib.request,zipfile,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.git/iteration68_log_source_pilot_20261110'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,obj):
    assert not p.exists();p.write_bytes((json.dumps(obj,indent=2)+'\n').encode())
def main():
    OUT.mkdir(exist_ok=False);source=Path(__file__);shutil.copy2(source,OUT/'source.py');rows=[]
    for name,url,expected in [('time_log.zip','https://zenodo.org/records/18045305/files/time%20log.zip?download=1','d5a519b78cc63ccda4760cb2a0175579'),('readme.txt','https://zenodo.org/records/18045305/files/readme.txt?download=1','fd46bbef42d74cad3d5abebe9e30fe5a')]:
        req=urllib.request.Request(url,headers={'User-Agent':'deziroGBS research source audit'})
        with urllib.request.urlopen(req,timeout=45) as response:
            payload=response.read(2_000_001);headers=dict(response.headers);final_url=response.url
        assert len(payload)<=2_000_000
        p=OUT/name;p.write_bytes(payload);actual=hashlib.md5(payload).hexdigest();row=dict(name=name,url=url,final_url=final_url,bytes=len(payload),sha256=digest(p),expected_md5=expected,actual_md5=actual,md5_matches=actual==expected,headers=headers);rows.append(row);save(OUT/(name+'.receipt.json'),row);assert actual==expected
    inventory=[]
    with zipfile.ZipFile(OUT/'time_log.zip') as z:
        for i in z.infolist():
            if i.is_dir():continue
            p=PurePosixPath(i.filename);assert not p.is_absolute() and '..' not in p.parts
            payload=z.read(i.filename);inventory.append(dict(name=i.filename,bytes=len(payload),compressed=i.compress_size,sha256=hashlib.sha256(payload).hexdigest(),text_preview=payload.decode('utf-8',errors='replace')[:300]))
    save(OUT/'inventory.json',inventory);assert digest(source)==digest(OUT/'source.py');save(OUT/'summary.json',dict(completed=True,checked_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),downloads=rows,archive_members=len(inventory),source_sha256=digest(source),scope='Official author logs downloaded and checksummed,not a reproduced runtime benchmark. Archive contents not executed.'))
    save(OUT/'pilot_manifest.json',dict(artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file()},scope='Source/inventory pilot; immutable raw bytes.'))
    print(json.dumps(dict(downloads=[{k:r[k] for k in ('name','bytes','md5_matches')} for r in rows],archive_members=len(inventory)),indent=2))
if __name__=='__main__':main()
