"""Losslessly package/restore the large frozen NPZ artifacts for GitHub."""
import argparse
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PACKAGE=ROOT/'research_archive/publication/iteration21_large_artifacts'


def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['package','restore','verify']); args=p.parse_args()
    frozen=json.loads((ROOT/'results/iteration21_manifest.json').read_text())['artifacts']
    if args.action=='package':
        PACKAGE.mkdir(parents=True,exist_ok=False); records={}
        for path,digest in frozen.items():
            source=ROOT/path
            if source.stat().st_size<=90*2**20: continue
            parts=[]; whole=hashlib.sha256()
            with source.open('rb') as f:
                for i in range(1000):
                    block=f.read(40*2**20)
                    if not block: break
                    dest=PACKAGE/f'{source.name}.part{i:03d}'; dest.write_bytes(block); whole.update(block)
                    parts.append(dict(path=dest.relative_to(ROOT).as_posix(),bytes=len(block),sha256=hashlib.sha256(block).hexdigest()))
            assert whole.hexdigest()==digest
            records[path]=dict(bytes=source.stat().st_size,sha256=digest,parts=parts)
        (PACKAGE/'manifest.json').write_text(json.dumps(dict(artifacts=records),indent=2)+'\n',encoding='utf-8')
        print(json.dumps(dict(packaged=len(records),parts=sum(len(r['parts']) for r in records.values())))); return
    records=json.loads((PACKAGE/'manifest.json').read_text())['artifacts']
    for path,record in records.items():
        digest=hashlib.sha256(); size=0
        for part in record['parts']:
            data=(ROOT/part['path']).read_bytes(); assert len(data)==part['bytes'] and hashlib.sha256(data).hexdigest()==part['sha256']
            digest.update(data); size+=len(data)
        assert size==record['bytes'] and digest.hexdigest()==record['sha256']==frozen[path]
        target=ROOT/path
        if target.exists():
            assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']
        elif args.action=='restore':
            target.parent.mkdir(parents=True,exist_ok=True)
            with target.open('xb') as f:
                for part in record['parts']: f.write((ROOT/part['path']).read_bytes())
            assert hashlib.sha256(target.read_bytes()).hexdigest()==record['sha256']
    print(json.dumps(dict(verified_original_artifacts=len(records),action=args.action)))


if __name__=='__main__': main()
