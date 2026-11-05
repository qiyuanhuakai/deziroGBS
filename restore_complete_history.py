#!/usr/bin/env python3
"""Verify and restore the complete original research Git history from local parts."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import subprocess
import tempfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination',type=Path,required=True)
    args=parser.parse_args()
    destination=args.destination.expanduser().resolve()
    if destination.exists():
        raise FileExistsError('Destination must be a new directory: '+str(destination))
    base=Path(__file__).resolve().parent/'research_archive'/'iteration9'
    manifest=json.loads((base/'MANIFEST.json').read_text())
    if manifest['format']!='deziroGBS-complete-git-bundle-parts-v1':
        raise ValueError('Unknown archive format')
    if not re.fullmatch('[0-9a-f]{40}',manifest['original_commit']):
        raise ValueError('Invalid expected commit')
    with tempfile.TemporaryDirectory(prefix='gbs-history-') as temp:
        bundle=Path(temp)/'history.bundle'
        digest=hashlib.sha256()
        count=0
        with bundle.open('wb') as output:
            for i,part in enumerate(manifest['parts']):
                expected_name=f'history.bundle.part{i:03d}'
                if part['name']!=expected_name:
                    raise ValueError('Invalid part order or path')
                data=(base/expected_name).read_bytes()
                if len(data)!=part['bytes'] or hashlib.sha256(data).hexdigest()!=part['sha256']:
                    raise ValueError('Part integrity failure: '+expected_name)
                output.write(data);digest.update(data);count+=len(data)
        if count!=manifest['bundle_bytes'] or digest.hexdigest()!=manifest['bundle_sha256']:
            raise ValueError('Complete bundle integrity failure')
        subprocess.run(['git','clone','--quiet','--branch',manifest['original_branch'],str(bundle),str(destination)],check=True)
    actual=subprocess.check_output(['git','-C',str(destination),'rev-parse','HEAD'],text=True).strip()
    if actual!=manifest['original_commit']:
        raise RuntimeError('Restored commit differs from expected commit')
    print(json.dumps({'destination':str(destination),'commit':actual,'archive_sha256':manifest['bundle_sha256'],'all_parts_verified':True},indent=2))

if __name__=='__main__':
    main()
