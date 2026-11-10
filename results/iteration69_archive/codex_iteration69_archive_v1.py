from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration69_archive';OUT.mkdir(exist_ok=False)
def H(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
records=[]
for top in sorted((ROOT/'.git').iterdir()):
    if not (top.name.startswith('iteration69') or top.name.startswith('codex_iteration69')):continue
    paths=sorted(top.rglob('*')) if top.is_dir() else [top]
    for source in paths:
        if not source.is_file():continue
        relative=source.relative_to(ROOT/'.git');portable=Path(*('git_source_snapshot' if part=='.git' else part for part in relative.parts));target=OUT/portable;target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();before=H(source);shutil.copy2(source,target);assert H(target)==before==H(source)
        records.append(dict(original=source.relative_to(ROOT).as_posix(),archive=target.relative_to(ROOT).as_posix(),sha256=before,bytes=source.stat().st_size))
manifest=dict(scope='Byte-preserved first executions, failed attempts, recovery sources and final private benchmarks. Embedded .git components are mapped to git_source_snapshot without changing file contents.',artifacts=records,total_bytes=sum(r['bytes'] for r in records))
(ROOT/'results/iteration69_archive_mapping.json').write_bytes((json.dumps(manifest,indent=2)+'\n').encode());print(json.dumps(dict(files=len(records),bytes=manifest['total_bytes'])))
