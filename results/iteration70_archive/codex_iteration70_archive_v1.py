from pathlib import Path
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration70_archive';OUT.mkdir(exist_ok=False)
def H(p):return hashlib.sha256(p.read_bytes()).hexdigest()
roots=sorted(p for p in (ROOT/'.git').iterdir() if p.name.startswith(('iteration70','codex_iteration70')))
rows=[]
for root in roots:
    paths=sorted(p for p in root.rglob('*') if p.is_file()) if root.is_dir() else [root]
    for source in paths:
        relative=source.relative_to(ROOT/'.git');parts=['git_source_snapshot' if x=='.git' else x for x in relative.parts];dest=OUT.joinpath(*parts);assert not dest.exists();dest.parent.mkdir(parents=True,exist_ok=True);before=H(source);shutil.copy2(source,dest);assert H(dest)==before==H(source);rows.append(dict(source=source.relative_to(ROOT).as_posix(),destination=dest.relative_to(ROOT).as_posix(),bytes=source.stat().st_size,sha256=before))
mapping=ROOT/'results/iteration70_archive_mapping.json';assert not mapping.exists();mapping.write_bytes((json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows),rows=rows),indent=2)+'\n').encode());print(json.dumps(dict(files=len(rows),bytes=sum(r['bytes'] for r in rows))))
