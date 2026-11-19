"""Complete the pinned MIT-licensed upstream snapshot, preserving exact bytes."""
from pathlib import Path
import subprocess,json,base64,hashlib,time,concurrent.futures
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration72_frugal_gpu_sources';OUT.mkdir(exist_ok=False)
REF='fa8b796f2afebd4845a2e335e457735952617ef8';REPO='tomdodd4598/FrugalGBS'
def api(path):return json.loads(subprocess.check_output(['gh','api',f'repos/{REPO}/'+path],text=True,encoding='utf-8'))
tree=api(f'git/trees/{REF}?recursive=1');assert not tree['truncated'];(OUT/'tree.json').write_text(json.dumps(tree,indent=2)+'\n',encoding='utf-8')
entries=[e for e in tree['tree'] if e['type']=='blob' and '/gpu/' in e['path']]
def fetch(e):
    relative=Path(e['path']);assert not relative.is_absolute() and '..' not in relative.parts
    response=api('contents/'+e['path']+'?ref='+REF);content=base64.b64decode(response['content']);sha=hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest();assert sha==e['sha']==response['sha']
    destination=ROOT/'third_party/FrugalGBS'/relative;destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert destination.read_bytes()==content
    else:destination.write_bytes(content)
    return dict(path=destination.relative_to(ROOT).as_posix(),git_blob=sha,sha256=hashlib.sha256(content).hexdigest(),bytes=len(content))
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(fetch,entries))
(OUT/'manifest.json').write_text(json.dumps(dict(upstream=REPO,commit=REF,files=rows,license='third_party/FrugalGBS/LICENSE.md',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),connector_note='Initial connector tree lookup succeeded; pending bulk raw fetches were terminated after no responses. GitHub CLI contents API completed with exact blob checks.'),indent=2)+'\n',encoding='utf-8');print(json.dumps(dict(passed=True,files=len(rows),bytes=sum(r['bytes'] for r in rows))),flush=True)
