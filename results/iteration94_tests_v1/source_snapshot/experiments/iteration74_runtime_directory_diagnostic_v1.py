"""Isolate Windows duplicate DLL-directory registration capacity."""
from pathlib import Path
import os,json,hashlib,shutil
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/iteration74_runtime_directory_diagnostic_v1';OUT.mkdir(exist_ok=False)
paths=json.loads((ROOT/'results/iteration70_tests/native_libraries.json').read_text());metadata=Path(paths['GBSLAB_WIRE_FOREST_LIBRARY']['path']).parent/'build.json';dep=Path(json.loads(metadata.read_text())['runtime_dependencies'][0]['path']).parent;handles=[];failure=None
try:
    for i in range(2048):handles.append(os.add_dll_directory(str(dep)))
except OSError as e:failure=dict(index=len(handles),error=str(e),winerror=e.winerror)
finally:
    for handle in handles:handle.close()
with os.add_dll_directory(str(dep)):recovered=True
result=dict(path=str(dep),path_length=len(str(dep)),registered=len(handles),failure=failure,recovered=recovered,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(OUT/'summary.json').write_bytes(json.dumps(result,indent=2).encode());shutil.copy2(__file__,OUT/Path(__file__).name);print(result,flush=True)
