"""Bound a live research-owned GPU baseline at 600 seconds; preserve interrupted evidence."""
from pathlib import Path
import argparse,ctypes as C,subprocess,json,time,hashlib,datetime
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();OUT=a.output.resolve();OUT.mkdir(parents=True,exist_ok=False);BASE=ROOT/'results/iteration75_scale_click_v2'
def save(name,obj):
    with (OUT/name).open('xb') as f:f.write((json.dumps(obj,indent=2)+'\n').encode())
query=f"Get-CimInstance Win32_Process -Filter 'ProcessId = {a.pid}' | Select-Object ProcessId,ParentProcessId,CommandLine,CreationDate | ConvertTo-Json -Compress"
r=subprocess.run(['pwsh','-NoProfile','-Command',query],capture_output=True,text=True,encoding='utf-8',errors='replace');assert r.returncode==0,r.stderr;process=json.loads(r.stdout);assert 'iteration75_scale_click_v2.py' in process['CommandLine']
kernel=C.WinDLL('kernel32',use_last_error=True);kernel.OpenProcess.argtypes=[C.c_uint32,C.c_int,C.c_uint32];kernel.OpenProcess.restype=C.c_void_p;kernel.WaitForSingleObject.argtypes=[C.c_void_p,C.c_uint32];kernel.WaitForSingleObject.restype=C.c_uint32;kernel.TerminateProcess.argtypes=[C.c_void_p,C.c_uint32];kernel.TerminateProcess.restype=C.c_int;kernel.CloseHandle.argtypes=[C.c_void_p]
handle=kernel.OpenProcess(0x00100000|0x1000|1,False,a.pid);assert handle,C.get_last_error();save('plan.json',dict(process=process,budget_seconds=600,scope=__doc__,reason='Large GPU sampling blocks new algorithm validation; complete CPU/fused panels will be rerun separately. No censored row is a successful timing.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()));started=time.monotonic();stopped=False;pending=None
try:
    while kernel.WaitForSingleObject(handle,0)==258:
        active=[f for f in BASE.glob('started_*.json') if not (BASE/f.name.replace('started_','row_',1)).exists()]
        if active:
            latest=max(active,key=lambda f:f.stat().st_mtime);row=json.loads(latest.read_text());elapsed=time.time()-latest.stat().st_mtime
            if row['method']=='gpu64' and elapsed>=600:
                pending=dict(panel=row,start_file=latest.name,observed_wall_since_started_file=elapsed,supervisor_elapsed_seconds=time.monotonic()-started,utc=datetime.datetime.now(datetime.timezone.utc).isoformat());save('stop_request.json',pending)
                assert kernel.TerminateProcess(handle,75),C.get_last_error();assert kernel.WaitForSingleObject(handle,60000)==0;stopped=True;break
        time.sleep(1)
finally:kernel.CloseHandle(handle)
files={f.relative_to(BASE).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in BASE.rglob('*') if f.is_file()};save('receipt.json',dict(status='operator_budget_interrupted' if stopped else 'completed_before_budget',pending=pending,completed_rows=len(list(BASE.glob('row_*.json'))),summary_present=(BASE/'summary.json').exists(),artifact_hashes=files));print(json.dumps(dict(stopped=stopped,completed_rows=len(list(BASE.glob('row_*.json'))),pending=pending)),flush=True)
