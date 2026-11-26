"""Check the actual MSVC Frugal worker seeds and independently rebuild the same-device comparison."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from iteration100_spatial_v1 import ROOT,H,manifest,save
from iteration103_sample_v1 import DATA


def worker(seed):
    mask=(1<<64)-1
    mt=[seed]
    for i in range(1,312):
        mt.append((6364136223846793005*(mt[-1]^(mt[-1]>>62))+i)&mask)
    combined=(mt[0]&0xffffffff80000000)|(mt[1]&0x7fffffff)
    y=mt[156]^(combined>>1)^(0xb5026f5aa96619e9 if combined&1 else 0)
    y^=(y>>29)&0x5555555555555555
    y^=(y<<17)&0x71d67fffeda60000
    y^=(y<<37)&0xfff7eee000000000
    y^=y>>43
    return y&0xffffffff


def main(out):
    out.mkdir(parents=True,exist_ok=False)
    directory=DATA/'iteration104_frugal_device_v1'
    files=json.loads((directory/'manifest.json').read_text())
    assert all(H(directory/f)==v for f,v in files.items())
    data=json.loads((directory/'summary.json').read_text())
    outer=[]
    for panel,n in enumerate((8,8,144)):
        for rep in range(7 if n==8 else 4):
            for i in range(2,6 if n==8 else 4):
                outer.extend(104091000+1000*panel+50*rep+2*i+j for j in range(2))
        if n==8:
            for i in range(2,6):
                for part,records in enumerate((32768,262144)):
                    for k in range(records//8192):
                        outer.extend(104100000+20000*panel+2000*i+500*part+2*k+j for j in range(2))
    assert len(outer)==len(set(outer))
    vc=Path(os.environ['ProgramFiles(x86)'])/'Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
    with tempfile.TemporaryDirectory() as tmp:
        tmp=Path(tmp)
        (tmp/'env.bat').write_text(f'@call "{vc}" >nul\n@set\n',encoding='ascii')
        env=os.environ.copy()
        for line in subprocess.check_output(['cmd','/d','/c','env.bat'],cwd=tmp,text=True,errors='replace').splitlines():
            if '=' in line:
                k,v=line.split('=',1)
                if k:env[k]=v
        from shutil import which
        cl=which('cl',path=env.get('Path',env.get('PATH')))
        src=ROOT/'native/frugal_device_seed104_v1.cpp'
        subprocess.run([cl,'/nologo','/std:c++20','/O2','/EHsc','/I'+str(DATA/'iteration75_frugal_cpu_build_v1/generated'),'/I'+str(ROOT/'third_party/FrugalGBS/include'),str(src),'/Fe:audit.exe'],cwd=tmp,env=env,check=True)
        result=subprocess.check_output([str(tmp/'audit.exe')],input='\n'.join(map(str,outer))+'\n',text=True)
    actual=list(map(int,result.split()))
    assert actual==list(map(worker,outer)) and len(set(actual))==len(actual)
    rows=[]
    for row in data['timing']:
        assert row['us']==np.median(row['timing'])
        info={k:row[k] for k in ('panel','modes','method','K','batch','us')}
        if row['modes']==8:
            ref=next(r for r in data['accuracy'] if (r['panel'],r['method'],r['K'])==(row['panel'],row['method'],row['K']))
            z=np.load(directory/f"panel_{row['panel']}_{row['method']}_{row['K']}.npz")
            assert z['hold'].sum()==ref['n'] and z['trial'].sum()==32768 and abs(z['truth'].sum()-1)<5e-12
            tv=float(abs(z['hold']/ref['n']-z['truth']).sum()/2)
            assert abs(tv-ref['empirical_tv'])<1e-13
            for key in ('empirical_tv','lower','upper'):info[key]=ref[key]
        rows.append(info)
    np.savez_compressed(out/'worker_seeds.npz',external=np.asarray(outer,np.uint32),device=np.asarray(actual,np.uint32))
    save(out,'validation.json',dict(device='NVIDIA GeForce RTX 5080 Laptop GPU',rows=rows,observed_ports=data['ports'],unique_external_seeds=len(outer),unique_official_device_seeds=len(set(actual)),protocol='Independent validation records, 262144 per eight-detector case. GPU Frugal transfers two independently sampled spectra, combined on the host. Calibrated methods return the full machine and copy the selected uint8 result.',source_manifest=H(directory/'manifest.json')))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path)
    main(p.parse_args().output.resolve())
