"""Read-back audit against pre-existing immutable baseline build evidence.
This records provenance after quality execution; it does not pretend these
additional records were a pre-draw freeze. The artifacts match earlier pins.
"""
import argparse,subprocess
from iteration93_common_v5 import *
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False);rows=[]
for folder,name in (('iteration89_gates_v1','puretree_probe.dll'),('iteration91_build_v3','optical.dll'),('iteration75_frugal_cpu_build_v1','frugal_cpu_fp64.dll')):
    manifest_path=ROOT/'results'/folder/'manifest.json';record=read(manifest_path);record=record.get('artifacts',record);path=ROOT/'results'/folder/name;assert H(path)==record[name];rows.append(dict(path=path.relative_to(ROOT).as_posix(),sha256=H(path),prior_manifest=manifest_path.relative_to(ROOT).as_posix(),prior_manifest_sha256=H(manifest_path)))
vendor=read(ROOT/'results/iteration73_frugal_cpu_build_v5/plan.json')['hashes'];checked={}
for name,h in vendor.items():
    if name.startswith('third_party/FrugalGBS/'):
        assert H(ROOT/name)==h,name;checked[name]=h
for name,h in read(ROOT/'results/iteration75_frugal_cpu_build_v1/plan.json')['header_hashes'].items():
    path=ROOT/'results/iteration75_frugal_cpu_build_v1'/name;assert H(path)==h,name;checked[path.relative_to(ROOT).as_posix()]=h
pe=ROOT.parent/'external_sources/photonic_engines_iteration92';head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=pe,text=True).strip();assert head=='6c1ed6f8f11ffc9f3c268757f0fe0bd48a768d82';assert not subprocess.check_output(['git','status','--porcelain'],cwd=pe,text=True).strip();pehash={str(path.relative_to(pe)):H(path) for path in (pe/'photonic_engines').rglob('*.py')};save(out,'pe_sources.json',pehash)
import scipy,torch
save(out,'validation.json',dict(passed=True,scope=__doc__,binary_pins=rows,frugal_checked_files=len(checked),frugal_source_hashes=checked,pe_commit=head,pe_clean=True,pe_sources=len(pehash),numpy=np.__version__,scipy=scipy.__version__,torch=torch.__version__,source_sha256=H(Path(__file__))));manifest(out);print(dict(passed=True,binaries=len(rows),frugal_files=len(checked),pe_sources=len(pehash)))
