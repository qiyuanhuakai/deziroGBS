"""Check stored distributions, matched replays, table ranges and archive hashes."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
import numpy as np

from iteration115_visible_v1 import ROOT
from iteration118_auxiliary import OUT


def read(name):
    return json.loads((OUT/(name+".json")).read_text(encoding="utf8"))


def verify(write):
    result=read("cap14_weak10_8")
    assert abs(result["full_TV_upper"]-np.sqrt(1-np.prod(result["retained"])))<1e-14
    for name in ("cap14_weak10_2","cap14_weak10_8","cap10_128"):
        info=read(name)
        with np.load(OUT/(name+".npz")) as data:
            for route,key in zip(info["rows"],("pnr","click")):
                a=data[key];assert a.shape==(route["shots"],1152)
                assert abs((a>0).sum(1).mean()-route["mean_clicks"])<1e-12
                if key=="click":assert a.max()<=1
                else:assert abs(a.sum(1).mean()-route["mean_photons"])<1e-12
    with np.load(OUT/"cap14_weak10_2.npz") as old,np.load(OUT/"cap14_weak10_8.npz") as new:
        assert all(np.array_equal(old[key],new[key][:2]) for key in old)
    for suffix in ("","_71_72","_102_143"):
        stats=read("panel_7_5_28"+suffix)
        dist=read("panel_distance"+suffix)
        assert max(stats["full_panel_click_tv"],stats["bank_pnr_modulo_tv"])<dist["panel_pnr_tv_upper"]
    with np.load(OUT/"panel_7_5_28.npz") as a,np.load(OUT/"panel_7_5_32.npz") as b:
        assert all(np.abs(a[key]-b[key]).sum()/2<1e-8 for key in a)
    assert read("environment_7_5")["upper99"]<result["full_TV_upper"]
    assert read("law_7_5")["total_number_tv_upper"]<.00018
    assert all(row["changed_rank_one_outcomes"]==0 for row in read("replay"))
    paper=ROOT/"papers/number-tilted-gbs"
    with zipfile.ZipFile(paper/"latex-source.zip") as source:
        assert source.testzip() is None
        for name in source.namelist():assert source.read(name)==(paper/name).read_bytes()
    files=[*sorted((ROOT/"experiments").glob("iteration118_*.py")),ROOT/"native/panel_void118.cpp",
           *sorted(paper.glob("*")),*(p for p in sorted(OUT.iterdir())
                                     if p.suffix in (".npz",".json") and p.name!="manifest.json")]
    files=[p for p in files if p.is_file()]
    digests={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    target=OUT/"manifest.json"
    if write:target.write_bytes((json.dumps(digests,indent=2)+"\n").encode())
    else:assert digests==json.loads(target.read_text(encoding="utf8"))
    print("Verified",len(digests),"files.")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--write",action="store_true")
    verify(parser.parse_args().write)
