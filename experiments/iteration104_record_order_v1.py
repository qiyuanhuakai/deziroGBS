"""Evaluate the fixed optical KS witness on consecutive windows in the original hardware file."""
import argparse
import ctypes as ct
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from iteration100_spatial_v1 import ROOT,HIST,I,B,SHORT,bind,H,save,manifest
from iteration103_sample_v1 import DATA
from iteration104_sample_v1 import FIT


def main(out,raw):
    out.mkdir(exist_ok=False,parents=True)
    plan=json.loads((HIST.parent/'plan.json').read_text())
    recorded,digest=next((k,v) for k,v in plan['hashes'].items() if k.endswith('high power (1).bin'))
    path=raw or Path(recorded)
    assert H(path)==digest
    binary=np.memmap(path,np.uint8,mode='r').reshape(-1,144)
    source=ROOT/'native/ordered_click_totals104_v1.cpp'
    dll=out/'totals.dll'
    subprocess.run([shutil.which('g++'),'-O3','-std=c++20','-march=x86-64-v3','-shared','-fopenmp',str(source),'-o',str(dll)],check=True)
    import os
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    fn=bind(ct.CDLL(str(dll)),'ordered104',[B,I,SHORT],None)
    c=np.empty(len(binary),np.uint16)
    fn(binary,len(c),c)
    hardware=np.load(HIST)['total_click_histogram']
    assert np.array_equal(np.bincount(c,minlength=len(hardware)),hardware)
    z=np.load(FIT/'witness.npz')
    w=z['h'][c.clip(0,len(z['h'])-1)]
    upper=float(z['upper'])
    rows=[]
    hist=[]
    for i,idx in enumerate(np.array_split(np.arange(len(c)),64)):
        vals=c[idx]
        h=np.bincount(vals,minlength=len(hardware))
        rows.append(dict(window=i,first=int(idx[0]),last=int(idx[-1]),records=len(idx),mean=float(vals.mean()),variance=float(vals.var()),empirical_optical_ks_lower=max(0.,float(w[idx].mean()-upper))))
        hist.append(h)
    segments=np.array(hist)
    first=segments[:32].sum(0);last=segments[32:].sum(0)
    ks=float(abs(first.cumsum()/first.sum()-last.cumsum()/last.sum()).max())
    correlations={}
    for lag in (1,2,8,64,1024):
        correlations[str(lag)]=dict(click=float(np.corrcoef(c[:-lag],c[lag:])[0,1]),witness=float(np.corrcoef(w[:-lag],w[lag:])[0,1]))
    np.savez_compressed(out/'windows.npz',histograms=segments)
    save(out,'summary.json',dict(source_sha256=digest,records=len(c),windows=rows,first_half_vs_last_half_ks=ks,lag_correlations=correlations,empirical_optical_lower_range=[min(v['empirical_optical_ks_lower'] for v in rows),max(v['empirical_optical_ks_lower'] for v in rows)],scope='Original file order, no time intervals or population independence assumed. The numerical optical reference is fixed.'))
    manifest(out)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--raw',type=Path)
    a=p.parse_args();main(a.output.resolve(),a.raw)
