"""An independent Gaussian optical reference for the complete 16-mode ring."""
import argparse
import ctypes as ct
import os
import shutil
from pathlib import Path
from iteration97_bridges_v1 import network,covariance
from iteration96_group_common_v1 import ROOT, I, D, H, np, bind, save, manifest

def main():
    p=argparse.ArgumentParser()
    p.add_argument('output',type=Path)
    out=p.parse_args().output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    dll=ROOT/'results/iteration96_quad_v1/oracle.dll'
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    oracle=bind(ct.CDLL(str(dll)),'oracle96',[I,I,D,D,D,D])
    v=network(2,0)[0]
    bs=np.array([[1,1],[-1,1]])/np.sqrt(2)
    for left,right in [(3,4),(7,0)]:v[[left,right]]=bs@v[[left,right]]
    q,b=covariance(v,.0576)
    mu=np.zeros((1,32))
    law=np.empty(65536)
    diag=np.empty(2)
    assert oracle(16,1,q,mu,law,diag)==0
    assert abs(law.sum()-1)<1e-12 and law.min()>-1e-13
    np.savez_compressed(out/'input.npz',v=v,barg=b,p=np.full(8,.0576),q=q,mu=mu,click=law)
    save(out,'plan.json',dict(oracle=H(dll),source=H(Path(__file__)),diagnostics=diag.tolist()))
    manifest(out)
if __name__=='__main__':main()
