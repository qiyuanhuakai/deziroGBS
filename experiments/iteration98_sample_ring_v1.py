"""Draw complete physical port records for a closed Fourier-cell ring."""
import argparse
import ctypes as ct
import os
import shutil
from fractions import Fraction
from pathlib import Path
from iteration98_cycle_v1 import ROOT, CAP, P, np, bind, I, F, W, T, D, U, B, draw, tables, couplers

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--cells',type=int,required=True)
    parser.add_argument('--route',choices=['pnr','click'],required=True)
    parser.add_argument('--shots',type=int,default=2048)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--seed',type=int,required=True)
    parser.add_argument('--offset',type=int,default=0)
    parser.add_argument('--phase',type=float,default=0.,help='phase on the right arm of the closing cell, radians')
    args=parser.parse_args()
    if not (2<=args.cells<=16384 and 1<=args.shots<2**31 and 1<=args.workers<=64 and
            0<=args.seed<2**64 and 0<=args.offset<=2**32-args.shots and np.isfinite(args.phase)):
        parser.error('invalid dimensions, phase, workers or RNG address')
    runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
    lib=ct.CDLL(str(ROOT/'results/iteration98_cycle_v1/cycle.dll'))
    fn=bind(lib,'cycle98_draw',[I,I,I,I,I,F,W,T,T,T,D,D,W,I,U,U,I,B,B])
    local,_=tables(lib,CAP,P)
    edges=couplers(CAP,Fraction(1,2))
    route=int(args.route=='pnr')
    x,flags=draw(fn,local[route],edges,args.cells,route,args.phase,1,args.shots,args.seed,args.offset,args.workers)
    with args.output.open('xb') as stream:
        np.savez_compressed(stream,records=x,ambiguous=flags,cells=args.cells,route=args.route,
                            phase=args.phase,seed=np.uint64(args.seed),offset=args.offset,p=float(P),cap=CAP)
if __name__=='__main__':main()
