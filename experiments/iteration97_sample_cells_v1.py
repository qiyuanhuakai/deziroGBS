"""Draw records for a chain of four-pair Fourier cells connected by terminal beamsplitters."""
import argparse
import ctypes as ct
from fractions import Fraction
from pathlib import Path
from iteration97_forest_v1 import local_tables,edge_tables
from iteration97_conditionals_v1 import ROOT,Kernel,np,I,U,W,B,bind
from iteration96_group_common_v1 import T

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output',type=Path)
    p.add_argument('--route',choices=['click','pnr'],required=True);p.add_argument('--cells',type=int,required=True)
    p.add_argument('--pair-probability',type=Fraction,default=Fraction(36,625))
    p.add_argument('--reflectivity',type=Fraction,required=True);p.add_argument('--max-pairs',type=int,default=7)
    p.add_argument('--shots',type=int,default=2048);p.add_argument('--workers',type=int,default=1);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--offset',type=int,default=0);a=p.parse_args()
    if not (1<=a.cells<=16384 and 0<a.pair_probability<1 and 0<=a.reflectivity<=1 and 1<=a.max_pairs<=9 and
        1<=a.workers<=64 and 0<a.shots<2**31 and 0<=a.seed<2**64 and 0<=a.offset<=2**32-a.shots):p.error('invalid optical input or record interval')
    runtime=Kernel();dll=ROOT/'results/iteration97_forest_v1/forest.dll';lib=ct.CDLL(str(dll))
    draw=bind(lib,'forest97_draw',[I,I,I,I,I,W,T,T,W,I,U,U,I,B]);cap=a.max_pairs;route=int(a.route=='pnr')
    tables,survival=local_tables(lib,cap,a.pair_probability);edge,edge_tv=edge_tables(cap,a.reflectivity)
    bits,cuts,alias,keys,quant,_=tables[route];x=np.empty((a.shots,8*a.cells),np.uint8)
    rc=draw(a.cells,route,bits,cap,cap,cuts,alias,keys,edge[route],a.shots,a.seed,a.offset,a.workers,x)
    if rc:raise RuntimeError(f'cell sampler error {rc}')
    # log form stays usable for very long chains without constructing enormous rational powers.
    tail=float(-np.expm1(a.cells*np.log1p(-float(1-survival))))
    upper=min(1.,tail+a.cells*quant+(a.cells-1)*edge_tv[route])
    with a.output.open('xb') as file:np.savez_compressed(file,records=x,full_tv_upper=upper,tail=tail,route=a.route,
        reflectivity=str(a.reflectivity),pair_probability=str(a.pair_probability),max_cell_pairs=cap)
    print(a.route,x.shape,'full TV upper',upper)
if __name__=='__main__':main()
