"""Generate full physical records using the archived optical input or a lossy closed ring."""
import argparse
import ctypes as ct
import hashlib
import json
import os
import shutil
from fractions import Fraction
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results'
PANEL=DATA/'iteration99_release_v1/prepared.npz'

class PreparedActual:
    def __init__(self,path=PANEL,verify=True):
        if verify:
            digest=json.loads((path.parent/'manifest.json').read_text())[path.name]
            if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('prepared table hash mismatch')
        with np.load(path) as z:
            for name in ['counts','cut','alias','group','keep','single','dcut','dalias','words']:
                setattr(self,name,z[name])
        self.runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        self.lib=ct.CDLL(str(DATA/'iteration99_marks_v1/marks.dll'))
        I,U=ct.c_int,ct.c_uint64
        W,T,J,S,B=[np.ctypeslib.ndpointer(dtype=t,flags='C_CONTIGUOUS') for t in (np.uint64,np.uint32,np.int32,np.uint16,np.uint8)]
        self.draw=self.lib.mark99_draw
        self.draw.argtypes=[I,I,I,I,I,W,W,T,J,I,W,W,I,W,T,T,I,U,U,I,S,B]
        self.draw.restype=I
    def sample(self,n,route,seed,offset,workers):
        modes=len(self.single);sources=len(self.counts);bits=(self.cut.shape[1]-1).bit_length()
        x=np.empty((n,modes),np.uint16);flags=np.empty(n,np.uint8)
        rc=self.draw(modes,sources,self.counts.shape[1]-1,bits,route,self.counts,self.cut,self.alias,self.group,len(self.group),
             self.keep,self.single,(len(self.dcut)-1).bit_length(),self.dcut,self.dalias,self.words,n,seed,offset,workers,x,flags)
        if rc or (flags&16).any():raise RuntimeError('prepared draw failed')
        return x,flags

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('model',choices=['actual','ring'])
    p.add_argument('output',type=Path)
    p.add_argument('--route',choices=['click','pnr'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--offset',type=int,default=0)
    p.add_argument('--shots',type=int,default=2048);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--cells',type=int,help='ring cells, default 144')
    p.add_argument('--phase',type=float,help='closing phase in radians, default zero')
    args=p.parse_args();route=int(args.route=='pnr')
    if not (0<=args.seed<2**64 and 1<=args.shots<2**31 and 0<=args.offset<=2**32-args.shots and 1<=args.workers<=64):
        p.error('invalid RNG address, batch size or worker count')
    if args.model=='actual':
        if args.cells is not None or args.phase is not None:p.error('cells and phase apply to the ring')
        x,flags=PreparedActual().sample(args.shots,route,args.seed,args.offset,args.workers)
    else:
        from iteration99_integer_v1 import IntegerRing
        cells=144 if args.cells is None else args.cells;phase=0. if args.phase is None else args.phase
        if not 2<=cells<=16384 or not np.isfinite(phase):p.error('invalid ring dimensions or phase')
        runtime=os.add_dll_directory(str(Path(shutil.which('g++')).parent))
        model=IntegerRing(ct.CDLL(str(DATA/'iteration99_loss_v1/loss.dll')),ct.CDLL(str(DATA/'iteration99_integer_v1/integer.dll')))
        x,flags=model.sample(cells,route,args.shots,args.seed,args.offset,args.workers,phase)
    with args.output.open('xb') as f:
        np.savez_compressed(f,records=x,flags=flags,model=args.model,route=args.route,seed=np.uint64(args.seed),
                            offset=args.offset,**({'cells':cells,'phase':phase} if args.model=='ring' else {}))
if __name__=='__main__':main()
