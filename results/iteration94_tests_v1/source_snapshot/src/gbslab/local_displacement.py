"""Matrix-free correlated Gaussian displacement through local passive gates.

For real standard normals z, alpha = i sqrt(Delta)/2 U z. Keeping z real is
essential: this noise is generally noncircular and correlates alpha with alpha.
No claim about the finite-precision Gaussian RNG distribution is made here.
"""
import math
import numpy as np
from .local_mps import _integer


class LocalPropagator:
    """Validated, compiled disjoint layers; no dense global interferometer."""
    def __init__(self, modes, gates):
        self.modes=_integer(modes,'modes',2,1024)
        self.layers=[]; phases=None; starts=[]; pairs=[]; occupied=set()
        def finish():
            if phases is not None or starts:
                factor=np.ones(self.modes,complex) if phases is None else np.exp(1j*phases)
                self.layers.append((factor,np.array(starts,dtype=np.intp),
                    np.array(pairs,complex).reshape(-1,2,2)))
        for gate in gates:
            if gate[0]=='phases':
                finish(); starts=[]; pairs=[]; occupied=set()
                phases=np.array(gate[1],dtype=float,copy=True)
                if phases.shape!=(self.modes,) or not np.all(np.isfinite(phases)):
                    raise ValueError('Require finite phase for each mode')
            elif gate[0]=='pair':
                i=_integer(gate[1],'pair index',0,self.modes-2)
                u=np.array(gate[2],dtype=complex,copy=True)
                if u.shape!=(2,2) or not np.all(np.isfinite(u)) or not np.allclose(u.conj().T@u,np.eye(2),atol=1e-12,rtol=0):
                    raise ValueError('Require unitary two-mode gate')
                if i in occupied or i+1 in occupied:
                    raise ValueError('Require disjoint pairs between phase layers')
                occupied.update((i,i+1)); starts.append(i); pairs.append(u)
            else: raise ValueError('Unknown gate kind')
        finish()

    @property
    def array_bytes(self):
        """Compiled arrays only, excluding Python containers and input gates."""
        return sum(x.nbytes for layer in self.layers for x in layer)

    def apply(self, vectors):
        """Return row-batch vectors @ U.T. Does not modify caller's input."""
        out=np.array(vectors,dtype=complex,copy=True,order='C')
        if out.ndim!=2 or out.shape[1]!=self.modes or not np.all(np.isfinite(out)):
            raise ValueError('Require finite row batch with one column per mode')
        for phases,starts,u in self.layers:
            out*=phases
            if len(starts):
                left=out[:,starts]; right=out[:,starts+1]
                out[:,starts]=left*u[:,0,0]+right*u[:,0,1]
                out[:,starts+1]=left*u[:,1,0]+right*u[:,1,1]
        return out


def iter_local_displacements(propagator, variance, samples, rng, *, batch_size=128):
    """Yield alpha batches without storing the entire real-normal/alpha panel.

    Use a dedicated noise RNG independent of the measurement RNG. This preserves
    the normal stream under batch changes and prevents measurement draws from
    changing subsequent displacements. No samples are drawn when samples=0.
    """
    count=_integer(samples,'samples',0,2**31-1)
    batch=_integer(batch_size,'batch_size',1,2**20)
    variance=float(variance)
    if not math.isfinite(variance) or variance<0: raise ValueError('Require finite nonnegative variance')
    scale=.5j*math.sqrt(variance)
    for first in range(0,count,batch):
        normals=rng.normal(size=(min(batch,count-first),propagator.modes))
        alpha=propagator.apply(normals)
        alpha*=scale
        yield alpha
