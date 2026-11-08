"""Circuit-local coherent Gaussian core MPS with explicit truncation ledger.

No global Fock tensor is constructed. Two-mode passive gates are evaluated in
complete fixed-total-number sectors BEFORE local output projection. The error
ledger bounds ideal-arithmetic state trace distance by a triangle inequality;
its floating values are diagnostics, not interval-certified sampler TV bounds.
"""
from __future__ import annotations
import math
from time import perf_counter
import numpy as np
from scipy.stats import nbinom
from .mps_probe import threshold_channels


def _integer(x, name, lo, hi):
    if isinstance(x, bool) or not isinstance(x, (int, np.integer)) or not lo <= x <= hi:
        raise ValueError(f'{name} must be an integer in [{lo},{hi}]')
    return int(x)


def local_circuit(modes, depth, rng):
    """Same random ensemble/stream as gaussian.local_unitary; return its gates."""
    m = _integer(modes, 'modes', 2, 1024)
    depth = _integer(depth, 'depth', 0, 128)
    u = np.eye(m, dtype=complex)
    gates = []
    for layer in range(depth):
        phases = rng.uniform(0, 2*np.pi, m)
        gates.append(('phases', phases))
        u *= np.exp(1j*phases)[:, None]
        for i in range(layer % 2, m-1, 2):
            theta, phi = rng.uniform(0, np.pi/2), rng.uniform(0, 2*np.pi)
            c, s, z = math.cos(theta), math.sin(theta), np.exp(1j*phi)
            pair = np.array([[c, -z*s], [z.conjugate()*s, c]])
            gates.append(('pair', i, pair))
            u[[i, i+1]] = pair @ u[[i, i+1]]
    return gates, u


def squeezed_product(modes, squeezing, dimension):
    m = _integer(modes, 'modes', 2, 1024)
    d = _integer(dimension, 'dimension', 1, 24)
    s = float(squeezing)
    if not math.isfinite(s) or not 0 <= s <= 5:
        raise ValueError('Require core squeezing in [0,5]')
    c = np.zeros(d, complex)
    c[0] = 1/math.sqrt(math.cosh(s))
    for n in range(2, d, 2):
        c[n] = -math.tanh(s)*math.sqrt((n-1)/n)*c[n-2]
    single_tail = float(nbinom.sf((d-1)//2, .5, 1/math.cosh(s)**2))
    joint_tail = -math.expm1(m*math.log1p(-single_tail))
    c /= np.linalg.norm(c)
    return [c.reshape(1,d,1).copy() for _ in range(m)], {
        'single_mode_input_tail':single_tail, 'joint_input_tail':joint_tail,
        'input_trace_budget':math.sqrt(joint_tail)}


def passive_gate_blocks(unitary, dimension):
    """Exact passive two-mode amplitudes, split into kept/leaking output rows.

    Columns are occupations (n1,n2) with each <d. Output rows cover all
    occupations of total <=2(d-1), including those outside the local box.
    This is NOT exp of a locally truncated quadratic generator.
    """
    d = _integer(dimension, 'dimension', 1, 24)
    u = np.asarray(unitary, dtype=complex)
    if u.shape != (2,2) or not np.all(np.isfinite(u)) or not np.allclose(u.conj().T@u,np.eye(2),atol=1e-12,rtol=0):
        raise ValueError('Require a unitary two-mode gate')
    outside = [(a,b) for a in range(2*d-1) for b in range(2*d-1)
               if a+b <= 2*(d-1) and (a>=d or b>=d)]
    lookup = {n:i for i,n in enumerate(outside)}
    kept = np.zeros((d*d,d*d),complex)
    leaked = np.zeros((len(outside),d*d),complex)
    fac = [math.factorial(n) for n in range(2*d)]
    for n1 in range(d):
        for n2 in range(d):
            total = n1+n2
            for k in range(total+1):
                amplitude = 0j
                for a in range(max(0,k-n2),min(n1,k)+1):
                    b = k-a
                    amplitude += (math.comb(n1,a)*math.comb(n2,b)
                        *u[0,0]**a*u[1,0]**(n1-a)*u[0,1]**b*u[1,1]**(n2-b))
                amplitude *= math.sqrt(fac[k]*fac[total-k]/(fac[n1]*fac[n2]))
                if k<d and total-k<d:
                    kept[k*d+total-k,n1*d+n2] = amplitude
                else:
                    leaked[lookup[k,total-k],n1*d+n2] = amplitude
    residual = float(np.max(np.abs(kept.conj().T@kept+leaked.conj().T@leaked-np.eye(d*d))))
    if residual > 1e-10:
        raise ArithmeticError(f'Two-mode gate isometry residual {residual}')
    return kept, leaked, residual


class LocalMPS:
    """Mixed-canonical MPS with explicit orthogonality center and loss records."""
    def __init__(self, tensors, max_bond, *, max_pair_elements=4_000_000):
        self.tensors = [np.asarray(a,dtype=complex).copy() for a in tensors]
        if len(tensors)<2 or any(a.ndim!=3 or not np.all(np.isfinite(a)) for a in self.tensors):
            raise ValueError('Invalid MPS tensors')
        self.dimension = self.tensors[0].shape[1]
        if (self.tensors[0].shape[0]!=1 or self.tensors[-1].shape[2]!=1
                or any(a.shape[1]!=self.dimension for a in self.tensors)
                or any(a.shape[2]!=b.shape[0] for a,b in zip(self.tensors,self.tensors[1:]))):
            raise ValueError('Incompatible MPS dimensions')
        self.max_bond = _integer(max_bond,'max_bond',1,512)
        self.max_pair_elements = _integer(max_pair_elements,'max_pair_elements',1,100_000_000)
        # Require a normalized state already right-canonical at site zero.
        for a in self.tensors[1:]:
            b=a.reshape(a.shape[0],-1)
            if not np.allclose(b@b.conj().T,np.eye(len(b)),atol=1e-10,rtol=0):
                raise ValueError('Initial MPS must be right-canonical')
        if abs(np.linalg.norm(self.tensors[0])-1)>1e-10:
            raise ValueError('Initial MPS must be normalized')
        self.center = 0
        self.ledger = []
        self.max_live_pair_elements = 0

    def move_center(self, target):
        target = _integer(target,'center',0,len(self.tensors)-1)
        while self.center < target:
            i=self.center; a=self.tensors[i]
            q,r=np.linalg.qr(a.reshape(-1,a.shape[2]),mode='reduced')
            self.tensors[i]=q.reshape(a.shape[0],a.shape[1],q.shape[1])
            self.tensors[i+1]=np.tensordot(r,self.tensors[i+1],axes=(1,0))
            self.center+=1
        while self.center > target:
            i=self.center; a=self.tensors[i]
            q,r=np.linalg.qr(a.reshape(a.shape[0],-1).T,mode='reduced')
            self.tensors[i]=q.T.reshape(q.shape[1],a.shape[1],a.shape[2])
            self.tensors[i-1]=np.tensordot(self.tensors[i-1],r.T,axes=(2,0))
            self.center-=1

    def phases(self, values):
        phases=np.asarray(values,dtype=float)
        if phases.shape!=(len(self.tensors),) or not np.all(np.isfinite(phases)):
            raise ValueError('Invalid phase layer')
        for i,phase in enumerate(phases):
            self.tensors[i]*=np.exp(1j*phase*np.arange(self.dimension))[None,:,None]

    def pair(self, first, unitary):
        i=_integer(first,'first',0,len(self.tensors)-2)
        self.move_center(i)
        a,b=self.tensors[i:i+2]; d=self.dimension
        # Count all kept/leaking branch arrays before allocation (not RSS/SVD workspace).
        elements=a.shape[0]*b.shape[2]*(d*d+d*(d-1))
        if elements>self.max_pair_elements:
            raise MemoryError(f'Pair branch cap: {elements}>{self.max_pair_elements}')
        kept,leaked,residual=passive_gate_blocks(unitary,d)
        theta=np.tensordot(a,b,axes=(2,0)).transpose(1,2,0,3).reshape(d*d,-1)
        out=kept@theta
        leak=leaked@theta
        pre=float(np.sum(np.abs(theta)**2))
        inside=float(np.sum(np.abs(out)**2)); outside=float(np.sum(np.abs(leak)**2))
        if abs(pre-1)>1e-9 or abs(inside+outside-pre)>1e-9:
            raise ArithmeticError('Gate norm/canonical check failed')
        if inside<=1e-28:
            raise ArithmeticError('Local projection retains no reliable state mass')
        local_tail=outside/(inside+outside)
        projected=(out/math.sqrt(inside)).reshape(d,d,a.shape[0],b.shape[2]).transpose(2,0,1,3)
        u,s,vh=np.linalg.svd(projected.reshape(a.shape[0]*d,d*b.shape[2]),full_matrices=False)
        rank=min(self.max_bond,len(s))
        retained=float(np.sum(s[:rank]**2)); dropped=float(np.sum(s[rank:]**2))
        bond_tail=dropped/(retained+dropped)
        step_tail=local_tail+(1-local_tail)*bond_tail
        self.tensors[i]=u[:,:rank].reshape(a.shape[0],d,rank)
        self.tensors[i+1]=((s[:rank]/math.sqrt(retained))[:,None]*vh[:rank]).reshape(rank,d,b.shape[2])
        self.center=i+1
        self.max_live_pair_elements=max(self.max_live_pair_elements,elements)
        record={'first':i,'local_projection_tail':local_tail,'bond_tail':bond_tail,
            'combined_step_tail':step_tail,'step_trace_budget':math.sqrt(step_tail),
            'rank':rank,'gate_isometry_residual':residual,
            'norm_residual':abs(inside+outside-pre)}
        self.ledger.append(record)
        return record


def build_local_core(modes, squeezing, dimension, max_bond, gates, *, max_pair_elements=4_000_000):
    start=perf_counter()
    tensors,input_info=squeezed_product(modes,squeezing,dimension)
    mps=LocalMPS(tensors,max_bond,max_pair_elements=max_pair_elements)
    for gate in gates:
        if gate[0]=='phases':
            mps.phases(gate[1])
        elif gate[0]=='pair':
            mps.pair(gate[1],gate[2])
        else:
            raise ValueError('Unknown gate kind')
    mps.move_center(0)
    gate_budget=sum(r['step_trace_budget'] for r in mps.ledger)
    return mps.tensors, dict(input_info, gates=mps.ledger,
        gate_trace_budget_sum=gate_budget,
        construction_trace_budget=min(1.,input_info['input_trace_budget']+gate_budget),
        unclipped_construction_trace_budget=input_info['input_trace_budget']+gate_budget,
        max_bond_actual=max(a.shape[2] for a in mps.tensors),
        stored_complex_elements=sum(a.size for a in mps.tensors),
        max_pair_branch_elements=mps.max_live_pair_elements,
        construction_seconds=perf_counter()-start,
        scope='Exact-arithmetic truncation triangle bound evaluated in float64; excludes numerical state and sampling errors')


class LocalSamplingError(ArithmeticError):
    """Failure retains completed samples and the failed binary prefix."""
    def __init__(self, message, completed_samples, sample_index, mode, prefix):
        super().__init__(message)
        self.completed_samples=np.asarray(completed_samples,dtype=np.uint8).copy()
        self.sample_index=sample_index
        self.mode=mode
        self.prefix=list(prefix)


def sample_local_threshold(tensors, displacements, rng):
    """Right-canonical rank-one sampling with explicit partial-failure records."""
    alpha=np.asarray(displacements,dtype=complex)
    if alpha.ndim!=2 or alpha.shape[1]!=len(tensors) or not np.all(np.isfinite(alpha)):
        raise ValueError('Require finite sample by mode displacements')
    for a in tensors[1:]:
        flat=a.reshape(a.shape[0],-1)
        if not np.allclose(flat@flat.conj().T,np.eye(len(flat)),atol=1e-10,rtol=0):
            raise ValueError('Require a right-canonical MPS')
    out=np.zeros(alpha.shape,dtype=np.uint8)
    max_residual=0.
    for i,row in enumerate(alpha):
        left=np.ones(1,complex)
        for j,(a,z) in enumerate(zip(tensors,row)):
            try:
                local=np.tensordot(left,a,axes=(0,0))
                branches=threshold_channels(z,a.shape[1])@local
                weights=np.sum(np.abs(branches)**2,axis=1)
                total=float(weights.sum())
                if not np.all(np.isfinite(weights)) or abs(total-1)>1e-9:
                    raise ArithmeticError('Conditional norm check failed')
                max_residual=max(max_residual,abs(total-1))
                channel=int(rng.choice(len(weights),p=weights/total))
                if weights[channel]<=0:
                    raise ArithmeticError('Selected zero branch')
                out[i,j]=channel!=0
                left=branches[channel]/math.sqrt(weights[channel])
            except (ArithmeticError,ValueError) as e:
                raise LocalSamplingError(str(e),out[:i],i,j,out[i,:j]) from e
    return out, {'max_observed_conditional_norm_residual':max_residual,
        'completed_samples':len(out),'scope':'Observed paths only; no global finite-precision failure or TV certificate'}
