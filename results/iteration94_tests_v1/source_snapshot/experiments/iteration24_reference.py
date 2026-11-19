"""Independent global sparse elimination and low-photon MPS diagnostics."""
import math
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu


def permutation_parity(p):
    visited=set(); cycles=0
    for i in range(len(p)):
        if i in visited: continue
        cycles+=1; j=i
        while j not in visited: visited.add(j); j=int(p[j])
    return (len(p)-cycles)%2


def global_elimination(kernels,bonds,*,max_virtual=20000):
    """All paired virtual copies at once: B=P+R.T (C-V)^(-1) R.

    Algebraic crosscheck, not an independent convergence proof. The global
    determinant only fixes the scalar squared, not its square-root sign.
    """
    m=len(kernels); count=2*sum(bonds[1:-1])
    if count>max_virtual: raise MemoryError('Global virtual dimension cap')
    maps=[]; position=0
    for i in range(m):
        l,r=bonds[i:i+2]; left=np.arange(position,position+l); position+=l
        right=np.arange(position,position+r); position+=r; maps.append((left,right))
    assert position==count
    rows=[]; cols=[]; values=[]; R=np.zeros((count,m),complex); P=np.zeros(m,complex)
    for i,K in enumerate(kernels):
        l,r=bonds[i:i+2]; ids=np.r_[maps[i][0],maps[i][1]]; local=np.r_[np.arange(l),np.arange(l+1,l+1+r)]
        block=K[np.ix_(local,local)]; rr,cc=np.nonzero(block)
        rows.append(ids[rr]); cols.append(ids[cc]); values.append(-block[rr,cc])
        R[ids,i]=K[local,l]; P[i]=K[l,l]
        if i:
            a=maps[i-1][1]; b=maps[i][0]; assert len(a)==len(b)
            rows.extend([a,b]); cols.extend([b,a]); values.extend([np.ones(len(a)),np.ones(len(a))])
    if not count: return np.diag(P),dict(virtual_modes=0,log_determinant_abs=0.,determinant_phase=[1.,0.],solve_relative_residual=0.,symmetry_residual=0.)
    M=coo_matrix((np.concatenate(values),(np.concatenate(rows),np.concatenate(cols))),shape=(count,count)).tocsc()
    lu=splu(M); X=lu.solve(R); B=np.diag(P)+R.T@X
    residual=float(np.linalg.norm(M@X-R)/max(np.linalg.norm(R),np.finfo(float).tiny))
    asym=float(np.max(np.abs(B-B.T))); diag=lu.U.diagonal()
    parity=(permutation_parity(lu.perm_r)+permutation_parity(lu.perm_c)+sum(bonds[1:-1]))%2
    angle=math.fsum(np.angle(diag))+math.pi*parity; phase=np.exp(1j*angle)
    info=dict(virtual_modes=count,sparse_nnz=int(M.nnz),factor_nnz=int(lu.L.nnz+lu.U.nnz),solve_relative_residual=residual,symmetry_residual=asym,log_determinant_abs=math.fsum(np.log(np.abs(diag))),determinant_phase=[float(phase.real),float(phase.imag)])
    if residual>1e-9 or asym>1e-9: raise ArithmeticError(info)
    return (B+B.T)/2,info


def low_sector_mps(tensors):
    """Vacuum then all exact two-photon occupation amplitudes, no sampling."""
    m=len(tensors); suffix=[None]*(m+1); suffix[m]=np.ones(1,complex)
    for i in range(m-1,-1,-1): suffix[i]=tensors[i][:,0,:]@suffix[i+1]
    values=[complex(suffix[0][0])]; pairs=[]; prefix=np.ones(1,complex)
    for i,A in enumerate(tensors):
        values.append(complex(prefix@A[:,2,:]@suffix[i+1]) if A.shape[1]>2 else 0j); pairs.append((i,i))
        work=prefix@A[:,1,:] if A.shape[1]>1 else np.zeros(A.shape[2],complex)
        for j in range(i+1,m):
            T=tensors[j]; values.append(complex(work@T[:,1,:]@suffix[j+1]) if T.shape[1]>1 else 0j); pairs.append((i,j))
            work=work@T[:,0,:]
        prefix=prefix@A[:,0,:]
    return np.asarray(values),np.asarray(pairs,int)


def low_sector_gaussian(B,log_normalization,pairs):
    c=math.exp(log_normalization)
    return np.r_[complex(c),[c*B[i,j]/(math.sqrt(2) if i==j else 1) for i,j in pairs]]


def partial_state_bounds(left,right):
    """Normalized full states assumed; selected orthogonal amplitudes supplied."""
    p=np.abs(left)**2; q=np.abs(right)**2; massp=math.fsum(p); massq=math.fsum(q)
    if min(massp,massq)<0 or max(massp,massq)>1+1e-10: raise ArithmeticError('Invalid partial state mass')
    upper=abs(np.vdot(left,right))+math.sqrt(max(0.,1-massp)*max(0.,1-massq))
    if upper>1+1e-10: raise ArithmeticError('Partial-state overlap exceeds one')
    return dict(left_selected_mass=massp,right_selected_mass=massq,left_vacuum_probability=float(p[0]),right_vacuum_probability=float(q[0]),
        vacuum_probability_gap=float(abs(p[0]-q[0])),pnr_coarse_tv_lower_estimate=.5*(math.fsum(np.abs(p-q))+abs(massp-massq)),
        overlap_upper_estimate=min(1.,upper),pure_trace_lower_estimate=math.sqrt(max(0.,1-min(1.,upper)**2)),
        scope='Bounds in exact arithmetic for the specified pure states, evaluated in complex128. Vacuum gap also lower-bounds zero-displacement threshold TV; other PNR bounds do not automatically transfer to threshold or displaced-mixture outputs.')
