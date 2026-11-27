"""Total-photon generating function conditioned on arbitrary physical detector vacuum subsets."""
import argparse
from pathlib import Path
from time import perf_counter
import numpy as np
from threadpoolctl import threadpool_limits
from iteration100_spatial_v1 import save,manifest,realify
from iteration106_field_v1 import DATA,actual_input
from iteration107_cliques_v1 import optical_x
from iteration107_fast_v1 import oracle,gaussian_click,ONE
from iteration108_laws_v1 import modes
from iteration109_model_v1 import build,panel
from iteration110_spectrum_v1 import Spectrum

ANCHOR=np.array([272,273,274,275,280,281,282,283])
DIRECT=np.arange(272,288)


def optical_parts(t,r,ports):
    u,x,_=modes(t,r)
    head=realify(u[ports])
    return [(e,head@v) for e,v in map(np.linalg.eigh,x)]


def model_parts(model,ports):
    g=len(model['weights']);fan=model['fan']
    group=ports//fan;scale=np.sqrt(model['weights'][group,ports%fan])
    compressed=dict(model,fan=1,weights=np.ones((g,1)))
    x=panel(compressed,np.arange(g))
    ids=np.r_[group,group+g];scale=np.tile(scale,2)
    return [(e,v[ids]*scale[:,None]) for e,v in map(np.linalg.eigh,x)]


def joint(parts,bits,size):
    # det(I+(1-z)X+z P_A X) splits into the unmarked total determinant
    # and a determinant on the 2|A| measured quadratures, even for dense X.
    z=np.exp(-2j*np.pi*np.arange(size//2+1)/size)
    masks=np.arange(1<<bits);logg=np.zeros((len(masks),len(z)),complex)
    all_e=[]
    for e,v in parts:
        all_e.extend(e);den=1+(1-z[:,None])*e
        logg-=np.log(den).sum(1)/2
        tilt=z[:,None,None]*np.einsum('ij,zj,kj->zik',v,e/den,v,optimize=True)
        tilt+=np.eye(2*bits)
        for mask in masks[1:]:
            pos=np.flatnonzero((mask>>np.arange(bits))&1)
            ix=np.r_[pos,pos+bits]
            sign,logdet=np.linalg.slogdet(tilt[:,ix][:,:,ix])
            phase=np.unwrap(np.angle(sign))
            if abs(phase[0])>1e-9 or np.max(abs(np.diff(phase)))>3.:raise ArithmeticError('Determinant phase needs a finer grid')
            logg[mask]-=(logdet+1j*phase)/2
    vacuum=np.fft.irfft(np.exp(logg),n=size,axis=1)
    word=vacuum[::-1].astype(np.longdouble)
    for j in range(bits):
        half=1<<j;rows=word.reshape(-1,2*half,size)
        rows[:,half:]-=rows[:,:half]
    law=np.asarray(word,float).T.copy()
    # Frequencies store N modulo size. Remove impossible low-count words;
    # their true wrapped mass is bounded by the all-port Chernoff tail.
    impossible=np.arange(size)[:,None]<np.array([int(w).bit_count() for w in masks])
    bad=float(np.abs(law[impossible]).sum());neg=float(-np.minimum(law,0).sum());law[impossible]=0
    law=law.clip(0);law/=law.sum()
    e=np.asarray(all_e);s=1+.6/max(1.,e.max());tail=float(np.exp(-.5*np.log1p((1-s)*e).sum()-size*np.log(s)))
    return law,dict(negative_mass=neg,impossible_mass=bad,tail_bound=tail)


def mapping(source,target):
    source=source.ravel().clip(0);source=source/source.sum()
    target=target.ravel().clip(0);target=target/target.sum()
    excess=(source-target).clip(0);missing=(target-source).clip(0);rho=missing.sum()
    move=np.divide(excess,source,out=np.zeros_like(excess),where=source>0)
    one=2**52
    thresh=np.rint(move*one).astype(np.uint64)
    if rho>0:cdf=np.rint(np.cumsum(missing/rho)*one).astype(np.uint64)
    else:cdf=np.full(len(source),one,np.uint64)
    cdf=np.minimum(cdf,np.uint64(one));cdf[-1]=one
    probability=thresh.astype(float)/one
    after=source*(1-probability)+(source@probability)*np.diff(cdf,prepend=np.uint64(0))/one
    return thresh,cdf,dict(edit_probability=float(source@probability),before_tv=float(abs(source-target).sum()/2),
                           after_tv=float(abs(after-target).sum()/2))


def run(out):
    out.mkdir(parents=True,exist_ok=False);t,r=actual_input();m=build(t,r,8,1);s=Spectrum(m)
    scan=np.load(DATA/'iteration110_spectrum_v1/laws.npz');fitted=s.materialize(scan['coeff'])
    ref=oracle(True);target16=np.load(DATA/'iteration106_packed_v1/panel_5.npz')['target']
    orig=optical_parts(t,r,ANCHOR);st=perf_counter();p,pc=joint(orig,len(ANCHOR),1024)
    # Two Fourier grids and an independent binary128 threshold calculation.
    p2,_=joint(orig,len(ANCHOR),512)
    alias_gap=float(abs(p2-p[:512]-p[512:]).sum()/2)
    exact=gaussian_click(ref,optical_x(t[ANCHOR],r),ONE)
    check=float(np.abs(exact-p.sum(0)).sum()/2)
    assert max(check,alias_gap)<1e-9 and pc['tail_bound']<1e-12
    arrays=dict(target=p,target16=target16,anchor=ANCHOR,direct=DIRECT,coeff=scan['coeff']);rows=[]
    for name,model in [('base',m),('fitted',fitted)]:
        q,qc=joint(model_parts(model,ANCHOR),len(ANCHOR),1024)
        expected=scan['before' if name=='base' else 'after'];fold=expected.reshape(-1,1024).sum(0)
        assert abs(fold-q.sum(1)).sum()/2<1e-9
        q16=gaussian_click(oracle(),panel(model,DIRECT),ONE)
        wt,ct,pnr=mapping(q,p);wb,cb,direct=mapping(q16,target16)
        arrays.update({name+'_q':q,name+'_q16':q16,name+'_pnr_move':wt,name+'_pnr_cdf':ct,name+'_click_move':wb,name+'_click_cdf':cb})
        rows.append(dict(name=name,pnr=pnr,direct=direct,source_checks=qc,total_tv=float(abs(q.sum(1)-p.sum(1)).sum()/2)))
    np.savez_compressed(out/'tables.npz',**arrays)
    save(out,'summary.json',dict(rows=rows,seconds=perf_counter()-st,reference_checks=pc,threshold_check_tv=check,grid_check_tv=alias_gap))
    manifest(out);print(rows,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path)
    with threadpool_limits(limits=2,user_api='blas'):run(ap.parse_args().output.resolve())
