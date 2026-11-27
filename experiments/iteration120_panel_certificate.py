"""Bound full measured TV on the registered sixteen-port panels after clipping."""
import json
import numpy as np
from flint import arb,acb,acb_mat,ctx
from iteration120_interval import OUT,exact,diagonal,optical_data


def overlap(g,f,ga,cut,z):
    I=diagonal([1]*g.nrows());adj=f.conjugate().transpose()
    h=adj*g*f*g.conjugate()
    p0=(I-h).det().real.sqrt()
    # The contraction per photon is D(1/z^2,z)=(G+(z-1)*G_A)/z in source coordinates.
    d=(g+(z-1)*ga)*acb(1/arb(z))
    tilted=adj*d*f*d.conjugate()
    det=(I-tilted).det()
    assert det.imag.contains(0) and det.real>0
    all_orders=p0/det.real.sqrt()
    terms=[p0];traces=[];power=tilted
    for n in range(1,cut+1):
       tr=power.trace();assert tr.imag.contains(0)
       traces.append(tr.real);terms.append(sum(traces[k]*terms[n-1-k] for k in range(n))/(2*n))
       if n<cut:power=power*tilted
    return (all_orders-sum(terms))*arb(z)**(2*cut)


def sectors(g,f,ga,k):
    """Positive-coefficient Cauchy extraction of the discarded photon count."""
    I=diagonal([1]*g.nrows());adj=f.conjugate().transpose()
    h=adj*g*f*g.conjugate();p0=(I-h).det().real.sqrt()
    size=40;radius=arb(1)/5;assert p0-radius*(1-p0)>0
    outside=g-ga
    a=adj*ga*f;b=adj*outside*f
    h0=a*ga.conjugate();h1=a*outside.conjugate()+b*ga.conjugate();h2=b*outside.conjugate()
    all_values=[];kept_values=[];grid=[]
    for j in range(1+size//2):
        angle=2*arb.pi()*j/size;w=acb(angle.cos(),angle.sin())
        z=radius*w;H=h0+z*h1+(z*z)*h2
        full=p0/(I-H).det().sqrt()
        assert full.real>0
        terms=[acb(p0)];trace=[];power=H
        for i in range(1,k+1):
            trace.append(power.trace())
            terms.append(sum(trace[j]*terms[i-1-j] for j in range(i))/(2*i))
            if i<k:power=power*H
        grid.append(w);all_values.append(full);kept_values.append(sum(terms))
    def extract(values,h):
        val=0
        for j,(w,s) in enumerate(zip(grid,values)):
            val+=(1 if j in (0,size//2) else 2)*(w.conjugate()**h*s).real
        return val/(size*radius**h)
    p=[extract(all_values,h)+arb(0,(radius**size).upper()) for h in range(2*k+1)]
    r=[extract(kept_values,h) for h in range(2*k+1)]
    p.append(1-sum(p));r.append(arb(0))
    return p,r


def run():
    ctx.prec=240;ctx.threads=2
    R,_,_,g,src,raw,u=optical_data();tau=1-R[-1]
    # The two-sided clipping error to the normalized 10/6 state, validated in the interval audit.
    clamp=arb('0.001460')
    records=[]
    for banks in [(34,35),(71,72),(102,143)]:
        sites=(8*np.array(banks)[:,None]+np.arange(8)).ravel();rows=raw[sites]
        t=acb_mat(rows.shape[0],rows.shape[1],[acb(exact(x.real),exact(x.imag)) for x in rows.ravel()])*u
        ga=t.conjugate().transpose()*t
        bets=[]
        for f,k in zip(src,(10,6)):
            options=[(z,overlap(g,f,ga,k,z)) for z in (2,3,4,5,6,8,10,12,16)]
            # Retain the upper endpoint, so the choice of bound needs no statistical adjustment.
            bets.append(min(options,key=lambda item:float(item[1].upper())))
        mass=sum(x.upper() for _,x in bets);assert 0<mass<tau
        # Coarse measurement on the discarded modes, with the retained ket entirely in its first branch.
        dist=((tau+mass)**2+4*R[-1]*mass).sqrt()/2+(tau-mass)/2
        cap=dist+clamp;decimal=int(float(cap.upper())*10**8)+1
        assert cap<arb(decimal)/10**8
        (ps,rs),(pt,rt)=[sectors(g,f,ga,k) for f,k in zip(src,(10,6))]
        assert (sum(rs)*sum(rt)-R[-1]).contains(0)
        fine=arb(0)
        for p,r in zip(ps,rs):
            for q,s in zip(pt,rt):
                true=p*q;unnorm=r*s;normed=unnorm/R[-1]
                rad=((true-normed)**2+4*normed*(true-unnorm)).upper()
                fine+=rad.sqrt()/2 if rad>0 else 0
        rounded=int(float((fine+clamp).upper())*10**8)+1
        assert fine+clamp<arb(rounded)/10**8 and fine<dist
        records.append(dict(banks=list(banks),radii=[z for z,x in bets],overlap_bounds=[str(x) for z,x in bets],
                            coarse_projected_tv=str(dist),resolved_projected_tv_upper=str(fine),
                            contour_nodes=40,contour_radius='1/5',full_panel_emitted_tv_upper=rounded/10**8))
    (OUT/'panel_certificate.json').write_text(json.dumps(records,indent=2)+'\n');print(records,flush=True)


if __name__=='__main__':run()
