"""Ball-arithmetic check of all 81 measured multiplicity cells for each response."""
import json
import math
import numpy as np
from flint import arb,acb,arb_mat,acb_mat,ctx
from iteration120_interval import INPUT,OUT,exact


def all_void(t,s):
    n,m=t.nrows(),t.ncols()
    r=arb_mat(2*n,2*m)
    for i in range(n):
        for j in range(m):
            x=t[i,j]
            r[2*i,j]=x.real;r[2*i,j+m]=-x.imag
            r[2*i+1,j]=x.imag;r[2*i+1,j+m]=x.real
    diag=[((sign*2*exact(z)).exp()-1)/2 for sign in (-1,1) for z in s]
    rd=arb_mat(2*n,2*m,[r[i,j]*diag[j] for i in range(2*n) for j in range(2*m)])
    Q=rd*r.transpose()
    for j in range(2*n):Q[j,j]+=1
    result=[None]*(1<<n)
    def visit(rows,offset,word,d):
        if not rows:
            result[word]=1/d.sqrt();return
        aa,ab=rows[0][:2];bb=rows[1][1]
        k=aa*bb-ab*ab;assert k>0
        rest=[x[2:] for x in rows[2:]]
        visit(rest,offset+1,word,d)
        x,y=rows[0][2:],rows[1][2:]
        scaled=[((bb*xi-ab*yi)/k,(aa*yi-ab*xi)/k) for xi,yi in zip(x,y)]
        schur=[[value-p*xx-q*yy for value,xx,yy in zip(row,x,y)] for row,(p,q) in zip(rest,scaled)]
        visit(schur,offset+1,word|(1<<offset),d*k)
    visit(Q.tolist(),0,0,arb(1))
    return result


def law(t,ss):
    v,u=[all_void(t,s) for s in ss]
    cells=[x*y for x,y in zip(reversed(v),reversed(u))]
    for i in range(16):
        step=1<<i
        for start in range(0,65536,2*step):
            for j in range(start,start+step):cells[j+step]-=cells[j]
    out=arb_mat(9,9)
    for word,x in enumerate(cells):
        out[(word&255).bit_count(),(word>>8).bit_count()]+=x
    return out


def run():
    ctx.prec=160;ctx.threads=2
    with np.load(INPUT) as f:t0=f['T'][np.argsort(f['order'])];ss=f['squeezing'].T.repeat(2,axis=1)
    with np.load(OUT/'response_predictions.npz') as f:pred={k:f[k] for k in f}
    errs=[];scale=1/arb(2).sqrt()
    for j,banks in enumerate([(34,35),(71,72),(102,143)]):
        sites=(8*np.array(banks)[:,None]+np.arange(8)).ravel();raw=t0[sites]
        t=acb_mat(16,50)
        for i in range(16):
            for k in range(0,50,2):
                x,y=[acb(exact(z.real),exact(z.imag)) for z in raw[i,k:k+2]]
                t[i,k]=acb(0,scale)*(x+y);t[i,k+1]=scale*(x-y)
        adjusted=acb_mat(16,50,[t[i,k]*exact(pred['gains'][banks[i//8]]).sqrt() for i in range(16) for k in range(50)])
        scores={'published':law(t,ss),'adjusted':law(adjusted,ss)}
        base=scores['published']
        for name in ('poisson','refill'):
            kernels=[]
            for side,bank in enumerate(banks):
                K=arb_mat(9,9)
                marg=[sum(base[i,k] if side==0 else base[k,i] for k in range(9)) for i in range(9)]
                if name=='refill':
                    a=exact(pred['empty_survival'][bank]);K[0,0]=a
                    for k in range(1,9):K[k,k]=1;K[0,k]=(1-a)*marg[k]/(1-marg[0])
                else:
                    z=(-exact(pred['poisson_mean_per_port'][bank])).exp()
                    for k in range(9):
                        for h in range(k,9):K[k,h]=math.comb(8-k,h-k)*(1-z)**(h-k)*z**(8-h)
                kernels.append(K)
            scores[name]=kernels[0].transpose()*base*kernels[1]
        row={}
        for name,a in scores.items():
            delta=sum(abs(x-exact(y)) for x,y in zip(a.entries(),pred[name+'_pairs'][j]))/2
            assert delta<arb('1e-9') and (sum(a.entries())-1).contains(0)
            row[name]=str(delta)
        errs.append(row);print(banks,row,flush=True)
    (OUT/'panel_law_checks.json').write_text(json.dumps(dict(precision_bits=ctx.prec,tv_rounding_bound='1e-9',actual=errs),indent=2)+'\n')


if __name__=='__main__':run()
