"""Width- and squeezing-uniform pure-core bounds, with exact rational tail arithmetic."""
import argparse
import json
from fractions import Fraction
from decimal import Decimal,localcontext

from iteration122_powers import OUT,POWERS,json_write


def pair_retained(q,rank,k):
    # For the displayed even rank, the two-spectral retention calculation
    # uses exactly rational numbers, including the zero-pair term.
    assert rank%2==0 and 0<=q<1
    mass=(1-q)**(rank//2)
    total=mass
    for j in range(1,k+1):
        mass=mass*q*(rank//2+j-1)/j
        total+=mass
    return total


def upper_decimal(squared,places=8):
    scale=10**places
    # Integer ceiling of 10^places * sqrt(squared).
    from math import isqrt
    n=isqrt((squared.numerator*scale**2)//squared.denominator)
    if Fraction(n*n,scale**2)<squared:
        n+=1
    assert Fraction(n*n,scale**2)>=squared
    return f'{n//scale}.{n%scale:0{places}d}'


def uniform():
    results=[]
    for eta,k in [(Fraction(7,100),2),(Fraction(1,5),5),(Fraction(3,10),7),
                  (Fraction(2,5),10),(Fraction(51,100),15)]:
        q=(eta/(2-eta))**2
        kept=pair_retained(q,50,k)
        d=1-kept**2
        results.append(dict(max_intensity_transmission=str(eta),sources_per_spectrum=50,spectra=2,
                            pair_cap_per_spectrum=k,full_tv_upper=upper_decimal(d),
                            exact_squared_upper=str(d)))
    return results


def positive_hermitian(a):
    # Positive Schur pivots prove positive definiteness of the true matrix.
    A=a.tolist()
    lower=[]
    while A:
        pivot=A[0][0]
        assert pivot.imag.contains(0) and pivot.real>0
        lower.append(float(pivot.real.lower()))
        col=[row[0] for row in A[1:]]
        weights=[z/pivot.real for z in col]
        A=[[value-w*z.conjugate() for value,z in zip(row[1:],col)] for row,w in zip(A[1:],weights)]
    return min(lower)


def grams():
    from flint import acb,arb,ctx
    from iteration120_interval import optical_data,diagonal
    ctx.prec=180;ctx.threads=2
    info={}
    for power in POWERS:
        _,_,_,g,*_=optical_data(OUT/power/'input.npz')
        eye=diagonal([1]*g.nrows())
        info[power]=[positive_hermitian(g-eye*acb(arb(3)/10)),
                     positive_hermitian(eye*acb(arb(51)/100)-g)]
    return info


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--write',action='store_true')
    args=p.parse_args()
    table=dict(isotropic=uniform(),gram_limits=['3/10','51/100'],strict_pivot_minima=grams())
    dst=OUT/'families.json'
    if args.write:json_write(dst,table)
    else:
        archived=json.loads(dst.read_bytes())
        assert archived['isotropic']==table['isotropic'] and min(min(x) for x in table['strict_pivot_minima'].values())>0
    print([(r['max_intensity_transmission'],r['pair_cap_per_spectrum'],r['full_tv_upper']) for r in table['isotropic']],flush=True)
