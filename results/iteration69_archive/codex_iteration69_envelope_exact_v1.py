from pathlib import Path
import json,hashlib,itertools
import sympy as s
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.git/iteration69_envelope_exact_v1_20261110';OUT.mkdir(exist_ok=False)
R=s.Rational;I=s.I;checks=0;rows=[]
def check(condition):
    global checks
    assert condition;checks+=1
def real(x):return s.simplify(s.expand_complex(x))
def sq(x):return real(x*s.conjugate(x))
for n in (1,2,3):
    unit=s.eye(4)
    for a,b in [(0,1),(1,2),(2,3),(0,2)]:
        rotation=s.eye(4);rotation[a,a]=rotation[b,b]=R(3,5);rotation[a,b]=R(4,5);rotation[b,a]=-R(4,5);unit=unit*rotation
    unit=s.diag(1,I,R(3,5)+R(4,5)*I,-I)*unit
    for sparse in (False,True):
        v=s.eye(4)[:,:n] if sparse else unit[:,:n];check(s.simplify(v.H*v)==s.eye(n));ell=[real((v.row(i)*v.row(i).H)[0]) for i in range(4)]
        for rank in (1,n):
            b=s.Matrix(n,rank,lambda j,k:R(j+k+1,5)+R(j-k,7)*I);h=s.simplify(b*b.H)
            diagonal=[sum(abs(s.re(h[j,k]))+abs(s.im(h[j,k])) for k in range(n)) for j in range(n)];d=s.diag(*diagonal)
            for size in range(1,n+1):
                for ids in itertools.combinations(range(n),size):check(real((d-h).extract(ids,ids).det())>=0)
            reference=s.Matrix([R(j+1,4)+R(j%2,3)*I for j in range(n)])
            for drift in (R(0),R(1,9),R(2,3)):
                z=reference*(R(3,7)+R(2,5)*I)+s.Matrix([drift*R((-1)**j,j+2)*(1+I) for j in range(n)])
                alpha=real((reference.H*z)[0]/(reference.H*reference)[0]);delta=s.simplify(z-alpha*reference);delta_norm=real((delta.H*delta)[0]);reference_weights=[sq((v.row(i)*reference)[0]) for i in range(4)]
                covrates=[real((v.row(i)*h*v.row(i).H)[0]) for i in range(4)];rates=[covrates[i]+sq((v.row(i)*z)[0]) for i in range(4)];energy=real(sum(rates));check(energy==real(s.trace(h)+(z.H*z)[0]))
                for t in (R(1,2),R(1),R(2)):
                    a=(1+t)*sq(alpha);b=(1+1/t)*delta_norm;ceilings=[real(sum(diagonal[j]*sq(v[i,j]) for j in range(n))+a*reference_weights[i]+b*ell[i]) for i in range(4)];normalizer=sum(ceilings);check(normalizer>0)
                    for rate,ceiling in zip(rates,ceilings):
                        check(0<=rate<=ceiling)
                        if ceiling==0:check(rate==0)
                        else:check(real((ceiling/normalizer)*(rate/ceiling)/(energy/normalizer))==real(rate/energy))
                    rows.append(dict(inputs=n,sparse=sparse,rank=rank,drift=str(drift),t=str(t),expected_proposals=str(real(normalizer/energy))))
result=dict(passed=True,checks=checks,cases=len(rows),rows=rows,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),general_argument='For Hermitian H>=0, d_j=sum_k(|Re H_jk|+|Im H_jk|) implies D-H is Hermitian diagonally dominant with nonnegative diagonal and hence PSD. Write z=alpha*z_ref+delta. For every detector row v_i, |v_i z|^2 <= (1+t)|alpha|^2|v_i z_ref|^2+(1+1/t)||delta||^2||v_i||^2 for t>0. Adding v_i D v_i* bounds the total rate. Sampling a mixture of the fixed column, coherent-reference and leverage distributions gives q_i=ceiling_i/Z; accepting rate_i/ceiling_i gives accepted law rate_i/sum(rate). Zero ceiling implies zero rate. This is an exact-arithmetic proof; finite rational fixtures verify its implementation, not a floating-point program certificate. The selector preserves the approximate displaced reset model, not the exact GBS target.')
(OUT/'summary.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ['rows','general_argument']},indent=2))
