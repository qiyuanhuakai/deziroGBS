import itertools
import math
import unittest
import mpmath as mp
import numpy as np
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.exact import ThresholdModel
from gbslab.meixner_high_order import DiagonalNormalWick,occupations,threshold_transform,projection_prefixes
from gbslab.meixner_probe import uniform_parameters
from gbslab.meixner_vacuum import determinant_series,vacuum_projection_prefixes,vacuum_to_click


class VacuumMeixnerTests(unittest.TestCase):
    def test_original_reference_regression(self):
        u=haar_unitary(4,np.random.default_rng(1801))
        for r,eta in ((.6,.35),(.9,.65),(1.2,.85)):
            b=uniform_parameters(4,r,eta)['b']; old=projection_prefixes(u,r,eta,8)
            new=vacuum_projection_prefixes(u,r,eta,8,reference_mean=b)
            for d in range(9): np.testing.assert_allclose(new['raw'][d],old['laws'][d]['raw'],atol=2e-14,rtol=0)

    def test_arbitrary_reference_against_explicit_wick(self):
        u=haar_unitary(3,np.random.default_rng(1802)); r=.7; eta=.45; n=eta*math.sinh(r)**2
        M=-eta*math.sinh(r)*math.cosh(r)*(u@u.T)
        for b in (n,.5*n,2*n):
            result=vacuum_projection_prefixes(u,r,eta,6,reference_mean=b); engine=DiagonalNormalWick(n-b,M)
            A=np.zeros(8); bits=(np.arange(8)[:,None]>>np.arange(3))&1
            base=np.prod(np.where(bits,b/(1+b),1/(1+b)),axis=1)
            for d in range(7):
                for k in occupations(3,d):
                    mask=sum(1<<i for i,x in enumerate(k) if x)
                    A[mask]+=engine.binomial_moment(k).real*(-b)**(d-mask.bit_count())/(b*(1+b))**d
                np.testing.assert_allclose(result['raw'][d],base*threshold_transform(A,b),atol=3e-14,rtol=0)

    def test_independent_mp_determinant_series(self):
        s=np.array([.1,.25]); n=.3; b=.4
        with mp.workdps(70):
            def g(x): return mp.fprod((1+x*(n-b+float(a))/(1+b))*(1+x*(n-b-float(a))/(1+b)) for a in s)**(-mp.mpf('.5'))
            coeff=determinant_series(s,n,b,10)
            for k in range(11): self.assertLess(abs(coeff[k]-float(mp.diff(g,0,k)/math.factorial(k))),1e-15)

    def test_inclusion_exclusion_independent(self):
        table=np.random.default_rng(1803).normal(size=(3,16)); expected=np.zeros_like(table)
        for bit in range(16):
            zero=15^bit; sub=bit
            while True:
                expected[:,bit]+=(-1)**sub.bit_count()*table[:,zero|sub]
                if sub==0: break
                sub=(sub-1)&bit
        np.testing.assert_allclose(vacuum_to_click(table),expected,atol=2e-15,rtol=0)

    def test_mean_reference_even_positive_and_tail(self):
        u=haar_unitary(4,np.random.default_rng(1804)); result=vacuum_projection_prefixes(u,.9,.65,64)
        self.assertTrue(np.all(result['vacuum_coefficients']>=0))
        self.assertTrue(np.all(result['vacuum_coefficients'][1::2]==0))
        target=ThresholdModel(covariance_from_interferometer(u,.9,.65)).full_distribution()
        for d,row in enumerate(result['rows']):
            signed=.5*np.abs(result['raw'][d]-target).sum(); repaired=.5*np.abs(result['repaired'][d]-target).sum()
            self.assertLessEqual(signed,row['signed_half_l1_bound_estimate']+1e-13)
            self.assertLessEqual(repaired,row['repaired_tv_bound_estimate']+1e-13)
        self.assertLess(.5*np.abs(result['repaired'][-1]-target).sum(),1e-12)

    def test_spectral_radius_minimized_at_physical_mean(self):
        for r,eta in ((.6,.35),(.9,.65),(1.2,.85)):
            n=eta*math.sinh(r)**2; a=eta*math.sinh(r)*math.cosh(r); optimum=a/(1+n)
            self.assertLess(optimum,1)
            for b in np.linspace(0,4*(1+n),101): self.assertGreaterEqual(max(abs(n-b+a),abs(n-b-a))/(1+b)+1e-15,optimum)

    def test_phase_and_permutation(self):
        u=haar_unitary(4,np.random.default_rng(1805)); perm=[2,0,3,1]
        a=vacuum_projection_prefixes(u,1.2,.85,32); b=vacuum_projection_prefixes(u[perm]*np.exp(1j*np.arange(4))[:,None],1.2,.85,32)
        ids=[sum(((x>>j)&1)<<perm[j] for j in range(4)) for x in range(16)]
        np.testing.assert_allclose(b['raw'],a['raw'][:,ids],atol=3e-14,rtol=0)

    def test_zero_squeezing_and_caps(self):
        result=vacuum_projection_prefixes(np.eye(4),0,.35,8)
        np.testing.assert_array_equal(result['raw'][:,0],np.ones(9))
        np.testing.assert_array_equal(result['raw'][:,1:],np.zeros((9,15)))
        for b in (-1,float('nan')):
            with self.assertRaises(ValueError): vacuum_projection_prefixes(np.eye(4),.6,.35,8,reference_mean=b)
        with self.assertRaises(ValueError): vacuum_projection_prefixes(np.eye(4),.6,.35,129)
        with self.assertRaises(ValueError): vacuum_projection_prefixes(np.eye(4),.6,.35,8,max_modes=3)

if __name__=='__main__': unittest.main()
