import itertools
import unittest
import mpmath as mp
import numpy as np
from gbslab.mixed_low_clicks import mixed_gaussian_low_clicks
from gbslab.low_clicks import gaussian_low_clicks
from gbslab.gaussian import haar_unitary,covariance_from_interferometer
from gbslab.banded_gaussian_mps import pairing_covariance
from gbslab.exact import ThresholdModel


def direct_complement_reference(V,patterns,digits=90):
    """Independent high-precision inclusion-exclusion on large Q complements."""
    m=len(V)//2
    with mp.workdps(digits):
        Q=mp.matrix([[(mp.mpf(float(V[i,j]))+int(i==j))/2 for j in range(2*m)] for i in range(2*m)])
        cache={():mp.mpf(1)}
        def vacuum(modes):
            modes=tuple(sorted(modes))
            if modes not in cache:
                indices=list(modes)+[i+m for i in modes]
                cache[modes]=1/mp.sqrt(mp.det(mp.matrix([[Q[i,j] for j in indices] for i in indices])))
            return cache[modes]
        values=[]
        for pattern in patterns:
            clicks=[int(i) for i in pattern if i>=0]; outside=[i for i in range(m) if i not in clicks]
            terms=[]
            for n in range(len(clicks)+1):
                for subset in itertools.combinations(clicks,n): terms.append((-1)**n*vacuum(outside+list(subset)))
            values.append(float(mp.fsum(terms)))
        return np.array(values)


class MixedLowClickTests(unittest.TestCase):
    def test_vacuum_and_product_thermal(self):
        for backend in ('float64','mpmath'):
            r=mixed_gaussian_low_clicks(np.eye(6),backend=backend)
            np.testing.assert_array_equal(r['probabilities'],[1]+[0]*6)
            n=np.array([.01,.1,.7]); V=np.diag(np.r_[1+2*n,1+2*n]); r=mixed_gaussian_low_clicks(V,backend=backend)
            zero=1/(1+n); p0=zero.prod(); expected=[p0]+[p0*n[i] for i in range(3)]+[p0*n[i]*n[j] for i in range(3) for j in range(i+1,3)]
            np.testing.assert_allclose(r['probabilities'],expected,atol=5e-16,rtol=5e-14)

    def test_loss_and_additive_noise_against_complete_threshold_law(self):
        for m in (2,4,6):
            for seed in (271001,271002):
                U=haar_unitary(m,np.random.default_rng(seed)); V=covariance_from_interferometer(U,np.linspace(.05,.8,m),np.linspace(.1,.9,m))+.03*np.eye(2*m)
                full=ThresholdModel(V).full_distribution()
                for backend in ('float64','mpmath'):
                    r=mixed_gaussian_low_clicks(V,backend=backend); indices=[sum(1<<int(i) for i in row if i>=0) for row in r['patterns']]
                    np.testing.assert_allclose(r['probabilities'],full[indices],atol=2e-13,rtol=0)

    def test_independent_high_precision_complement_determinants(self):
        U=haar_unitary(4,np.random.default_rng(271003)); V=covariance_from_interferometer(U,[.1,.2,.4,.7],[.2,.4,.6,.8])
        r=mixed_gaussian_low_clicks(V,backend='mpmath'); reference=direct_complement_reference(V,r['patterns'])
        np.testing.assert_allclose(r['probabilities'],reference,atol=0,rtol=2e-15)

    def test_pure_generic_reduces_to_pairing_formula(self):
        U=haar_unitary(5,np.random.default_rng(271004)); B=-(U*np.tanh(np.linspace(.1,.6,5)))@U.T; V=pairing_covariance(B)
        for backend in ('float64','mpmath'):
            r=mixed_gaussian_low_clicks(V,backend=backend)
            np.testing.assert_allclose(r['probabilities'],gaussian_low_clicks(B)['probabilities'],atol=3e-14,rtol=0)

    def test_near_zero_refusal_and_explicit_precision(self):
        V=np.eye(4)*(1+2e-15)
        with self.assertRaisesRegex(ArithmeticError,'Unresolved'): mixed_gaussian_low_clicks(V)
        r=mixed_gaussian_low_clicks(V,backend='mpmath')
        reference=direct_complement_reference(V,r['patterns'],digits=120)
        np.testing.assert_allclose(r['probabilities'],reference,atol=0,rtol=2e-15)
        self.assertGreater(r['probabilities'][-1],0)
        # Exactly representable pure two-mode squeezed covariance.
        V=np.array([[1.25,.75,0,0],[.75,1.25,0,0],[0,0,1.25,-.75],[0,0,-.75,1.25]])
        with self.assertRaisesRegex(ArithmeticError,'Unresolved'): mixed_gaussian_low_clicks(V)
        r=mixed_gaussian_low_clicks(V,backend='mpmath')
        np.testing.assert_allclose(r['probabilities'],[8/9,0,0,1/9],atol=1e-16,rtol=0)

    def test_weak_cross_correlation_retained(self):
        V=1.2*np.eye(4); V[0,1]=V[1,0]=1e-20; V[2,3]=V[3,2]=-1e-20
        r=mixed_gaussian_low_clicks(V,backend='mpmath'); reference=direct_complement_reference(V,r['patterns'],digits=120)
        np.testing.assert_allclose(r['probabilities'],reference,atol=0,rtol=2e-15)

    def test_permutation_covariance_and_isolated_vacuum(self):
        V=covariance_from_interferometer(haar_unitary(3,np.random.default_rng(271005)),[.2,.3,.4],.6)
        p=[2,0,1]; ix=p+[i+3 for i in p]; a=mixed_gaussian_low_clicks(V); b=mixed_gaussian_low_clicks(V[np.ix_(ix,ix)])
        lookup={tuple(sorted(i for i in row if i>=0)):v for row,v in zip(a['patterns'],a['probabilities'])}
        for row,value in zip(b['patterns'],b['probabilities']): self.assertAlmostEqual(value,lookup[tuple(sorted(p[i] for i in row if i>=0))],delta=3e-15)
        V=np.eye(6); V[1,1]=V[4,4]=1.4; r=mixed_gaussian_low_clicks(V)
        self.assertEqual(r['isolated_vacuum_modes'],[0,2]); self.assertEqual(np.count_nonzero(r['probabilities']),2)

    def test_invalid_and_underflow_inputs(self):
        for V in (np.eye(3),.9*np.eye(4),np.array([[1.,.1],[0.,1.]]),np.eye(2)*(1+1j)):
            with self.assertRaises(ValueError): mixed_gaussian_low_clicks(V)
        for backend in ('float64','mpmath'):
            with self.assertRaisesRegex(ArithmeticError,'underflowed'): mixed_gaussian_low_clicks(1e200*np.eye(4),backend=backend)
        with self.assertRaisesRegex(ArithmeticError,'supplied covariance'):
            mixed_gaussian_low_clicks((1-2e-12)*np.eye(2),backend='mpmath')


if __name__=='__main__': unittest.main()
