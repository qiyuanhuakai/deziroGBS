import itertools
import math
import unittest
import mpmath as mp
import numpy as np
from gbslab.gaussian import haar_unitary
from gbslab.meixner_probe import threshold_projection, uniform_parameters
from gbslab.meixner_high_order import DiagonalNormalWick, occupations, projection_prefixes, threshold_transform


class HighMeixnerTests(unittest.TestCase):
    def test_degree_two_regression(self):
        u=haar_unitary(4,np.random.default_rng(1701))
        for r,eta in ((.6,.35),(.9,.65),(1.2,.85)):
            result=projection_prefixes(u,r,eta,2)
            for degree in (0,1,2):
                np.testing.assert_allclose(result['laws'][degree]['raw'],threshold_projection(u,r,eta,degree)['raw'],atol=3e-14,rtol=1e-12)

    def test_shift_identity_against_binomial_expansion(self):
        u=haar_unitary(3,np.random.default_rng(1702)); r=.7; eta=.4
        p=uniform_parameters(3,r,eta); M=-eta*np.sinh(r)*np.cosh(r)*(u@u.T)
        original=DiagonalNormalWick(p['nout'],M); shifted=DiagonalNormalWick(p['nout']-p['b'],M)
        for degree in range(6):
            for index in occupations(3,degree):
                expected=0j
                for sub in itertools.product(*(range(k+1) for k in index)):
                    coefficient=math.prod(math.comb(k,j) for k,j in zip(index,sub))*(-p['b'])**(degree-sum(sub))
                    expected+=coefficient*original.binomial_moment(sub)
                self.assertAlmostEqual(abs(shifted.binomial_moment(index)-expected),0,delta=2e-13)

    def test_diagonal_thermal_wick_moments(self):
        for n in (.3,-.3):
            engine=DiagonalNormalWick(n,np.zeros((3,3)))
            for degree in range(7):
                for index in occupations(3,degree): self.assertAlmostEqual(engine.binomial_moment(index).real,n**degree,delta=1e-14)

    def test_wick_pair_correlations(self):
        n=.3; M=np.array([[.2,.1j],[.1j,-.4]])
        w=DiagonalNormalWick(n,M)
        self.assertAlmostEqual(w.binomial_moment((2,0)).real,n*n+.5*.2**2)
        self.assertAlmostEqual(w.binomial_moment((1,1)).real,n*n+.1**2)

    def test_independent_determinant_generating_function(self):
        # Formal real-quadrature covariance shifted by reference mean.
        n=-.12; M=np.array([[.2+.1j,.12-.03j],[.12-.03j,-.15+.05j]])
        C=np.block([[n*np.eye(2)+M.real,M.imag],[M.imag,n*np.eye(2)-M.real]])
        engine=DiagonalNormalWick(n,M)
        with mp.workdps(70):
            matrix=mp.matrix([[mp.mpf(float(x)) for x in row] for row in C])
            def generating(x,y): return mp.det(mp.eye(4)-matrix*mp.diag([x,y,x,y]))**(-mp.mpf('.5'))
            for index in ((1,0),(2,2),(4,2),(3,3)):
                coefficient=mp.diff(generating,(0,0),index)/math.prod(math.factorial(k) for k in index)
                self.assertLess(abs(float(coefficient)-engine.binomial_moment(index).real),2e-14)

    def test_threshold_transform_brute_force(self):
        A=np.random.default_rng(1703).normal(size=16); b=.4
        expected=[]
        for mask in range(16):
            expected.append(sum(A[s]*math.prod(1 if (mask>>i)&1 else -b for i in range(4) if (s>>i)&1) for s in range(16)))
        np.testing.assert_allclose(threshold_transform(A,b),expected,atol=2e-15)

    def test_geometric_threshold_identity_through_six(self):
        b=.4; x=np.arange(300,dtype=float); weights=(b/(1+b))**x/(1+b)
        for degree in range(1,7):
            values=np.array([sum(math.comb(degree,j)*math.comb(int(n),j)*(-b)**(degree-j) for j in range(min(degree,int(n))+1)) for n in x])
            self.assertAlmostEqual(float(weights@values),0,delta=3e-14)
            self.assertAlmostEqual(float(weights[1:]@values[1:]/(b/(1+b))),(-b)**(degree-1),delta=3e-14)

    def test_phase_permutation_mass_and_repair(self):
        u=haar_unitary(4,np.random.default_rng(1704)); original=projection_prefixes(u,.9,.65,6)
        perm=[2,0,3,1]; transformed=projection_prefixes(u[perm]*np.exp(1j*np.arange(4))[:,None],.9,.65,6)
        indices=[sum(((mask>>j)&1)<<perm[j] for j in range(4)) for mask in range(16)]
        for d in range(7):
            row=original['rows'][d]; law=original['laws'][d]
            self.assertAlmostEqual(row['raw_mass'],1,delta=1e-12)
            self.assertAlmostEqual(law['distribution'].sum(),1,delta=1e-12)
            self.assertAlmostEqual(row['repair_l1'],2*row['negative_mass'],delta=1e-12)
            np.testing.assert_allclose(transformed['laws'][d]['raw'],law['raw'][indices],atol=5e-13)

    def test_vacuum_coefficients(self):
        result=projection_prefixes(np.eye(4),0,.35,8); b=result['parameters']['b']
        np.testing.assert_allclose(result['coefficients'],[(-b)**sum(k) for k in result['indices']],atol=2e-15)
        self.assertEqual(len(result['coefficients']),math.comb(12,8))

    def test_resource_caps_and_invalid_degree(self):
        with self.assertRaises(MemoryError): projection_prefixes(np.eye(12),.6,.35,6,max_coefficients=5)
        with self.assertRaises(MemoryError): projection_prefixes(np.eye(4),.6,.35,3,max_states=3)
        with self.assertRaises(ValueError): projection_prefixes(np.eye(4),.6,.35,9)


if __name__=='__main__': unittest.main()
