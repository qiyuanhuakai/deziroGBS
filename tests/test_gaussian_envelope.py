import unittest
from fractions import Fraction as F
from math import comb
import numpy as np
from gbslab.certified_gaussian import ExactPairingModel
from gbslab.gaussian_envelope import GaussianEnvelope,binomial_tail,binomial_lower,target_count_tail_upper


class GaussianEnvelopeTests(unittest.TestCase):
    def test_vacuum_exact(self):
        e=GaussianEnvelope(ExactPairingModel(np.zeros((3,3)),np.zeros((6,0))));law=e.law((0,1,2));self.assertEqual(law['intervals'],[(F(1),F(1))]+[(F(0),F(0))]*7)

    def test_thermal_full_law(self):
        e=GaussianEnvelope(ExactPairingModel(np.zeros((3,3)),np.eye(6)/2));r=e.law((0,1,2));q=F(1,9)
        for mask,(lo,hi) in enumerate(r['intervals']):
            p=q**mask.bit_count()*(1-q)**(3-mask.bit_count());self.assertLessEqual(lo,p);self.assertGreaterEqual(hi,p);self.assertLess(hi-lo,F(1,1<<100))

    def test_rational_squeezed_vacuum(self):
        e=GaussianEnvelope(ExactPairingModel([[.25]],np.zeros((2,0)),bits=64));a,b=e.vacuum((0,));self.assertLessEqual(a*a,F(15,16));self.assertGreaterEqual(b*b,F(15,16))

    def test_moments_independent_thermal(self):
        e=GaussianEnvelope(ExactPairingModel(np.zeros((4,4)),np.eye(8)/2));r=e.click_moments()
        for interval,truth in ((r['mean_interval'],F(4,9)),(r['variance_interval'],F(32,81)),(r['second_moment_interval'],F(16,27))):self.assertLessEqual(interval[0],truth);self.assertGreaterEqual(interval[1],truth)

    def test_reject_ambiguous_modes(self):
        e=GaussianEnvelope(ExactPairingModel(np.zeros((2,2)),np.zeros((4,0))))
        for modes in ((1,0),(0,0),(-1,),(2,)):
            with self.assertRaises(ValueError):e.vacuum(modes)

    def test_model_premise_required(self):
        with self.assertRaises(TypeError):GaussianEnvelope(np.eye(4))

    def test_binomial_quantile_adjacent(self):
        for k in (1,3,8):
            x=binomial_lower(8,k,F(1,100),bits=16);self.assertLessEqual(binomial_tail(8,k,x),F(1,100));self.assertGreater(binomial_tail(8,k,x+F(1,1<<16)),F(1,100))
        self.assertEqual(binomial_lower(8,0,F(1,100)),0)

    def test_binomial_coverage_exact(self):
        n=8;alpha=F(1,20);bounds=[binomial_lower(n,k,alpha,bits=16) for k in range(n+1)]
        for j in range(17):
            p=F(j,16);rejection=sum(F(comb(n,k))*p**k*(1-p)**(n-k) for k in range(n+1) if bounds[k]>p);self.assertLessEqual(rejection,alpha)

    def test_binomial_invalid_inputs(self):
        for args in ((0,0,F(1,100)),(5,6,F(1,100)),(5,2,F(0))):
            with self.assertRaises(ValueError):binomial_lower(*args)

    def test_cantelli_and_markov(self):
        self.assertEqual(target_count_tail_upper(8,10,(F(4),F(4)),(F(2),F(2))),F(1,9));self.assertEqual(target_count_tail_upper(4,10,(F(4),F(4)),(F(2),F(2))),1);self.assertEqual(target_count_tail_upper(11,10,(F(4),F(4)),(F(2),F(2))),0)


if __name__=='__main__':unittest.main()
