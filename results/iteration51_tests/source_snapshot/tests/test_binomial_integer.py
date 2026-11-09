import unittest
from fractions import Fraction as F
from unittest.mock import patch
from gbslab.binomial_integer import tail_ratio,lower_bound
from gbslab.gaussian_envelope import binomial_tail,binomial_lower


class IntegerBinomialTests(unittest.TestCase):
    def test_integer_recurrence_matches_direct_polynomial(self):
        for n in range(1,21):
            for k in range(n+1):
                for p in (F(0),F(1,7),F(1,2),F(6,7),F(1)):
                    a,b=tail_ratio(n,k,p);self.assertEqual(F(a,b),binomial_tail(n,k,p))
    def test_quantile_grid_matches_existing_full_bisection(self):
        for n,k in ((32,1),(32,16),(32,31),(32,32),(96,0),(96,40)):
            self.assertEqual(lower_bound(n,k,F(1,14000),bits=16),binomial_lower(n,k,F(1,14000),bits=16))
    def test_incorrect_proposals_fall_back_to_exact_search(self):
        expected=lower_bound(32,20,F(1,1000),bits=16,proposal=False)
        for value in (float('nan'),0.,1.,.25):
            with patch('scipy.stats.beta.ppf',return_value=value):self.assertEqual(lower_bound(32,20,F(1,1000),bits=16),expected)
    def test_invalid_arguments(self):
        for args in ((0,0,F(1,2)),(2,3,F(1,2)),(2,1,-1),(True,0,F(1,2))):
            with self.assertRaises(ValueError):tail_ratio(*args)
        for alpha in (0,1,-1):
            with self.assertRaises(ValueError):lower_bound(4,2,alpha)
