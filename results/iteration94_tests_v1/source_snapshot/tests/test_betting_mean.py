import unittest
from fractions import Fraction as F
from itertools import product
from math import prod
from unittest.mock import patch
from gbslab.betting_mean import mixture_ratio,lower_bound,mean_interval,DEFAULT_STAKES


class BettingMeanTests(unittest.TestCase):
    def test_integer_mixture_matches_direct_products(self):
        values=[F(0),F(1,7),F(2,3),F(1),F(1,7)]
        for m in (F(1,31),F(1,2),F(1)):
            exact=sum(prod((1-t+t*x/m for x in values),start=F(1)) for t in DEFAULT_STAKES)/len(DEFAULT_STAKES);a,b=mixture_ratio(values,m);self.assertEqual(F(a,b),exact)

    def test_finite_law_level_by_exhaustive_outcomes(self):
        # Exact non-Bernoulli population, including zero and one.
        atoms=(F(0),F(1,3),F(1));probs=(F(1,4),F(1,2),F(1,4));mean=sum(x*p for x,p in zip(atoms,probs));alpha=F(1,5);expectation=F(0);rejected=F(0)
        for indexes in product(range(3),repeat=4):
            values=[atoms[i] for i in indexes];mass=prod((probs[i] for i in indexes),start=F(1));a,b=mixture_ratio(values,mean);e=F(a,b);expectation+=mass*e
            if e>=1/alpha:rejected+=mass
        self.assertEqual(expectation,1);self.assertLessEqual(rejected,alpha)

    def test_quantile_exact_endpoints_and_complement(self):
        values=[F(n,12) for n in (0,1,3,3,4,6,7,7,8,9,11,12)];alpha=F(1,100);r=mean_interval(values,alpha,bits=16);scale=1<<16
        for sample,q in ((values,r['lower']),([1-v for v in values],1-r['upper'])):
            if q:self.assertGreaterEqual(F(*mixture_ratio(sample,q)),1/alpha)
            self.assertLess(F(*mixture_ratio(sample,q+F(1,scale))),1/alpha)
        self.assertEqual(r['lower'],lower_bound(values,alpha,bits=16,proposal=False))

    def test_corrupt_proposals_cannot_change_answer(self):
        values=[F(i,20) for i in range(21)];alpha=F(1,100);expected=lower_bound(values,alpha,bits=12,proposal=False)
        for bad in (float('nan'),0.,1.,.2):
            with patch('gbslab.betting_mean._proposal',return_value=bad):self.assertEqual(lower_bound(values,alpha,bits=12),expected)

    def test_boundary_and_repeated_values(self):
        a=mean_interval([0]*16,F(1,100),bits=16);b=mean_interval([1]*16,F(1,100),bits=16);self.assertEqual(a['lower'],0);self.assertEqual(b['upper'],1);self.assertEqual(a['upper'],1-b['lower']);self.assertGreater(a['upper'],0);self.assertLess(b['lower'],1)
        r=mean_interval([F(1,2)]*64,F(1,100));self.assertLess(r['lower'],F(1,2));self.assertGreater(r['upper'],F(1,2))

    def test_single_full_stake_matches_original_product_bound(self):
        from gbslab.mean_witness import product_mean_interval
        for values in ([F(1,3),F(1,2),F(2,3)]*8,[F(0),F(1,2)]*12):
            r=mean_interval(values,F(1,100),stakes=(F(1),),bits=16);old=product_mean_interval(values,F(1,100),bits=16);self.assertEqual((r['lower'],r['upper']),(old['lower'],old['upper']))

    def test_invalid_values_and_parameters(self):
        for values in ([],[-1],[F(11,10)]):
            with self.assertRaises(ValueError):lower_bound(values,F(1,100))
        for stakes in ((),(0,),(2,),(F(1,2),F(1,2))):
            with self.assertRaises(ValueError):lower_bound([F(1,2)],F(1,100),stakes=stakes)
        for alpha in (0,1):
            with self.assertRaises(ValueError):lower_bound([F(1,2)],alpha)
        with self.assertRaises(ValueError):mixture_ratio([F(1,2)],0)

if __name__=='__main__':unittest.main()
