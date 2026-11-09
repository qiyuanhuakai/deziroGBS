import unittest
from decimal import Decimal,localcontext
from fractions import Fraction as F
from gbslab.pairwise_completion import feature_masks,character,exact_moment_projection,log_interval,sqrt_upper,entropy_certificate,fit_candidate

class PairwiseCompletionTests(unittest.TestCase):
    def test_projection_preserves_unconstrained_parity(self):
        q=[(1+F(1,10)*character(k,1)+F(1,5)*character(k,3)+F(1,4)*character(k,7))/8 for k in range(8)]
        r=exact_moment_projection(q,[0]*6,3)['law']
        self.assertEqual(r,tuple((1+F(1,4)*character(k,7))/8 for k in range(8)))
        self.assertEqual(sum(r),1)

    def test_nonpositive_projection_rejected(self):
        with self.assertRaises(ArithmeticError):exact_moment_projection([F(1,4)]*4,[1,1,-1],2)

    def test_uniform_entropy(self):
        c=entropy_certificate([F(1,8)]*8,[0]*6,[0]*6,3)
        self.assertLess(c['entropy_gap_upper'],F(1,10**40))
        self.assertLess(c['TV_to_exact_maximum_entropy_upper'],F(1,10**20))

    def test_invalid_tables_and_moments_rejected(self):
        for n in (0,13,2.5):
            with self.assertRaises(ValueError):feature_masks(n)
        with self.assertRaises(ValueError):exact_moment_projection([0,1],[0],1)
        with self.assertRaises(ValueError):entropy_certificate([F(1,2)]*2,[1],[0],1)
        with self.assertRaises(ValueError):log_interval(0)

    def test_log_range_reduction_and_sign(self):
        with localcontext() as c:
            c.prec=180
            for value in (F(1),F(2),F(1,2),F(3,2),F(7,13),F(1,2**120),F(2**80+1,3)):
                lo,hi=log_interval(value);truth=(Decimal(value.numerator)/Decimal(value.denominator)).ln()
                self.assertLessEqual(Decimal(lo.numerator)/Decimal(lo.denominator),truth)
                self.assertGreaterEqual(Decimal(hi.numerator)/Decimal(hi.denominator),truth)
                self.assertLess(hi-lo,F(1,10**40))

    def test_sqrt_is_minimal_outward_gridpoint(self):
        for x in (F(0),F(1),F(2),F(1,10**50),F(17,31)):
            y=sqrt_upper(x);self.assertGreaterEqual(y*y,x)
            if x:self.assertLess((y-F(1,1<<80))**2,x)

    def test_maxent_certificate_is_not_target_accuracy(self):
        target=[(1+F(1,2)*character(k,7))/8 for k in range(8)];uniform=[F(1,8)]*8
        self.assertTrue(all(sum(p*character(k,m) for k,p in enumerate(target))==0 for m in feature_masks(3)))
        c=entropy_certificate(uniform,[0]*6,[0]*6,3)
        self.assertLess(c['TV_to_exact_maximum_entropy_upper'],F(1,10**20))
        self.assertEqual(sum(abs(x-y) for x,y in zip(target,uniform))/2,F(1,4))

    def test_two_mode_full_moments_recover_target(self):
        target=[F(j,10) for j in (1,2,3,4)];mu=[sum(p*character(k,m) for k,p in enumerate(target)) for m in feature_masks(2)];fit=fit_candidate(mu,2);r=exact_moment_projection(fit['probabilities'],mu,2)['law']
        self.assertEqual(list(r),target)
        self.assertLess(entropy_certificate(r,mu,fit['theta'],2)['TV_to_exact_maximum_entropy_upper'],F(1,10000))

if __name__=='__main__':unittest.main()
