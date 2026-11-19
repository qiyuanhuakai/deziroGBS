import unittest
import math
from fractions import Fraction as F
from itertools import product
from gbslab.interval_completion import moment_bridge,physical_tv_interval
from gbslab.pairwise_completion import feature_masks,character,exact_moment_projection,entropy_certificate
from gbslab.rational_tree import tree_control

class IntervalCompletionTests(unittest.TestCase):
    def test_every_interval_corner_has_exact_positive_bridge(self):
        law=[F(1,4)]*4;intervals=[(-F(1,100),F(1,100))]*3;c=moment_bridge(law,intervals,[0]*3,2)
        for corner in product(*intervals):
            r=exact_moment_projection(law,corner,2)['law'];self.assertGreater(min(r),0)
            self.assertLessEqual(sum(abs(x-y) for x,y in zip(r,law))/2,c['TV_to_true_feasible_upper'])
            for x,(a,b) in zip(r,c['feasible_atom_intervals']):self.assertLessEqual(a,x);self.assertLessEqual(x,b)

    def test_degenerate_intervals_recover_exact_certificate(self):
        law=[F(1,4)]*4;c=moment_bridge(law,[(0,0)]*3,[0]*3,2);e=entropy_certificate(law,[0]*3,[0]*3,2)
        self.assertEqual(c['TV_to_true_feasible_upper'],0)
        self.assertEqual(c['TV_to_true_maxent_upper'],e['TV_to_exact_maximum_entropy_upper'])

    def test_uncertain_true_maxent_two_modes_is_enclosed(self):
        law=[F(1,4)]*4;intervals=[(-F(1,50),F(1,50))]*3;c=moment_bridge(law,intervals,[0]*3,2)
        # For two modes the three moments determine the entire law.
        for corner in product(*intervals):
            true=exact_moment_projection(law,corner,2)['law'];self.assertLessEqual(sum(abs(x-y) for x,y in zip(true,law))/2,c['TV_to_true_maxent_upper'])

    def test_bridge_refuses_unsafe_positive_claim(self):
        with self.assertRaises(ArithmeticError):moment_bridge([F(1,4)]*4,[(-1,1)]*3,[0]*3,2)

    def test_bridge_refuses_moments_outside_input_box(self):
        with self.assertRaises(ValueError):moment_bridge([F(1,2)]*2,[(F(1,10),F(1,5))],[0],1)

    def test_physical_interval_bound_handles_zero_padding(self):
        q=[F(1,3),F(1,3),F(1,3),F(0)];p=[(F(1,4),F(1,4))]*4
        self.assertEqual(physical_tv_interval(q,p),(F(1,4),F(1,4)))
        with self.assertRaises(ValueError):physical_tv_interval([F(1,2)],[(0,1)])

class RationalTreeTests(unittest.TestCase):
    def test_two_mode_tree_recovers_positive_joint(self):
        law=[F(j,10) for j in (1,2,3,4)];mu=[sum(x*character(k,m) for k,x in enumerate(law)) for m in feature_masks(2)]
        self.assertEqual(tree_control(mu,2)['law'],law)

    def test_independent_tree_has_product_law(self):
        means=[F(1,3),F(-1,5),F(1,7)];mu=means+[means[i]*means[j] for j in range(3) for i in range(j)];t=tree_control(mu,3)
        self.assertEqual(t['law'],[math.prod((1+(-1)**((mask>>j)&1)*means[j])/2 for j in range(3)) for mask in range(8)])

    def test_tree_does_not_claim_all_pair_moments(self):
        # Positive ferromagnetic parity table with equal pair moments.
        law=[(1+F(1,5)*sum(character(k,m) for m in (3,5,6)))/8 for k in range(8)];mu=[sum(x*character(k,m) for k,x in enumerate(law)) for m in feature_masks(3)];t=tree_control(mu,3);edges={(i,j) for i,j,_ in t['selected_edges']}
        for i,j in edges:self.assertEqual(sum(x*character(k,(1<<i)|(1<<j)) for k,x in enumerate(t['law'])),F(1,5))
        i,j=next((i,j) for j in range(3) for i in range(j) if (i,j) not in edges)
        self.assertEqual(sum(x*character(k,(1<<i)|(1<<j)) for k,x in enumerate(t['law'])),F(1,25))

    def test_nonpositive_pair_rejected(self):
        with self.assertRaises(ArithmeticError):tree_control([0,0,2],2)

if __name__=='__main__':unittest.main()
