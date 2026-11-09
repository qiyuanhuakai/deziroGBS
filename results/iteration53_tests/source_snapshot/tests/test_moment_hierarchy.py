import unittest
from fractions import Fraction as F
from gbslab.moment_hierarchy import feature_masks,moments_from_vacua,project,bridge

class MomentHierarchyTests(unittest.TestCase):
    def example(self):
        p=[F(j+1,136) for j in range(16)];v=[sum(p[x] for x in range(16) if not x&m) for m in range(16)];return p,[(x,x) for x in v]
    def test_feature_counts(self):
        self.assertEqual([len(feature_masks(8,k)) for k in (2,3,4)],[36,92,162])
    def test_full_information_reconstructs(self):
        p,v=self.example();mu=[a for a,b in moments_from_vacua(v,4,4)];self.assertEqual(list(project([F(1,16)]*16,mu,4,4)['law']),p)
    def test_third_moments_from_direct_law(self):
        p,v=self.example();masks=feature_masks(4,3);got=moments_from_vacua(v,4,3);want=[sum(p[x]*(-1)**((x&m).bit_count()) for x in range(16)) for m in masks];self.assertEqual(got,[(x,x) for x in want])
    def test_uniform_bridge(self):
        r=bridge([F(1,16)]*16,[(F(0),F(0))]*14,[F(0)]*14,4,3);self.assertLess(r['TV_to_true_maxent_upper'],F(1,10**20))
    def test_invalid_degree(self):
        for n,k in [(4,5),(3,4),(0,1),(9,1),(4,True)]:
            with self.assertRaises(ValueError):feature_masks(n,k)
    def test_infeasible_projection_refused(self):
        with self.assertRaises(ArithmeticError):project([F(1,4)]*4,[F(2),F(0)],2,1)
