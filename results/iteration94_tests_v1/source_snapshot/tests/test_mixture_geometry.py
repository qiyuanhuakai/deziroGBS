import unittest
from fractions import Fraction as F
from gbslab.mixture_geometry import cancellation_geometry


class MixtureGeometryTests(unittest.TestCase):
    def test_balanced_opposites_cancel(self):
        g=cancellation_geometry([1,0],[0,1],[(F(1,2),F(1,2))]*2);self.assertEqual(g['cancellation_interval'],(F(1,2),F(1,2)));self.assertEqual(g['improvement_over_best_interval'],(F(1,2),F(1,2)))
    def test_mixing_perfect_branch_can_hurt(self):
        g=cancellation_geometry([1,0],[0,1],[(1,1),(0,0)]);self.assertEqual(g['cancellation_interval'],(0,0));self.assertEqual(g['improvement_over_best_interval'],(-F(1,2),-F(1,2)))
    def test_common_bias_does_not_cancel(self):
        g=cancellation_geometry([F(3,4),F(1,4)],[F(2,3),F(1,3)],[(F(1,2),F(1,2))]*2);self.assertEqual(g['cancellation_interval'],(0,0));self.assertEqual(g['shared_over_interval'],(F(1,6),F(1,6)));self.assertEqual(g['shared_under_interval'],(F(1,6),F(1,6)))
    def test_interval_crossings_are_enclosed(self):
        p=[F(3,4),F(1,4)];q=[F(1,4),F(3,4)];g=cancellation_geometry(p,q,[(F(1,8),F(7,8))]*2)
        for numerator in range(1,8):
            t=F(numerator,8);r=cancellation_geometry(p,q,[(t,t),(1-t,1-t)])
            for name in ('cancellation_interval','improvement_over_best_interval'):
                self.assertLessEqual(g[name][0],r[name][0]);self.assertGreaterEqual(g[name][1],r[name][1])
    def test_invalid_laws_rejected(self):
        with self.assertRaises(ValueError):cancellation_geometry([1,1],[1,0],[(0,1)]*2)
        with self.assertRaises(ValueError):cancellation_geometry([1,0],[1,0],[(0,F(1,4))]*2)

if __name__=='__main__':unittest.main()
