import unittest
from fractions import Fraction as F
from gbslab.gaussian_box_reference import gaussian_box_reference

class GaussianBoxReferenceTests(unittest.TestCase):
    def test_vacuum(self):
        r=gaussian_box_reference([[0,0],[0,0]],[[0,0],[0,0]],[3,3]);self.assertEqual(r['distribution'],[1,0,0,0]);self.assertEqual(r['projection_probability_interval'],(1,1))
    def test_single_squeezed_dimension_three(self):
        r=gaussian_box_reference([[F(1,2)]],[[0]],[3]);self.assertEqual(r['distribution'],[F(8,9),F(1,9)]);self.assertEqual(r['integer_hermite'],[(1,0),(0,0),(1,0)])
    def test_pair_conditional_law(self):
        r=gaussian_box_reference([[0,F(1,2)],[F(1,2),0]],[[0,0],[0,0]],[3,3]);self.assertEqual(r['distribution'],[F(16,21),0,0,F(5,21)]);self.assertEqual(r['projection_probability_interval'],(F(63,64),F(63,64)))
    def test_complex_pair_phase(self):
        a=gaussian_box_reference([[0,F(1,2)],[F(1,2),0]],[[0,0],[0,0]],[3,3]);b=gaussian_box_reference([[0,0],[0,0]],[[0,F(1,2)],[F(1,2),0]],[3,3]);self.assertEqual(a['distribution'],b['distribution']);self.assertEqual(a['projection_probability_interval'],b['projection_probability_interval'])
    def test_nested_projection_mass(self):
        a=gaussian_box_reference([[F(1,2)]],[[0]],[3]);b=gaussian_box_reference([[F(1,2)]],[[0]],[5]);self.assertLess(a['projection_probability_interval'][1],b['projection_probability_interval'][0])
    def test_mixed_dimensions(self):
        r=gaussian_box_reference([[F(1,2),0],[0,0]],[[0,0],[0,0]],[3,1]);self.assertEqual(r['distribution'],[F(8,9),F(1,9),0,0])
    def test_reject_nonphysical(self):
        with self.assertRaises(ValueError):gaussian_box_reference([[1]],[[0]],[3])
    def test_reject_box_cap(self):
        with self.assertRaises(ValueError):gaussian_box_reference([[0]*8 for _ in range(8)],[[0]*8 for _ in range(8)],[5]*8)
    def test_previous_small_bridge_regression(self):
        from gbslab.gaussian_fock_bridge import hermite_box
        re=[[F(1,4),F(1,8)],[F(1,8),F(1,3)]];im=[[0,F(1,16)],[F(1,16),0]];r=gaussian_box_reference(re,im,[4,4]);occ,h=hermite_box(re,im,[4,4]);D=r['pairing_denominator'];self.assertEqual(h,[(F(a,D**(sum(o)//2)),F(b,D**(sum(o)//2))) for o,(a,b) in zip(occ,r['integer_hermite'])])
