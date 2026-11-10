import unittest
from fractions import Fraction as F
from gbslab.joint_displacement_coupling import quantization_moments,joint_coupling,checked_gram


class JointDisplacementCouplingTests(unittest.TestCase):
    def test_reflection_preserves_bin_moments(self):
        m=quantization_moments(F(1,4),4,2)
        for a,b in zip(m['rows'],reversed(m['rows'])):
            for key in ('absolute_interval','squared_interval'):
                self.assertLessEqual(max(a[key][0],b[key][0]),min(a[key][1],b[key][1]))

    def test_scale_homogeneity(self):
        a=quantization_moments(F(1,8),16,3);b=quantization_moments(F(1,4),16,3)
        self.assertEqual(tuple(2*x for x in a['absolute_interval']),b['absolute_interval'])
        self.assertEqual(tuple(4*x for x in a['squared_interval']),b['squared_interval'])

    def test_moments_obey_cauchy_and_old_l1_bound(self):
        for N,R in ((4,2),(16,3)):
            m=quantization_moments(F(1,4),N,R);g=m['grid']
            self.assertLess(m['absolute_interval'][1]**2,m['squared_interval'][0])
            self.assertLess(m['absolute_interval'][1],(1-g['tail_interval'][0])*g['interior_half_width']+g['tail_mean_excess_interval'][1])

    def test_refinement_reduces_both_residual_moments(self):
        a=quantization_moments(F(1,4),4,2);b=quantization_moments(F(1,4),16,3)
        for key in ('absolute_interval','squared_interval'):self.assertLess(b[key][1],a[key][0])

    def test_cross_terms_cancel_only_under_stated_assumptions(self):
        axes=[dict(sigma=F(1,4),cells=16,radius=3)]*2
        a=joint_coupling([[1,0],[0,1]],axes);b=joint_coupling([[1,F(1,2)],[F(1,2),1]],axes)
        self.assertEqual(a['mean_quadratic_interval'],b['mean_quadratic_interval'])
        self.assertEqual(a['joint_curved_upper'],b['joint_curved_upper'])

    def test_joint_curvature_improves_rms(self):
        r=joint_coupling([[1,0],[0,1]],[dict(sigma=F(1,4),cells=16,radius=3)]*2)
        self.assertLess(r['joint_curved_upper'],r['rms_upper'])

    def test_rank_four_independent_drives(self):
        gram=[[int(i==j) for j in range(4)] for i in range(4)]
        r=joint_coupling(gram,[dict(sigma=F(1,4),cells=16,radius=3)]*4)
        self.assertLess(r['joint_curved_upper'],r['exact_l1_upper'])

    def test_semidefinite_duplicate_directions(self):
        self.assertEqual(checked_gram([[1,1],[1,1]])[1],[1,0])

    def test_zero_generator(self):
        r=joint_coupling([[0]],[dict(sigma=F(1,4),cells=4,radius=2)])
        self.assertEqual(r['best_upper'],0);self.assertEqual(r['joint_curved_upper'],0)

    def test_large_exponent_falls_back_to_unit_bound(self):
        r=joint_coupling([[10**8]],[dict(sigma=F(1,2),cells=4,radius=2)])
        self.assertIsNone(r['exponent_at_mean_upper_interval']);self.assertEqual(r['joint_curved_upper'],1)

    def test_invalid_gram_rejected(self):
        for a in ([[0,1],[1,0]],[[1,2],[2,1]],[[1,0],[1,1]],[] ):
            with self.assertRaises(ValueError):checked_gram(a)

    def test_invalid_quantizer_rejected(self):
        for args in ((0,4,2),(F(3,4),4,2),(F(1,4),5,2),(F(1,4),4,9)):
            with self.assertRaises(ValueError):quantization_moments(*args)


if __name__=='__main__':unittest.main()
