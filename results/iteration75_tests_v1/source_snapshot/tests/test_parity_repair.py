import unittest
from fractions import Fraction as F
from gbslab.parity_repair import shift,marginal,feasible_interval,feasible_correction,oracle_tv_projection,first_negative_marginal

class ParityRepairTests(unittest.TestCase):
    def test_preserves_proper_marginals(self):
        p=[F(j+1,36) for j in range(8)];q=shift(p,F(1,100))
        for subset in ((),(0,),(1,),(2,),(0,1),(0,2),(1,2)):self.assertEqual(marginal(p,subset),marginal(q,subset))
    def test_endpoints_and_projection(self):
        p=[F(1,4)]*4;self.assertEqual(feasible_interval(p)['lower'],F(-1,4));self.assertEqual(feasible_interval(p)['upper'],F(1,4));r=feasible_correction(p,1);self.assertEqual(r['delta'],F(1,4));self.assertEqual(r['law'],(F(1,2),F(0),F(0),F(1,2)))
    def test_infeasible_even_when_all_proper_marginals_positive(self):
        p=[F(1,5)]*8;p[0]=p[7]=F(-1,10)
        self.assertIsNone(first_negative_marginal(p));self.assertFalse(feasible_interval(p)['feasible']);self.assertIsNone(feasible_correction(p,0)['law'])
    def test_oracle_reaches_target_in_same_fiber(self):
        p=[F(1,8)]*8;q=shift(p,F(1,32));r=oracle_tv_projection(p,q);self.assertEqual(r['delta'],F(1,32));self.assertEqual(r['TV'],0)
    def test_oracle_nonunique_flat_interval(self):
        q=[F(1,4)]*4;p=[F(3,10),F(3,10),F(1,5),F(1,5)];r=oracle_tv_projection(q,p);self.assertEqual(r['median_interval'],(F(-1,20),F(1,20)));self.assertEqual(r['TV'],F(1,10))
    def test_negative_marginal_certificate(self):
        q=[F(-1,4),F(-1,4),F(3,4),F(3,4)];r=first_negative_marginal(q);self.assertEqual(r,dict(subset=(1,),atom=0,mass=F(-1,2),order=1))
    def test_two_atom_dual_invariant(self):
        q=[F(1,5)]*8;q[0]=q[7]=F(-1,10)
        for delta in (F(-2),F(0),F(1,13)):r=shift(q,delta);self.assertEqual(r[0]+r[7],F(-1,5))
    def test_invalid_tables(self):
        for p in ([],[1],[F(1,3)]*3,[0,0]):
            with self.assertRaises(ValueError):feasible_interval(p)
        with self.assertRaises(ValueError):marginal([F(1,4)]*4,(0,0))
        with self.assertRaises(ValueError):oracle_tv_projection([F(1,4)]*4,[1,0])
if __name__=='__main__':unittest.main()
