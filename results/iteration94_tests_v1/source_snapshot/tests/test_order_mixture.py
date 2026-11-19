import unittest
from fractions import Fraction as F
from gbslab.order_mixture import reverse_active,reversal_mixture

class OrderMixtureTests(unittest.TestCase):
    def test_known_three_bit_reversal(self):
        p=[F(0)]*8;p[1]=1;q=reverse_active(p,3);self.assertEqual(q[4],1)
    def test_padding_bits_stay_fixed(self):
        p=[F(0)]*16;p[9]=1;q=reverse_active(p,3);self.assertEqual(q[12],1)
    def test_fair_normalized_law(self):
        r=reversal_mixture([1,0,0,0],[0,1,0,0],2);self.assertEqual(r['law'],(F(1,2),0,F(1,2),0));self.assertEqual(sum(r['law']),1)
    def test_reversal_equivariance(self):
        p=[F(i+1,36) for i in range(8)];q=list(reversed(p));a=reversal_mixture(p,q,3);b=reversal_mixture(q,p,3);self.assertEqual(a['law'],reverse_active(b['law'],3))
    def test_mixture_can_hurt_better_branch(self):
        r=reversal_mixture([1,0],[0,1],1);self.assertEqual(sum(abs(a-b) for a,b in zip(r['law'],[1,0]))/2,F(1,2))
    def test_invalid_probability_table(self):
        for p,n in [([1,1],1),([1,0,0],1),([-1,2],1),([1,0],2),([1,0],True)]:
            with self.assertRaises(ValueError):reverse_active(p,n)
