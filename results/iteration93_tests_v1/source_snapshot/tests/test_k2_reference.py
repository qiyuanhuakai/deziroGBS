import sys,unittest
from pathlib import Path
from fractions import Fraction as F
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
from iteration40_k2_reference import Dyadic,forward


class K2ReferenceTests(unittest.TestCase):
    def test_outward_arithmetic_including_negative_values(self):
        ar=Dyadic(128)
        for x in (F(-7,3),F(-1,17),F(0),F(1,19),F(7,3)):
            for y in (F(-5,7),F(1,11),F(3)):
                a,b=ar.enclose(x),ar.enclose(y)
                for result,want in ((ar.add(a,b),x+y),(ar.mul(a,b),x*y),(ar.neg(a),-x)):
                    lo,hi=ar.fractions(result);self.assertLessEqual(lo,want);self.assertGreaterEqual(hi,want)
                c=ar.exact(F(-3,8));lo,hi=ar.fractions(ar.coefficient(a,c));self.assertLessEqual(lo,x*F(-3,8));self.assertGreaterEqual(hi,x*F(-3,8))

    def test_independent_zero_cumulants(self):
        target=dict(modes=8,cutoff=2,first_hex=[float(F(i-4,8)).hex() for i in range(8)],pairs=[[j,i,0.0.hex()] for i in range(8) for j in range(i)])
        for prefix in ('01010101','11111111','00000000'):
            r=forward(target,prefix);mass=F(1)
            for i,row in enumerate(r['records']):
                q=(1+F(float.fromhex(target['first_hex'][i])))/2;self.assertEqual(row['threshold'],(q,q));self.assertEqual(row['previous'],(mass,mass));mass*=q if prefix[i]=='0' else 1-q
            self.assertEqual(r['final_previous'],(mass,mass))

    def test_two_mode_covariance_term(self):
        target=dict(modes=2,cutoff=2,first_hex=[0.0.hex(),0.0.hex()],pairs=[[0,1,float(F(1,2)).hex()]])
        self.assertEqual(forward(target,'00')['records'][1]['threshold'],(F(3,4),F(3,4)))
        self.assertEqual(forward(target,'10')['records'][1]['threshold'],(F(1,4),F(1,4)))

    def test_absolute_clamp_is_not_conditional_clamp(self):
        target=dict(modes=2,cutoff=2,first_hex=[0.0.hex(),0.0.hex()],pairs=[[0,1,2.0.hex()]])
        r=forward(target,'00');self.assertEqual(r['records'][1]['threshold'],(F(3,2),F(3,2)))
        with self.assertRaises(ArithmeticError):forward(target,'01')

if __name__=='__main__':unittest.main()
