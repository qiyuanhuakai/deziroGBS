import sys,unittest
from pathlib import Path
from fractions import Fraction as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'experiments'))
from iteration41_ablation import common_thermal,common_thermal_law,matching_mass,recurrence,enumerate_laws
from iteration41_physical import coefficients,target_law,covariance,half_l1
from iteration41_independent import determinant,parity_law,word_probability
from iteration41_symbolic import identity

class AblationTests(unittest.TestCase):
    def test_independent_variables_recover_product(self):
        p=[F(1,3),F(2,5),F(3,7)];c={(j,i):F(0) for i in range(3) for j in range(i)}
        result=enumerate_laws(p,c)
        self.assertEqual(result['matching'],result['sampler']);self.assertEqual(result['matching'],parity_law(p,c))
    def test_weighted_physical_vacuum_determinant(self):
        w=[F(1),F(1,2),F(1,4)];a=F(3,2);V=covariance(w,a,total=3)
        Q=[[(V[i][j]+(i==j))/2 for j in range(6)] for i in range(6)]
        self.assertEqual(determinant(Q),(1+a*sum(x*x for x in w))**2)
        law=target_law(w,a);self.assertEqual(sum(law),1);self.assertGreater(min(law),0)
    def test_four_mode_negative_mass_and_threshold(self):
        p,c=common_thermal(4,4);self.assertEqual(matching_mass('1110',p,c),F(-256,16875))
        self.assertEqual(recurrence('1100',p,c)['records'][-1]['threshold'],F(23,15))
    def test_four_mode_observable_error(self):
        p,c=common_thermal(4,4);r=enumerate_laws(p,c);t=common_thermal_law(4,4)
        self.assertEqual(sum(t[m] for m in range(16) if m.bit_count()==3),F(512,3315))
        self.assertEqual(sum(r['sampler'][m] for m in range(16) if m.bit_count()==3),0)
        self.assertEqual(half_l1(r['sampler'],t),F(229376,1243125))
    def test_seven_mode_no_factorization_difference(self):
        p,c=coefficients([F(1,1<<(j%3)) for j in range(7)],F(1,4));r=enumerate_laws(p,c)
        self.assertEqual(r['matching'],r['unclamped'])
    def test_eight_mode_difference_without_clamps(self):
        p,c=common_thermal(8,F(1,4));r=enumerate_laws(p,c)
        self.assertGreater(min(r['matching']),0);self.assertEqual(r['sampler'],r['unclamped']);self.assertEqual(r['saturated'],[])
        self.assertEqual(half_l1(r['matching'],r['sampler']),F(32768,31640625))
        self.assertEqual({(r['matching'][m]-r['sampler'][m])*(-1)**m.bit_count() for m in range(256)},{16*F(2,75)**4})
    def test_symbolic_first_missing_matchings(self):
        self.assertEqual(identity(7)[2].terms,{})
        delta=identity(8)[2];self.assertEqual(len(delta.terms),16)
        self.assertTrue(all(len(m)==4 and c==1 for m,c in delta.terms.items()))
    def test_finite_word_zero_endpoint_is_not_zero(self):
        self.assertEqual(word_probability(0.),F(1,1<<64));self.assertEqual(word_probability(-1.),0);self.assertEqual(word_probability(float('nan')),1)
    def test_invalid_dimension_and_coefficients(self):
        with self.assertRaises(ValueError):common_thermal(0,1)
        with self.assertRaises(ValueError):matching_mass('01',[F(1,2)]*2,{})
        with self.assertRaises(ValueError):coefficients([],1)
if __name__=='__main__':unittest.main()
