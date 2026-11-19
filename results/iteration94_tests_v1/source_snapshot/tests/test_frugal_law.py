import unittest,math
from fractions import Fraction as F
from gbslab.frugal_law import word_uniform,zero_probability,full_law,distance_interval,uniform_model_transfer_bound,WORDS,MAX_UNIFORM


class FrugalLawTests(unittest.TestCase):
    def test_nearest_even_integer_ties_and_upper_clamp(self):
        for shift in range(1,12):
            for significand in ((1<<52),(1<<52)+1,(1<<53)-1):
                w=(significand<<shift)+(1<<(shift-1))
                if w>=WORDS:continue
                expected=min(F((significand+(significand%2))<<shift,WORDS),MAX_UNIFORM)
                self.assertEqual(word_uniform(w),expected)
        self.assertEqual(word_uniform(WORDS-1),MAX_UNIFORM)
        self.assertEqual(word_uniform(1),F(1,WORDS))

    def test_special_comparisons_and_finite_zero(self):
        self.assertEqual(zero_probability(0),F(1,WORDS))
        self.assertEqual(zero_probability(-0.0),F(1,WORDS))
        self.assertEqual(zero_probability(0,'continuous'),0)
        for q,w in ((float('nan'),1),(float('inf'),1),(-float('inf'),0),(-1,0),(2,1)):
            self.assertEqual(zero_probability(q),w)
        self.assertEqual(zero_probability(float(MAX_UNIFORM)),1)

    def test_exact_cdf_endpoint_against_independent_native_float_conversion(self):
        values=[0.,.1,.5,math.nextafter(1,0)]
        for exponent in range(-64,0):
            q=2.**exponent;values.extend((math.nextafter(q,0),q,math.nextafter(q,1)))
        for q in values:
            count=zero_probability(q)*WORDS;self.assertEqual(count.denominator,1);count=int(count)
            if count:self.assertLessEqual(min(float(count-1)*2.**-64,float(MAX_UNIFORM)),q)
            if count<WORDS:self.assertGreater(min(float(count)*2.**-64,float(MAX_UNIFORM)),q)

    def test_rare_zero_branch_continues_through_nan(self):
        rows=[dict(prefix='',threshold_hex='0x0p+0'),dict(prefix='0',threshold_hex='nan'),dict(prefix='1',threshold_hex='0x1p-1')]
        finite=full_law(2,rows);ideal=full_law(2,rows,model='continuous')
        self.assertEqual(finite['distribution'][0],F(1,WORDS));self.assertEqual(finite['distribution'][2],0)
        self.assertEqual(finite['reachable_nonfinite_prefixes'],('0',));self.assertEqual(ideal['reachable_nonfinite_prefixes'],())
        tv=sum(abs(a-b) for a,b in zip(finite['distribution'],ideal['distribution']))/2
        self.assertLessEqual(tv,uniform_model_transfer_bound(2))

    def test_bit_order_and_traversal_independence(self):
        rows=[dict(prefix='',threshold_hex='0x1p-2'),dict(prefix='0',threshold_hex='0x1p+0'),dict(prefix='1',threshold_hex='0x0p+0')]
        a=full_law(2,rows,model='continuous');b=full_law(2,list(reversed(rows)),model='continuous')
        self.assertEqual(a,b);self.assertEqual(a['distribution'],(F(1,4),0,0,F(3,4)))

    def test_tv_interval_contains_all_normalized_endpoint_targets(self):
        q=(F(1,3),F(2,3));target=((F(1,5),F(2,5)),(F(3,5),F(4,5)))
        lo,hi=distance_interval(q,target)
        for p in (F(1,5),F(1,3),F(2,5)):
            self.assertLessEqual(lo,abs(p-q[0]));self.assertGreaterEqual(hi,abs(p-q[0]))

    def test_invalid_words_trees_and_targets_are_rejected(self):
        for w in (-1,WORDS,True,1.5):
            with self.assertRaises(ValueError):word_uniform(w)
        with self.assertRaises(ValueError):full_law(2,[])
        with self.assertRaises(ValueError):full_law(1,[dict(prefix='',threshold_hex='0x0p0')]*2)
        with self.assertRaises(ValueError):zero_probability(.5,'numpy')
        with self.assertRaises(ValueError):distance_interval((F(1,2),),((F(0),F(1)),))
        with self.assertRaises(ValueError):uniform_model_transfer_bound(0)
