import itertools,unittest
from fractions import Fraction as F
from gbslab.complete_path_overlap import path_probability,selected_overlap


class CompletePathOverlapTests(unittest.TestCase):
    def test_exact_full_law_identity(self):
        q=[F(1,2),F(1,3),F(1,6),0];p=[F(1,4)]*4
        r=selected_overlap(['00','10','01','11'],q,[(v,v) for v in p])
        tv=sum(abs(x-y) for x,y in zip(q,p))/2
        self.assertEqual(r['full_TV_interval'],(tv,tv))

    def test_selected_bounds_all_rational_completions(self):
        labels=['00','10'];q=[F(1,3),F(1,6),F(1,6),F(1,3)]
        r=selected_overlap(labels,q[:2],[(F(1,6),F(1,3)),(F(1,6),F(1,2))])
        for parts in itertools.product(range(7),repeat=3):
            if sum(parts)>6:continue
            p=[F(k,6) for k in (*parts,6-sum(parts))]
            if not F(1,6)<=p[0]<=F(1,3) or not F(1,6)<=p[1]<=F(1,2):continue
            tv=sum(abs(a-b) for a,b in zip(q,p))/2
            self.assertLessEqual(r['full_TV_interval'][0],tv);self.assertGreaterEqual(r['full_TV_interval'][1],tv)

    def test_unknown_suffix_not_same_as_complete_atom(self):
        # A single matching coarse marginal does not certify any fine overlap.
        r=selected_overlap(['00'],[F(1,2)],[(0,0)])
        self.assertEqual(r['full_TV_interval'][1],1)

    def test_gap_below_double_resolution_is_preserved(self):
        r=selected_overlap(['0'*144],[F(1,10**30)],[(F(1,10**29),F(1,10**29))])
        self.assertLess(r['full_TV_interval'][1],1);self.assertEqual(float(r['full_TV_interval'][1]),1.)
        self.assertEqual(r['upper_gap_from_one'],F(1,10**30))

    def test_zero_threshold_nan_path(self):
        r=path_probability('00',[['0x0p+0'],['nan']]);self.assertEqual(r['probability'],F(1,1<<64))
        self.assertEqual(path_probability('01',[['0x0p+0'],['nan']])['probability'],0)
        self.assertEqual(path_probability('00',[['0x0p+0'],['nan']],model='continuous')['probability'],0)

    def test_output_product_normalization(self):
        steps=[['0x1p-1'],['0x1p-2']]
        self.assertEqual(sum(path_probability(''.join(b),steps)['probability'] for b in itertools.product('01',repeat=2)),1)

    def test_interval_upper_need_not_equal_half_l1_upper(self):
        q=[F(1,2)]*2;p=[(F(2,5),F(9,10))]*2
        r=selected_overlap(['0','1'],q,p)
        old=sum(max(abs(x-a),abs(x-b)) for x,(a,b) in zip(q,p))/2
        self.assertEqual(r['full_TV_interval'][1],F(1,5));self.assertEqual(old,F(2,5))

    def test_invalid_inputs(self):
        for labels,q,p in ((['0','0'],[0,0],[(0,1)]*2),(['0','11'],[0,0],[(0,1)]*2),(['0'],[2],[(0,1)]),(['0','1'],[0,0],[(1,1)]*2)):
            with self.assertRaises(ValueError):selected_overlap(labels,q,p)
        with self.assertRaises(ValueError):path_probability('01',[['0x0p0']])


if __name__=='__main__':unittest.main()
