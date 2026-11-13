import unittest,copy
from fractions import Fraction as F
from gbslab.compact_gaussian_evidence import pack_enclosure,unpack_enclosure,semantic_digest,integer_mobius_law

class CompactEvidenceTests(unittest.TestCase):
    def fixture(self):
        return dict(modes=[0],vacuum_interval=[F(1),F(1)],relative_perturbation=F(0),perturbed_determinant_interval=[F(1),F(1)],raw_vacuum_interval=[F(1),F(1)],certificate=dict(real=[[1,0],[0,1]],imag=[[0,0],[0,0]],denominator=1,R_real=[[16,0],[0,16]],R_imag=[[0,0],[0,0]],scale=16,diagnostics={'test':True},determinant_interval=[F(1),F(1)]))
    def test_lossless_roundtrip(self):
        r=self.fixture();p=pack_enclosure(r,[[8,0],[0,8]],8);self.assertEqual(unpack_enclosure(p,[[8,0],[0,8]],8),r)
    def test_digest_tuple_list_canonical(self):self.assertEqual(semantic_digest({'x':(F(1,3),)}),semantic_digest({'x':[F(1,3)]}))
    def test_wrong_center_rejected(self):
        with self.assertRaises(ValueError):pack_enclosure(self.fixture(),[[9,0],[0,8]],8)
    def test_complex_witness_rejected(self):
        r=self.fixture();r['certificate']['R_imag'][0][0]=1
        with self.assertRaises(ValueError):pack_enclosure(r,[[8,0],[0,8]],8)
    def test_tampered_witness_rejected(self):
        p=pack_enclosure(self.fixture(),[[8,0],[0,8]],8);p['R_lower'][0]+=1
        with self.assertRaises(ValueError):unpack_enclosure(p,[[8,0],[0,8]],8)
    def test_wrong_length_rejected(self):
        p=pack_enclosure(self.fixture(),[[8,0],[0,8]],8);p['R_lower'].pop()
        with self.assertRaises(ValueError):unpack_enclosure(p,[[8,0],[0,8]],8)
    def test_two_independent_modes(self):
        p,q=F(2,3),F(3,5);v=[(1,1),(p,p),(q,q),(p*q,p*q)];r=integer_mobius_law(v);self.assertEqual(r['intervals'],[(p*q,p*q),((1-p)*q,(1-p)*q),(p*(1-q),p*(1-q)),((1-p)*(1-q),(1-p)*(1-q))])
    def test_direct_interval_inclusion(self):
        n=4;v=[]
        for mask in range(1<<n):
            p=F(1,2**mask.bit_count());v.append((max(F(0),p-F(1,10000)),min(F(1),p+F(1,10000))))
        r=integer_mobius_law(v);raw=[]
        for clicks in range(1<<n):
            zero=15^clicks;subset=clicks;a=b=F(0)
            while True:
                l,h=v[zero|subset]
                if subset.bit_count()%2:a-=h;b-=l
                else:a+=l;b+=h
                if not subset:break
                subset=(subset-1)&clicks
            raw.append((a,b))
        self.assertEqual(r['raw_intervals'],raw);self.assertEqual(r['interval_subtractions'],32)
    def test_invalid_table(self):
        with self.assertRaises(ValueError):integer_mobius_law([(0,1)]*3)
        with self.assertRaises(ValueError):integer_mobius_law([(1,0)]*2)
