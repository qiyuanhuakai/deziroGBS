import os,unittest
from pathlib import Path
import numpy as np
from gbslab.exact import ThresholdModel
from gbslab.gaussian import covariance_from_interferometer,haar_unitary
from gbslab.positive_completion import PositiveCompletionSampler,TableEventOracle
from gbslab.positive_forest import ForestCompletionSampler
from gbslab.wire_forest import WireForestSampler
from gbslab.batched_forest import install_canonical_selector


def model():
    rng=np.random.default_rng(370061);return ThresholdModel(covariance_from_interferometer(haar_unitary(6,rng),np.linspace(.3,.65,6),np.linspace(.5,.9,6)))


class ConditionalMIForestTests(unittest.TestCase):
    def test_requires_gaussian_source(self):
        with self.assertRaises(ValueError):PositiveCompletionSampler(TableEventOracle(np.ones(4)/4),1,selection='conditional_mi')

    def test_selector_against_conditioned_dense_pair(self):
        M=model();p=M.full_distribution();sampler=ForestCompletionSampler(M,2,selection='conditional_mi');prefix=(1,1,0,1,1);scores={}
        for j in (0,1,3,4):
            joint=np.zeros((2,2))
            for mask,value in enumerate(p):
                if not mask&4:joint[(mask>>j)&1,(mask>>5)&1]+=value
            joint/=joint.sum();product=joint.sum(1)[:,None]*joint.sum(0)[None,:];scores[j]=float(np.sum(joint*np.log(joint/product)))
        expected=tuple(sorted(scores,key=lambda j:(-scores[j],j))[:2]);self.assertEqual(sampler.conditional_details(prefix)['retained_clicks'],expected)

    def test_product_api_routes_to_existing_selector(self):
        s=PositiveCompletionSampler(model(),2,selection='conditional_mi');self.assertEqual(s._selector.selection,'conditional_mi');self.assertTrue(0<=s.conditional_click_probability((1,0,1,1))<=1)


class NativeConditionalMIForestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p=os.environ.get('GBSLAB_WIRE_FOREST_LIBRARY')
        if not p:raise unittest.SkipTest('Use the explicit scalar-wire library')
        cls.library=Path(p);assert cls.library.is_file()

    def test_all_prefixes_against_independent_event_table(self):
        M=model();p=M.full_distribution()
        for forest in ('chain','static_mi_tree'):
            reference=ForestCompletionSampler(M,2,selection='conditional_mi',forest=forest,cache_size=0,event_oracle=TableEventOracle(p));install_canonical_selector(reference)
            with WireForestSampler(M,2,selection='conditional_mi',forest=forest,cache_size=0,library=self.library) as s:
                for length in range(6):
                    for mask in range(1<<length):
                        h=tuple((mask>>j)&1 for j in range(length));a=s.conditional_details(h);b=reference.conditional_details(h);self.assertEqual(a['retained_clicks'],b['retained_clicks']);self.assertEqual(a['forest_edges'],b['forest_edges']);self.assertAlmostEqual(a['click_probability'],b['click_probability'],places=10)

    def test_trajectory_reset_reproduces_selection_and_probability(self):
        with WireForestSampler(model(),2,selection='conditional_mi',cache_size=0,library=self.library) as s:
            h=(1,1,0,1,1);s.conditional_details(());a=s.conditional_details(h);s.conditional_details((0,1,0));s.conditional_details(());b=s.conditional_details(h);self.assertEqual(a['retained_clicks'],b['retained_clicks']);self.assertEqual(a['click_probability'].hex(),b['click_probability'].hex())


if __name__=='__main__':unittest.main()
