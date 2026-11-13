import os,shutil,unittest,json
from pathlib import Path
import numpy as np
from gbslab.native_number_pnr import NativeNumberPNRSampler
from gbslab.native_budget_number_pnr import NativeBudgetNumberPNRSampler
ROOT=Path(__file__).resolve().parents[1]
@unittest.skipUnless(os.environ.get('GBSLAB_NUMBER_PNR_LIBRARY') and os.environ.get('GBSLAB_BUDGET_NUMBER_PNR_LIBRARY'),'explicit number libraries required')
class NumberPNRTests(unittest.TestCase):
    def model(self,budget=None,**kwargs):
        cls=NativeNumberPNRSampler if budget is None else NativeBudgetNumberPNRSampler
        lib=os.environ['GBSLAB_NUMBER_PNR_LIBRARY' if budget is None else 'GBSLAB_BUDGET_NUMBER_PNR_LIBRARY']
        return cls(kwargs.pop('v',np.eye(2,dtype=complex)),kwargs.pop('t',.3),lib,runtime_dirs=[Path(shutil.which('g++')).parent],**({} if budget is None else {'exact_total_budget':budget}),**kwargs)
    def test_archived_worker_replay(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');base=ROOT/'results/iteration77_number_pilot_v2';seed=json.loads((base/'row_1.json').read_text())['seed'];old=np.load(base/'counts_1.npz')['counts'][:1024]
        for budget in (None,64):
            with self.model(budget,v=d['v'],t=d['t']) as model:
                a=model.sample(1024,seed,workers=1);b=model.sample(1024,seed,workers=16)
            np.testing.assert_array_equal(old,a['counts']);np.testing.assert_array_equal(old,b['counts']);np.testing.assert_array_equal(a['counts'].sum(axis=1),a['total_photons'])
    def test_phase_sensitive_two_photon_interference(self):
        v=np.array([[1,1j],[1j,1]],complex)/np.sqrt(2)
        with self.model(v=v) as model:r=model.sample(4096,771901,fixed_total=2,workers=16)
        np.testing.assert_array_equal(r['counts'],np.ones((4096,2),np.uint32))
        with self.model() as model:r=model.sample(4096,771902,fixed_total=2)
        self.assertTrue(np.all(r['counts']%2==0));self.assertLess(abs((r['counts'][:,0]==2).mean()-.5),.04)
    def test_single_input_conditional_multinomial(self):
        probabilities=np.array([.2,.3,.5]);v=np.sqrt(probabilities).astype(complex)[:,None]
        with self.model(v=v,t=.4,workers=16) as model:r=model.sample(20000,771903,fixed_total=10)
        self.assertTrue(np.all(r['counts'].sum(axis=1)==10));np.testing.assert_allclose(r['counts'].mean(axis=0),10*probabilities,atol=.04,rtol=0);np.testing.assert_allclose(np.cov(r['counts'].T),10*(np.diag(probabilities)-np.outer(probabilities,probabilities)),atol=.09,rtol=0)
    def test_negative_binomial_total_mean_variance(self):
        with self.model(0,v=np.eye(32,dtype=complex),t=.5,workers=16) as model:r=model.sample(100000,771904)
        totals=r['total_photons'];self.assertFalse((totals%2).any());self.assertLess(abs(totals.mean()-32/3),.08);self.assertLess(abs(totals.var()-256/9),.5);np.testing.assert_array_equal(r['counts'].sum(axis=1),totals)
    def test_explicit_tail_and_coupling(self):
        d=np.load(ROOT/'results/iteration74_pure_quality_v1/input_1.npz');values={}
        for budget in (0,8,64):
            with self.model(budget,v=d['v'],t=d['t'],workers=16) as model:r=model.sample(8192,771905)
            values[budget]=r;self.assertEqual(r['statistics']['tail_records'],int((r['total_photons']>budget).sum()))
        for budget in (0,8):
            a,b=values[budget],values[64];np.testing.assert_array_equal(a['total_photons'],b['total_photons']);mask=a['total_photons']<=budget;np.testing.assert_array_equal(a['counts'][mask],b['counts'][mask])
        with self.model(0) as model:r=model.sample(16,771906,fixed_total=100)
        self.assertTrue(np.all(r['counts'].sum(axis=1)==100));self.assertEqual(r['statistics']['tail_records'],16)
    def test_explicit_resource_failure(self):
        with self.model(v=np.eye(32,dtype=complex)) as model:
            with self.assertRaisesRegex(ArithmeticError,'resource estimate'):model.sample(1,0,fixed_total=22)
        with self.model(22,v=np.eye(128,dtype=complex)) as model:
            with self.assertRaisesRegex(ArithmeticError,'resource estimate'):model.sample(1,0)
    def test_vacuum_empty_and_closed(self):
        for budget in (None,20):
            with self.model(budget,t=0.) as model:
                self.assertFalse(model.sample(64,0)['counts'].any());self.assertEqual(model.sample(0,0)['counts'].shape,(0,2))
                with self.assertRaisesRegex(ArithmeticError,'vacuum'):model.sample(1,0,fixed_total=2)
            with self.assertRaises(RuntimeError):model.sample(1,0)
    def test_input_contracts(self):
        for t in (-.1,1.,[.1,.2],[np.nan,np.nan]):
            with self.assertRaises(ValueError):self.model(t=t)
        for budget in (True,-2,3,500,2.5):
            with self.assertRaises(ValueError):self.model(budget)
        with self.assertRaises(ValueError):self.model(mean=[1,0,0,0])
        with self.model(20) as model:
            for fixed in (-1,1,2.5,True):
                with self.assertRaises(ValueError):model.sample(1,0,fixed_total=fixed)
