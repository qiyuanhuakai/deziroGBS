from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
import mpmath as mp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'experiments'))
import iteration5_quality as quality
from gbslab.positive_forest import _positive_log_ratio,ForestCompletionSampler
from gbslab.exact import ThresholdModel
from positive_event_fixtures import ConditionalIndependentEvents


class PlatformNumericsTests(unittest.TestCase):
    def test_unavailable_child_rss_is_null_not_zero(self):
        with patch.object(quality,'resource',None): result=quality.child_resource_measurement()
        self.assertFalse(result['available']); self.assertIsNone(result['native_child_peak_rss_bytes'])

    def test_linux_child_high_water_unit_and_scope_are_preserved(self):
        fake=SimpleNamespace(RUSAGE_CHILDREN=7,getrusage=lambda kind: SimpleNamespace(ru_maxrss=123) if kind==7 else None)
        with patch.object(quality,'resource',fake),patch.object(quality.platform,'system',return_value='Linux'):
            result=quality.child_resource_measurement()
        self.assertTrue(result['available']); self.assertEqual(result['native_child_peak_rss_bytes'],123*1024)
        self.assertIn('does not isolate',result['scope'])

    def test_macos_child_high_water_is_already_bytes(self):
        fake=SimpleNamespace(RUSAGE_CHILDREN=7,getrusage=lambda kind: SimpleNamespace(ru_maxrss=123))
        with patch.object(quality,'resource',fake),patch.object(quality.platform,'system',return_value='Darwin'):
            self.assertEqual(quality.child_resource_measurement()['native_child_peak_rss_bytes'],123)

    def test_log_ratio_matches_high_precision_and_avoids_overflow(self):
        for a,b in [('7e-201','3e-201'),('1e300','1e-300'),('1e-300','1e300'),('1.0000000000000002','1')]:
            x,y=np.longdouble(a),np.longdouble(b); result=_positive_log_ratio(x,y)
            with mp.workdps(100): expected=mp.log(mp.mpf(str(x))/mp.mpf(str(y)))
            self.assertLess(abs(float(result)-float(expected)),2e-12 if abs(expected)>100 else 3e-16)

    def test_paired_forest_odds_match_analytic_posteriors(self):
        for odds in (-30.,-5.,0.,5.,30.):
            for forest in ('chain','static_mi_tree'):
                oracle=ConditionalIndependentEvents(odds)
                sampler=ForestCompletionSampler(ThresholdModel(1.8*np.eye(12)),0,selection='recent',forest=forest,event_oracle=oracle)
                result=sampler.conditional_details((1,)*5)
                with mp.workdps(80): expected=float(1/(1+mp.exp(-mp.mpf(odds))))
                self.assertAlmostEqual(result['click_probability'],expected,delta=3e-15)
                self.assertAlmostEqual(result['log_odds_from_paired_events'],-odds,delta=5e-13)
                self.assertLess(result['log_odds_roundoff_indicator'],1e-6)


if __name__=='__main__': unittest.main()
