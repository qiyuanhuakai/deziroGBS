"""Independent branch-mass controls for the research enumeration harness."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"experiments"))
from iteration6_completion import extended_law
from gbslab.exact import NumericalProbabilityError


class EnumerationMassTests(unittest.TestCase):
    def test_failed_prefix_keeps_all_its_mass_without_conditioning(self):
        class FixedFailure:
            def conditional_click_probability(self,prefix):
                if not prefix:
                    return .25
                if prefix==(0,):
                    raise NumericalProbabilityError("Controlled failure after zero")
                return .6
        q,failure,records,_=extended_law(FixedFailure(),2)
        np.testing.assert_allclose(q,[0,.1,0,.15],rtol=0,atol=1e-16)
        self.assertEqual(failure,.75)
        self.assertEqual(len(records),1)
        self.assertEqual(records[0]["prefix"],[0])

    def test_unrepresentable_positive_output_aborts_enumeration(self):
        class Tiny:
            def conditional_click_probability(self,prefix):
                return np.nextafter(0.,1.)
        with self.assertRaisesRegex(ArithmeticError,"Positive output mass"):
            extended_law(Tiny(),2)


if __name__=="__main__":
    unittest.main()
