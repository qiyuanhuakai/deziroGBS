import math
import unittest
import numpy as np
from gbslab.gaussian_pair_recurrence import pair_recurrence_tensor
from gbslab.gaussian_schmidt_mps import projected_pair_tensor


class GaussianPairRecurrenceTests(unittest.TestCase):
    def test_complex_coefficients(self):
        rng = np.random.default_rng(220901)
        basis = np.array([[0, 0], [1, 0], [0, 1], [2, 0], [1, 1], [0, 2]])
        for scale in (.1, 1.2):
            Q = scale * (rng.normal(size=(5, 5)) + 1j * rng.normal(size=(5, 5)))
            Q = (Q + Q.T) / 2
            actual, info = pair_recurrence_tensor(Q, basis, 5, basis)
            np.testing.assert_allclose(actual, projected_pair_tensor(Q, basis, 5, basis)[0], atol=2e-11, rtol=2e-12)
            self.assertLessEqual(info['cache_entries'], actual.size)

    def test_squeezed_single_mode(self):
        empty = np.zeros((1, 0), int); z = .31 + .22j
        actual, _ = pair_recurrence_tensor([[z]], empty, 12, empty)
        expected = [0 if n % 2 else z**(n//2)*math.sqrt(math.factorial(n))/(2**(n//2)*math.factorial(n//2)) for n in range(12)]
        np.testing.assert_allclose(actual.ravel(), expected, atol=1e-15)

    def test_two_mode(self):
        basis = np.arange(7)[:, None]; empty = np.zeros((1, 0), int); z = .3j
        actual, _ = pair_recurrence_tensor([[0, z], [z, 0]], basis, 7, empty)
        np.testing.assert_allclose(actual[:, :, 0], np.diag(z**np.arange(7)), atol=1e-15)

    def test_reordered_vacuum_and_spectators(self):
        basis = np.array([[1, 0], [0, 0], [2, 0]])
        Q = np.ones((5, 5), complex) * .2
        actual, _ = pair_recurrence_tensor(Q, basis, 4, basis)
        np.testing.assert_allclose(actual, projected_pair_tensor(Q, basis, 4, basis)[0], atol=1e-15)

    def test_invalid_basis(self):
        empty = np.zeros((1, 0), int)
        for basis in (np.array([[0.], [1.5]]), np.array([[0], [2]]), np.array([[0], [0]]), np.array([[1]]), np.array([[-1], [0]])):
            with self.assertRaises(ValueError):
                pair_recurrence_tensor(np.eye(2), basis, 3, empty)

    def test_guards(self):
        empty = np.zeros((1, 0), int)
        for d in (0, 25, True, 2.5):
            with self.assertRaises(ValueError): pair_recurrence_tensor([[0]], empty, d, empty)
        with self.assertRaises(MemoryError): pair_recurrence_tensor([[0]], empty, 3, empty, max_elements=2)
        with self.assertRaises(ValueError): pair_recurrence_tensor([[np.nan]], empty, 3, empty)


if __name__ == '__main__': unittest.main()
