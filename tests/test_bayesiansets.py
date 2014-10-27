import numpy as np
import unittest
import warnings
from unittest import TestCase
from bayesiansets import BayesianSets

__author__ = 'henk'


class TestBayesianSets(TestCase):
    """
    Tests BayesianSets.
    """

    def setUp(self):
        self.X = [[1, 1, 0, 0, 0],
                  [0, 1, 1, 0, 0],
                  [1, 0, 1, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 1, 0, 1],
                  [0, 0, 0, 0, 1]]

    def test_init(self):
        """
        Test initialization of class.
        """
        c = 3
        bs = BayesianSets(3)
        self.assertEqual(c, bs.c)

    def test_check_params(self):
        """
        Test checking of X.
        """
        # Invalid value.
        X = np.array([[1, 0, 2], [1, 1, 1]])
        bs = BayesianSets()
        with self.assertRaises(ValueError):
            bs.fit(X)

        # Columns containing all zeros.
        X = np.array([[1, 0, 0], [0, 0, 1]])
        bs = BayesianSets()
        with warnings.catch_warnings(True) as w:
            bs.fit(X)
            self.assertTrue(len(w) >= 1)
        self.assertTrue(np.allclose(bs.X, [[1, 0], [0, 1]]))

        # Columns containing all ones.
        X = np.array([[1, 1, 0], [1, 0, 1]])
        bs = BayesianSets()
        with warnings.catch_warnings(True) as w:
            bs.fit(X)
            self.assertTrue(len(w) >= 1)
        self.assertTrue(np.allclose(bs.X, [[1, 0], [0, 1]]))

    def test_fit(self):
        """
        Test computation of mean, alpha and beta.
        """
        c = 2
        mean_X = np.array([2, 2, 3, 1, 2]) / 6.

        bs = BayesianSets()
        bs.fit(np.array(self.X))
        self.assertTrue(np.allclose(bs.mean_X, mean_X))

        alpha = mean_X * c
        self.assertTrue(np.allclose(bs.alpha, alpha))

        beta = c * (1 - mean_X)
        self.assertTrue(np.allclose(bs.beta, beta))

    def test_predict(self):
        """
        Test ordering and log score using hand-validated values.
        """
        bs = BayesianSets()
        bs.fit(np.array(self.X))
        computed_ix = bs.predict([0, 1])
        expected_ix = [0, 1, 2]
        self.assertTrue(np.allclose(computed_ix[:3], expected_ix))
        self.assertTrue(np.allclose(bs.log_scores_,
                                    [-4.06, -4.41, -5.44, -6.59, -6.72, -6.72],
                                    rtol=1E-2, atol=1E-2))

if __name__ == '__main__':
    unittest.main()
