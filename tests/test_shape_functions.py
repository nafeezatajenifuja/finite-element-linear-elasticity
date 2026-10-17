"""Interpolation and reference-derivative checks with deterministic points."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from fem.elements import shape_functions
from fem.quadrature import gauss_points


class ShapeFunctionTests(unittest.TestCase):
    def test_kronecker_property(self):
        nodes = {
            "L2": [[-1.], [1.]],
            "Q4": [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]],
            "Q8": [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.], [0., -1.], [1., 0.], [0., 1.], [-1., 0.]],
        }
        for element, coordinates in nodes.items():
            for index, point in enumerate(coordinates):
                with self.subTest(element=element, node=index):
                    values, derivatives = shape_functions(element, point)
                    assert_allclose(values, np.eye(len(coordinates))[index], atol=1e-14)
                    self.assertEqual(derivatives.shape, (len(coordinates), len(point)))

    def test_partition_and_derivatives(self):
        rng = np.random.default_rng(7336)
        for element, dim in [("L2", 1), ("Q4", 2), ("Q8", 2)]:
            for point in rng.uniform(-.8, .8, (5, dim)):
                values, derivatives = shape_functions(element, point)
                assert_allclose(values.sum(), 1., atol=1e-14)
                assert_allclose(derivatives.sum(axis=0), 0., atol=1e-14)
                numerical = []
                for axis in range(dim):
                    step = np.zeros(dim)
                    step[axis] = 1e-6
                    numerical.append((shape_functions(element, point+step)[0]-shape_functions(element, point-step)[0])/(2e-6))
                assert_allclose(derivatives, np.array(numerical).T, atol=2e-10)

    def test_gauss_measures_and_polynomials(self):
        for element, measure in [("L2", 2.), ("Q4", 4.), ("Q8", 4.)]:
            points, weights = gauss_points(element)
            assert_allclose(weights.sum(), measure, atol=1e-14)
            assert_allclose(np.dot(weights, points[:, 0]**2), measure/3, atol=1e-14)
            self.assertTrue(np.all(weights > 0))

    def test_invalid_reference_and_rule(self):
        for element, point in [("Q4", [0.]), ("Q8", [np.nan, 0.]), ("Q4", [2., 0.]), ("TET4", [0., 0., 0.])]:
            with self.assertRaises(ValueError):
                shape_functions(element, point)
        with self.assertRaises(ValueError):
            gauss_points("Q8", 2)


if __name__ == "__main__":
    unittest.main()
