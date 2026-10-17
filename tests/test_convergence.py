"""Small analytical refinement checks, without claiming universal convergence."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from fem import Material, solve
from fem.meshes import cylinder_mesh, rod_mesh
from fem.postprocessing import error_norms


class ConvergenceTests(unittest.TestCase):
    def test_rod_error_rates(self):
        material = Material(0., .5)
        sizes, l2, h1 = [], [], []
        for elements in [5, 10, 20, 40]:
            mesh, constraints = rod_mesh(elements)
            displacement = solve(mesh, material, constraints, body_force=[1.])
            errors = error_norms(mesh, displacement,
                                 lambda x: np.array([x[0]-.5*x[0]**2]),
                                 lambda x: np.array([[1-x[0]]]))
            sizes.append(1/elements)
            l2.append(errors[0])
            h1.append(errors[1])
        assert_allclose(np.polyfit(np.log(sizes), np.log(l2), 1)[0], 2., atol=1e-8)
        assert_allclose(np.polyfit(np.log(sizes), np.log(h1), 1)[0], 1., atol=1e-8)

    def test_cylinder_refinement_reduces_error(self):
        material = Material.from_young_poisson(1e5, .3, "plane_stress")
        def exact(x):
            radius = np.linalg.norm(x)
            return (.05/(1-.1**2))*(radius-.1**2/radius)*x/radius
        errors = []
        for radial in [4, 8]:
            mesh, constraints = cylinder_mesh(radial, 4*radial)
            displacement = solve(mesh, material, constraints)
            errors.append(error_norms(mesh, displacement, exact)[0])
        self.assertTrue(0 < errors[1] < errors[0])


if __name__ == "__main__":
    unittest.main()
