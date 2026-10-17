"""Independent exact element checks, patch tests, and approved bug regressions."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from fem import Material, Mesh, solve
from fem.boundary_conditions import equation_ids
from fem.elements import element_matrices
from fem.meshes import cantilever_mesh, plate_with_hole_mesh, rod_mesh
from fem.stress_recovery import nodal_stresses, rod_stresses


class ElementMatrixTests(unittest.TestCase):
    def test_exact_rod_stiffness_and_body_load(self):
        stiffness, force = element_matrices([[0.], [2.]], "L2", Material(0., 5.), [3.])
        assert_allclose(stiffness, [[5., -5.], [-5., 5.]], atol=1e-13)
        assert_allclose(force, [3., 3.], atol=1e-13)

    def test_q4_rigid_modes_and_body_resultant(self):
        points = np.array([[0., 0.], [2., 0.], [2., 1.], [0., 1.]])
        stiffness, force = element_matrices(points, "Q4", Material(2., 3.), [4., -5.])
        assert_allclose(stiffness, stiffness.T, atol=1e-13)
        for field in [np.tile([1., 0.], (4, 1)), np.tile([0., 1.], (4, 1)), np.column_stack([-points[:, 1], points[:, 0]])]:
            assert_allclose(stiffness @ field.ravel(), 0., atol=2e-13)
        assert_allclose(force.reshape(4, 2).sum(axis=0), [8., -10.], atol=1e-13)
        self.assertGreaterEqual(np.linalg.eigvalsh(stiffness).min(), -1e-12)

    def test_nonzero_dirichlet_values(self):
        mesh, _ = rod_mesh(5)
        constraints = np.array([[1., 1., .2], [6., 1., .7]])
        displacement = solve(mesh, Material(0., 5.), constraints)
        assert_allclose(displacement[:, 0], .2+.5*mesh.coordinates[:, 0], atol=1e-14)

    def test_point_loads_accumulate(self):
        mesh, constraints = rod_mesh(3)
        displacement = solve(mesh, Material(0., 5.), constraints,
                             point_loads=[[4, 1, 2.], [4, 1, 3.]])
        assert_allclose(displacement[:, 0], .5*mesh.coordinates[:, 0], atol=1e-14)

    def test_patch_reproduces_affine_displacements(self):
        mesh, _ = cantilever_mesh(2, "Q4", layers=2)
        x, y = mesh.coordinates.T
        expected = np.column_stack([.03*x-.02*y+.1, .04*x+.01*y-.2])
        boundary = np.isclose(x, 0.) | np.isclose(x, 1.) | np.isclose(np.abs(y), .05)
        constraints = [[node+1, component+1, expected[node, component]]
                       for node in np.flatnonzero(boundary) for component in range(2)]
        displacement = solve(mesh, Material(2., 3.), constraints)
        assert_allclose(displacement, expected, atol=2e-14)

    def test_engineering_shear_stress_regression(self):
        mesh, _ = cantilever_mesh(1, "Q4")
        gamma, mu = .02, 7.
        displacement = np.column_stack([gamma*mesh.coordinates[:, 1], np.zeros(4)])
        recovered = nodal_stresses(mesh, displacement, Material(2., mu))
        assert_allclose(recovered[:, :2], 0., atol=1e-13)
        assert_allclose(recovered[:, 2], mu*gamma, atol=1e-13)

    def test_rod_stress_uses_force_per_volume(self):
        mesh, constraints = rod_mesh(8)
        body, young = 15000., 250e6
        material = Material.from_young_poisson(young, 0.)
        displacement = solve(mesh, material, constraints, body_force=[body])
        self.assertAlmostEqual(displacement[-1, 0], body/(2*young), places=14)
        gauss = rod_stresses(mesh, displacement, material, "GPT")
        self.assertEqual(gauss.shape, (16, 2))
        average = rod_stresses(mesh, displacement, material)
        assert_allclose(average[1:-1, 1], body*(1-mesh.coordinates[1:-1, 0]), atol=1e-9)

    def test_signed_ids_and_invalid_constraints(self):
        mesh, constraints = rod_mesh(2)
        ids, prescribed = equation_ids(mesh, constraints)
        assert_array_equal(ids[:, 0], [-1, 1, 2])
        assert_allclose(prescribed, [0.])
        for invalid in [[[1, 1, 0.], [1, 1, 0.]], [[4, 1, 0.]], [[1.5, 1, 0.]]]:
            with self.assertRaises(ValueError):
                equation_ids(mesh, invalid)

    def test_inverted_element_rejected(self):
        with self.assertRaises(ValueError):
            element_matrices([[1.], [0.]], "L2", Material(0., 1.))

    def test_archived_coarse_scf_duplicate_restraint_is_rejected(self):
        mesh, _, _, sampled, _ = plate_with_hole_mesh(.2/.6, 10., .2, 41, 21)
        points = mesh.coordinates[sampled-1]
        left = sampled[np.argmin(points[:, 0])]
        right = sampled[np.argmax(points[:, 0])]
        self.assertEqual(left, right)
        with self.assertRaises(ValueError):
            equation_ids(mesh, [[left, 1, 0.], [left, 2, 0.], [right, 2, 0.]])


if __name__ == "__main__":
    unittest.main()
