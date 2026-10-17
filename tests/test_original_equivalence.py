"""Compare the unchanged formulation with the preserved student source files."""

import contextlib
import importlib
import io
from pathlib import Path
import sys
import unittest

import numpy as np
from numpy.testing import assert_allclose

from fem import Material, solve
from fem.elements import element_matrices, shape_functions
from fem.meshes import cylinder_mesh, rod_mesh


class OriginalEquivalenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_bytecode_setting = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        root = Path(__file__).resolve().parents[1]/"archive/original_course_structure/FEM_LE"
        cls.paths = [str(root/"FemFramework"), str(root/"Physics_model/Kernel_LE")]
        sys.path[:0] = cls.paths
        cls.original_shape = staticmethod(importlib.import_module("Shape_functions").ShapeFunctions)
        cls.original_local = staticmethod(importlib.import_module("calculate_local_matrices_LE").CalculateLocalMatrices)
        cls.original_gauss = staticmethod(importlib.import_module("gauss_points").GaussPoints)
        cls.original_driver = staticmethod(importlib.import_module("Driver_LE").Driver_LE)

    @classmethod
    def tearDownClass(cls):
        for path in cls.paths:
            sys.path.remove(path)
        sys.dont_write_bytecode = cls.previous_bytecode_setting

    def test_original_shape_values_and_derivatives(self):
        for element, points in [("L2", [[-.37], [.28]]), ("Q4", [[-.37, .28], [.4, -.6]]), ("Q8", [[-.37, .28], [.4, -.6]])]:
            for point in points:
                new = shape_functions(element, point)
                old = self.original_shape(element, point)
                for first, second in zip(new, old):
                    assert_allclose(first, second, atol=2e-14, rtol=2e-14)

    def test_original_element_stiffness_and_load(self):
        coordinates = np.array([[.2, .1], [1.4, .3], [1.3, 1.2], [.1, 1.1]])
        for element in ["Q4", "Q8"]:
            points = coordinates if element == "Q4" else np.vstack([coordinates, (coordinates+np.roll(coordinates, -1, axis=0))/2])
            material = Material(2., 3.)
            quadrature, weights = self.original_gauss(2, element, 2 if element == "Q4" else 3)
            old_stiffness, old_force = self.original_local(
                {"type": "Lame_params", "lambda": 2., "mu": 3.}, 2,
                np.arange(1, len(points)+1), element, {"case": "2D_gravity", "rhob": 7.},
                quadrature, weights, points,
            )
            stiffness, force = element_matrices(points, element, material, [0., -7.])
            assert_allclose(stiffness, old_stiffness, atol=5e-13, rtol=2e-13)
            assert_allclose(force, old_force[:, 0], atol=5e-13, rtol=2e-13)

    def test_original_global_solutions_with_nonzero_constraints(self):
        for mesh, constraints, material, force, case in [
            (*rod_mesh(8), Material(0., 125e6), [15000.], "1D_gravity"),
            (*cylinder_mesh(3, 12), Material.from_young_poisson(1e5, .3, "plane_stress"), [0., 0.], "2D_gravity"),
        ]:
            new = solve(mesh, material, constraints, body_force=force)
            old_material = {"type": "Lame_params", "lambda": material.lambda_, "mu": material.mu}
            with contextlib.redirect_stdout(io.StringIO()):
                old = self.original_driver(
                    mesh.connectivity, constraints, mesh.coordinates, old_material,
                    mesh.dimension, mesh.dimension, mesh.element,
                    {"case": case, "rhob": force[0] if mesh.dimension == 1 else 0.},
                    len(constraints), len(mesh.connectivity), 2, len(mesh.coordinates), None,
                )
            assert_allclose(new, old, atol=2e-13, rtol=2e-12)


if __name__ == "__main__":
    unittest.main()
