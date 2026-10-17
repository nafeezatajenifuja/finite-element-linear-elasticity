"""Displacement reconstruction, quadrature error estimates, and strain energy."""

from typing import Callable

import numpy as np

from .elasticity import Material
from .elements import map_gradients, reference_data
from .mesh import FloatArray, IntArray, Mesh


def reconstruct_displacements(ids: IntArray, free: FloatArray, prescribed: FloatArray) -> FloatArray:
    """Undo the signed equation numbering and return (n_nodes, dim)."""
    displacement = np.zeros(ids.shape)
    active = ids > 0
    displacement[active] = free[ids[active]-1]
    displacement[~active] = prescribed[-ids[~active]-1]
    return displacement


def error_norms(
    mesh: Mesh, displacement: FloatArray, exact: Callable[[FloatArray], FloatArray],
    exact_gradient: Callable[[FloatArray], FloatArray] | None = None,
    order: int | None = None,
) -> tuple[float, float | None]:
    """Estimate displacement L2 error and optionally the H1 seminorm by quadrature.

    Using the experiment's default quadrature reproduces its error estimator;
    this does not imply exact integration of the squared error.
    """
    displacement = np.asarray(displacement, dtype=float)
    if displacement.shape != mesh.coordinates.shape or not np.all(np.isfinite(displacement)):
        raise ValueError("displacements must have shape (n_nodes, dim) and be finite")
    l2_squared = 0.0
    h1_squared = 0.0
    for nodes in mesh.connectivity:
        coordinates = mesh.coordinates[nodes-1]
        local = displacement[nodes-1]
        for _, weight, values, derivatives in reference_data(mesh.element, order):
            gradients, determinant = map_gradients(coordinates, derivatives)
            location = values @ coordinates
            exact_value = np.asarray(exact(location), dtype=float)
            if exact_value.shape != (mesh.dimension,) or not np.all(np.isfinite(exact_value)):
                raise ValueError("exact displacement must be a finite vector of length dim")
            error = values @ local - exact_value
            l2_squared += weight * np.dot(error, error) * determinant
            if exact_gradient is not None:
                gradient = np.asarray(exact_gradient(location), dtype=float)
                if gradient.shape != (mesh.dimension, mesh.dimension) or not np.all(np.isfinite(gradient)):
                    raise ValueError("exact_gradient must return a finite (dim, dim) matrix")
                difference = local.T @ gradients - gradient
                h1_squared += weight * np.sum(difference**2) * determinant
    return float(np.sqrt(l2_squared)), (float(np.sqrt(h1_squared)) if exact_gradient is not None else None)


def strain_energy(mesh: Mesh, displacement: FloatArray, material: Material, order: int | None = None) -> float:
    """Integrate 0.5 sigma:epsilon using the same material constants as the solve."""
    energy = 0.0
    for nodes in mesh.connectivity:
        coordinates = mesh.coordinates[nodes-1]
        local = displacement[nodes-1]
        for _, weight, _, derivatives in reference_data(mesh.element, order):
            gradients, determinant = map_gradients(coordinates, derivatives)
            gradient = local.T @ gradients
            strain = 0.5 * (gradient + gradient.T)
            density = 0.5 * material.lambda_ * np.trace(strain)**2 + material.mu * np.sum(strain**2)
            energy += weight * density * determinant
    return float(energy)
