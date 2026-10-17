"""Shape functions, isoparametric mapping, and the original element integrals."""

from functools import lru_cache
from typing import Callable

import numpy as np
from numpy.typing import ArrayLike
import sympy as sp

from .elasticity import Material, transpose_matrix
from .mesh import ELEMENTS, FloatArray
from .quadrature import gauss_points

BodyForce = ArrayLike | Callable[[FloatArray], ArrayLike]


@lru_cache(maxsize=3)
def _shape_evaluators(element: str):
    """Differentiate the student's original polynomials once using SymPy."""
    x, y = sp.symbols("xi eta")
    if element == "L2":
        symbols = (x,)
        expressions = [(1 - x) / 2, (1 + x) / 2]
    elif element == "Q4":
        symbols = (x, y)
        expressions = [(1-x)*(1-y)/4, (1+x)*(1-y)/4, (1+x)*(1+y)/4, (1-x)*(1+y)/4]
    elif element == "Q8":
        symbols = (x, y)
        expressions = [
            (1-x)*(1-y)*(-x-y-1)/4, (1+x)*(1-y)*(x-y-1)/4,
            (1+x)*(1+y)*(x+y-1)/4, (1-x)*(1+y)*(-x+y-1)/4,
            (1-x**2)*(1-y)/2, (1+x)*(1-y**2)/2,
            (1-x**2)*(1+y)/2, (1-x)*(1-y**2)/2,
        ]
    else:
        raise ValueError("shape functions support L2, Q4, and Q8")
    values = sp.Matrix(expressions)
    derivatives = values.jacobian(symbols)
    return sp.lambdify(symbols, values, "numpy"), sp.lambdify(symbols, derivatives, "numpy")


def shape_functions(element: str, reference: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Return N (n_local,) and dN/dxi (n_local, dim).

    Q4/Q8 corners are BL, BR, TR, TL; Q8 then adds bottom, right, top,
    and left midside nodes. This is the original polynomial/node ordering.
    """
    element = element.upper()
    if element not in ELEMENTS:
        raise ValueError("shape functions support L2, Q4, and Q8")
    point = np.asarray(reference, dtype=float)
    if point.shape != (ELEMENTS[element][1],) or not np.all(np.isfinite(point)):
        raise ValueError("reference coordinates have the wrong dimension or are nonfinite")
    if np.any(np.abs(point) > 1 + 1e-12):
        raise ValueError("reference coordinates must lie in [-1, 1]")
    values, derivatives = _shape_evaluators(element)
    return np.asarray(values(*point), dtype=float).ravel(), np.asarray(derivatives(*point), dtype=float)


@lru_cache(maxsize=16)
def reference_data(element: str, order: int | None = None):
    """Cache shape values at the fixed quadrature points; no formulation change."""
    points, weights = gauss_points(element, order)
    return tuple((point, weight, *shape_functions(element, point)) for point, weight in zip(points, weights))


def map_gradients(coordinates: FloatArray, derivatives: FloatArray) -> tuple[FloatArray, float]:
    """Map shape gradients with J=x.T dN/dxi; reject inverted elements."""
    jacobian = coordinates.T @ derivatives
    determinant = float(np.linalg.det(jacobian))
    if not np.isfinite(determinant) or determinant <= 0:
        raise ValueError("element Jacobian must be finite and positive; check node ordering")
    return derivatives @ np.linalg.inv(jacobian), determinant


def element_matrices(
    coordinates: ArrayLike, element: str, material: Material,
    body_force: BodyForce | None = None, order: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Integrate stiffness and body load using the unchanged Kronecker formula.

    Returns K_e (n_local*dim, n_local*dim) and f_e (n_local*dim,).
    Body force is per volume (per length after the rod's area cancellation).
    Two-dimensional examples use unit thickness.
    """
    element = element.upper()
    if element not in ELEMENTS:
        raise ValueError("unsupported element")
    nodes, dim = ELEMENTS[element]
    coordinates = np.asarray(coordinates, dtype=float)
    if coordinates.shape != (nodes, dim) or not np.all(np.isfinite(coordinates)):
        raise ValueError("element coordinates have an invalid shape or nonfinite values")
    stiffness = np.zeros((nodes * dim, nodes * dim))
    force = np.zeros(nodes * dim)
    identity = np.eye(dim)
    transposer = transpose_matrix(dim)
    for _, weight, values, derivatives in reference_data(element, order):
        gradients, determinant = map_gradients(coordinates, derivatives)
        vector = gradients.T.flatten(order="F")
        kronecker = np.kron(gradients, identity)
        stiffness += weight * material.lambda_ * np.outer(vector, vector) * determinant
        stiffness += weight * material.mu * (kronecker @ kronecker.T) * determinant
        stiffness += weight * material.mu * (kronecker @ transposer @ kronecker.T) * determinant
        location = coordinates.T @ values
        load = np.zeros(dim) if body_force is None else np.asarray(
            body_force(location) if callable(body_force) else body_force, dtype=float
        )
        if load.shape != (dim,) or not np.all(np.isfinite(load)):
            raise ValueError(f"body_force must return a finite vector of shape ({dim},)")
        force += weight * (np.kron(values[:, None], identity) @ load) * determinant
    return stiffness, force
