"""Gauss-Legendre quadrature for the maintained L2, Q4, and Q8 elements."""

from functools import lru_cache
from numbers import Integral

import numpy as np
from scipy.special import roots_legendre

from .mesh import FloatArray


@lru_cache(maxsize=16)
def gauss_points(element: str, order: int | None = None) -> tuple[FloatArray, FloatArray]:
    """Return points (n_gauss, dim) and weights (n_gauss,).

    The original Q8 path always uses a 3 x 3 rule; this API makes that
    restriction explicit. L2 and Q4 default to two points per direction.
    """
    element = element.upper()
    if element not in {"L2", "Q4", "Q8"}:
        raise ValueError("quadrature supports L2, Q4, and Q8")
    order = (3 if element == "Q8" else 2) if order is None else order
    if isinstance(order, bool) or not isinstance(order, Integral) or order < 1:
        raise ValueError("quadrature order must be a positive integer")
    if element == "Q8" and order != 3:
        raise ValueError("the original Q8 implementation uses order 3")
    points_1d, weights_1d = roots_legendre(int(order))
    if element == "L2":
        points, weights = points_1d[:, None], weights_1d
    else:
        x, y = np.meshgrid(points_1d, points_1d, indexing="ij")
        points = np.column_stack([x.ravel(), y.ravel()])
        weights = np.outer(weights_1d, weights_1d).ravel()
    points.setflags(write=False)
    weights.setflags(write=False)
    return points, weights
