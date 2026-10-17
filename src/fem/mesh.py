"""Validated mesh data, retaining the original one-based node convention."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ELEMENTS = {"L2": (2, 1), "Q4": (4, 2), "Q8": (8, 2)}


@dataclass(frozen=True)
class Mesh:
    """Coordinates (n, dim) and one-based connectivity (n_elements, n_local)."""

    coordinates: FloatArray
    connectivity: IntArray
    element: str

    def __post_init__(self) -> None:
        element = self.element.upper()
        if element not in ELEMENTS:
            raise ValueError("the maintained solver supports L2, Q4, and Q8")
        nodes, dim = ELEMENTS[element]
        coordinates = np.asarray(self.coordinates, dtype=float)
        raw_connectivity = np.asarray(self.connectivity)
        if coordinates.ndim != 2 or coordinates.shape[1] != dim or not len(coordinates):
            raise ValueError(f"coordinates must have shape (n_nodes, {dim})")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates must be finite")
        if raw_connectivity.ndim != 2 or raw_connectivity.shape[1] != nodes or not len(raw_connectivity):
            raise ValueError(f"connectivity must have shape (n_elements, {nodes})")
        if not np.all(np.isfinite(raw_connectivity)) or not np.all(raw_connectivity == np.floor(raw_connectivity)):
            raise ValueError("connectivity must contain integer node IDs")
        connectivity = raw_connectivity.astype(np.int64)
        if np.any(connectivity < 1) or np.any(connectivity > len(coordinates)):
            raise ValueError("connectivity uses one-based node IDs in [1, n_nodes]")
        if any(len(set(row)) != nodes for row in connectivity):
            raise ValueError("an element cannot repeat a node ID")
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "connectivity", connectivity)
        object.__setattr__(self, "element", element)

    @property
    def dimension(self) -> int:
        return self.coordinates.shape[1]


def nodal_data(values: ArrayLike | None, mesh: Mesh, name: str) -> FloatArray:
    """Validate rows [one-based node, one-based component, value]."""
    array = np.empty((0, 3)) if values is None else np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite (n, 3) array")
    if not np.all(array[:, :2] == np.floor(array[:, :2])):
        raise ValueError(f"{name} node and component IDs must be integers")
    if np.any(array[:, 0] < 1) or np.any(array[:, 0] > len(mesh.coordinates)):
        raise ValueError(f"{name} node IDs are out of range")
    if np.any(array[:, 1] < 1) or np.any(array[:, 1] > mesh.dimension):
        raise ValueError(f"{name} component IDs are out of range")
    return array
