"""The course solver's signed equation numbering and Dirichlet constraints."""

import numpy as np
from numpy.typing import ArrayLike

from .mesh import FloatArray, IntArray, Mesh, nodal_data


def equation_ids(mesh: Mesh, constraints: ArrayLike | None) -> tuple[IntArray, FloatArray]:
    """Number free DOFs positively and prescribed DOFs negatively, from one."""
    data = nodal_data(constraints, mesh, "constraints")
    if len(data) and len(np.unique(data[:, :2], axis=0)) != len(data):
        raise ValueError("duplicate constrained DOFs are ambiguous; supply each DOF once")
    ids = np.zeros(mesh.coordinates.shape, dtype=np.int64)
    if len(data):
        nodes, components = data[:, 0].astype(int)-1, data[:, 1].astype(int)-1
        ids[nodes, components] = -np.arange(1, len(data)+1)
    free = ids == 0
    ids[free] = np.arange(1, np.count_nonzero(free)+1)
    return ids, data[:, 2].copy()


def add_point_loads(force: FloatArray, ids: IntArray, loads: ArrayLike | None, mesh: Mesh) -> None:
    """Accumulate repeated loads on free nodal DOFs, in place."""
    for node, component, value in nodal_data(loads, mesh, "point_loads"):
        equation = ids[int(node)-1, int(component)-1]
        if equation <= 0 and value != 0:
            raise ValueError("point load targets a prescribed DOF; reactions are not reported")
        if equation > 0:
            force[equation-1] += value
