"""Dense free/free and free/prescribed assembly, as in the original solver."""

import numpy as np

from .elasticity import Material
from .elements import BodyForce, element_matrices
from .mesh import FloatArray, IntArray, Mesh


def assemble(
    mesh: Mesh, material: Material, ids: IntArray,
    body_force: BodyForce | None = None, order: int | None = None,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Assemble K_FF, K_FP, and f_F with the original node-major DOF ordering."""
    n_free = int(np.max(ids, initial=0))
    n_prescribed = int(np.max(-ids, initial=0))
    free_free = np.zeros((n_free, n_free))
    free_prescribed = np.zeros((n_free, n_prescribed))
    force = np.zeros(n_free)
    for nodes in mesh.connectivity:
        indices = nodes - 1
        local_stiffness, local_force = element_matrices(
            mesh.coordinates[indices], mesh.element, material, body_force, order
        )
        local_ids = ids[indices].ravel()
        rows, columns = np.meshgrid(local_ids, local_ids, indexing="ij")
        ff = (rows > 0) & (columns > 0)
        fp = (rows > 0) & (columns < 0)
        np.add.at(free_free, (rows[ff]-1, columns[ff]-1), local_stiffness[ff])
        np.add.at(free_prescribed, (rows[fp]-1, -columns[fp]-1), local_stiffness[fp])
        active = local_ids > 0
        np.add.at(force, local_ids[active]-1, local_force[active])
    return free_free, free_prescribed, force
