"""Global quasi-static linearized elasticity driver."""

import numpy as np
from numpy.typing import ArrayLike

from .assembly import assemble
from .boundary_conditions import add_point_loads, equation_ids
from .elasticity import Material
from .elements import BodyForce
from .mesh import FloatArray, Mesh
from .postprocessing import reconstruct_displacements


def solve(
    mesh: Mesh, material: Material, constraints: ArrayLike | None,
    body_force: BodyForce | None = None, point_loads: ArrayLike | None = None,
    order: int | None = None,
) -> FloatArray:
    """Solve K_FF u_F = f_F - K_FP u_P and return nodal displacements.

    The dense direct solve, partitioning, and physical formulation match the
    course implementation. No Newton iterations or time integration are used.
    """
    ids, prescribed = equation_ids(mesh, constraints)
    free_free, free_prescribed, force = assemble(mesh, material, ids, body_force, order)
    add_point_loads(force, ids, point_loads, mesh)
    try:
        free = np.linalg.solve(free_free, force - free_prescribed @ prescribed)
    except np.linalg.LinAlgError as exc:
        raise ValueError("singular stiffness system; check constraints, mesh, and material") from exc
    if not np.all(np.isfinite(free)):
        raise ValueError("linear solve returned nonfinite displacements")
    return reconstruct_displacements(ids, free, prescribed)
