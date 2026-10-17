"""Direct and averaged stress recovery from the student's implementation.

The 1D recovery strategies follow the course reference's GPT / EleNodes /
AvgNodes organization. The maintained 2D Q4 path corrects the approved
engineering-shear factor: sigma_xy = mu * gamma_xy.
"""

import numpy as np

from .elasticity import Material
from .elements import map_gradients, shape_functions
from .mesh import FloatArray, Mesh
from .quadrature import gauss_points

Q4_CORNERS = np.array([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])


def rod_stresses(mesh: Mesh, displacement: FloatArray, material: Material, mode: str = "AvgNodes") -> FloatArray:
    """Return [x, sigma] rows for the L2 rod (lambda=0, 2 mu=E).

    GPT evaluates at quadrature points; EleNodes retains separate element
    endpoint values; AvgNodes takes the arithmetic mean of adjacent values.
    """
    if mesh.element != "L2" or material.lambda_ != 0:
        raise ValueError("rod recovery is maintained for L2 with lambda=0")
    if mode not in {"GPT", "EleNodes", "AvgNodes"}:
        raise ValueError("mode must be GPT, EleNodes, or AvgNodes")
    references = gauss_points("L2")[0] if mode == "GPT" else np.array([[-1.], [1.]])
    rows = []
    accumulator = np.zeros(len(mesh.coordinates))
    count = np.zeros(len(mesh.coordinates))
    for nodes in mesh.connectivity:
        coordinates = mesh.coordinates[nodes-1]
        local = displacement[nodes-1]
        for index, reference in enumerate(references):
            values, derivatives = shape_functions("L2", reference)
            gradients, _ = map_gradients(coordinates, derivatives)
            sigma = 2 * material.mu * float((gradients.T @ local)[0, 0])
            rows.append([float((values @ coordinates)[0]), sigma])
            if mode == "AvgNodes":
                node = nodes[index]-1
                accumulator[node] += sigma
                count[node] += 1
    if mode == "AvgNodes":
        return np.column_stack([mesh.coordinates[:, 0], accumulator / np.maximum(count, 1)])
    return np.asarray(rows)


def nodal_stresses(mesh: Mesh, displacement: FloatArray, material: Material) -> FloatArray:
    """Recover Q4 [sigma_xx, sigma_yy, sigma_xy] by averaging corner stresses."""
    if mesh.element != "Q4":
        raise ValueError("the demonstrated 2D nodal recovery uses Q4 elements")
    stress = np.zeros((len(mesh.coordinates), 3))
    count = np.zeros(len(mesh.coordinates))
    for nodes in mesh.connectivity:
        coordinates = mesh.coordinates[nodes-1]
        local = displacement[nodes-1]
        for local_index, reference in enumerate(Q4_CORNERS):
            _, derivatives = shape_functions("Q4", reference)
            gradients, _ = map_gradients(coordinates, derivatives)
            gradient = local.T @ gradients
            engineering_strain = np.array([
                gradient[0, 0], gradient[1, 1], gradient[0, 1] + gradient[1, 0]
            ])
            node = nodes[local_index]-1
            stress[node] += material.matrix() @ engineering_strain
            count[node] += 1
    return stress / np.maximum(count[:, None], 1)


def plane_strain_trace(stress: FloatArray, material: Material) -> FloatArray:
    """Include sigma_zz in the trace for the plane-strain Cook experiment."""
    in_plane_trace = stress[:, 0] + stress[:, 1]
    return in_plane_trace * (1 + material.lambda_ / (2 * (material.lambda_ + material.mu)))
