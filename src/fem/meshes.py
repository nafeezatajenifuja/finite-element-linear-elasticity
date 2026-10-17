"""Deterministic mesh generators extracted from the student's final notebook.

No supplied .msh/.mat file is needed. The plate example deliberately retains
the original stair-step geometry and legacy sampling rule, with limitations
documented in the methodology rather than silently changing the model.
"""

import numpy as np

from .mesh import Mesh


def _count(value: int, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"mesh counts must be integers >= {minimum}")


def rod_mesh(elements: int, length: float = 1.0):
    """Uniform L2 bar with the first node fixed."""
    _count(elements)
    coordinates = np.linspace(0, length, elements+1)[:, None]
    connectivity = np.column_stack([np.arange(1, elements+1), np.arange(2, elements+2)])
    return Mesh(coordinates, connectivity, "L2"), np.array([[1., 1., 0.]])


def cantilever_mesh(nx: int, element: str, layers: int = 1, length: float = 1., height: float = .1):
    """Clamped rectangular beam; Q8 retains the original constrained placeholders."""
    _count(nx)
    _count(layers)
    if element == "Q4":
        x = np.linspace(0, length, nx+1)
        y = np.linspace(-height/2, height/2, layers+1)
        xx, yy = np.meshgrid(x, y)
        coordinates = np.column_stack([xx.ravel(), yy.ravel()])
        connectivity = []
        for j in range(layers):
            for i in range(nx):
                first = j*(nx+1)+i+1
                connectivity.append([first, first+1, first+nx+2, first+nx+1])
    elif element == "Q8":
        if layers != 1:
            raise ValueError("the original Q8 beam uses one layer through the depth")
        width = 2*nx+1
        xx, yy = np.meshgrid(np.linspace(0, length, width), [-height/2, 0., height/2])
        coordinates = np.column_stack([xx.ravel(), yy.ravel()])
        connectivity = []
        for e in range(nx):
            left, middle, right = 2*e, 2*e+1, 2*e+2
            connectivity.append(np.array([
                left, right, 2*width+right, 2*width+left,
                middle, width+right, 2*width+middle, width+left,
            ])+1)
    else:
        raise ValueError("beam mesh requires Q4 or Q8")
    connectivity = np.asarray(connectivity, dtype=int)
    constrained = list(np.flatnonzero(np.isclose(coordinates[:, 0], 0.)))
    used = np.unique(connectivity)-1
    constrained.extend(np.setdiff1d(np.arange(len(coordinates)), used))
    constraints = np.array([[node+1, component, 0.] for node in constrained for component in (1, 2)])
    return Mesh(coordinates, connectivity, element), constraints


def cylinder_mesh(radial: int, angular: int, inner: float = .1, outer: float = 1., displacement: float = .05):
    """Full annulus: inner surface fixed, outer radial displacement prescribed."""
    _count(radial)
    _count(angular, 3)
    if not 0 < inner < outer:
        raise ValueError("require 0 < inner radius < outer radius")
    radii = np.linspace(inner, outer, radial+1)
    angles = np.linspace(0, 2*np.pi, angular+1)[:-1]
    coordinates = np.array([[r*np.cos(t), r*np.sin(t)] for t in angles for r in radii])
    connectivity = []
    constraints = []
    for j in range(angular):
        next_j = (j+1) % angular
        for i in range(radial):
            connectivity.append([
                j*(radial+1)+i+1, j*(radial+1)+i+2,
                next_j*(radial+1)+i+2, next_j*(radial+1)+i+1,
            ])
        node = j*(radial+1)
        constraints.extend([[node+1, 1, 0.], [node+1, 2, 0.]])
    for j in range(angular):
        node = j*(radial+1)+radial
        direction = coordinates[node] / np.linalg.norm(coordinates[node])
        constraints.extend([[node+1, 1, displacement*direction[0]], [node+1, 2, displacement*direction[1]]])
    return Mesh(coordinates, np.asarray(connectivity), "Q4"), np.asarray(constraints)


def cook_mesh(nx: int, ny: int | None = None):
    """Map a Q4 grid to the original Cook geometry, with the left side clamped."""
    ny = nx if ny is None else ny
    _count(nx)
    _count(ny)
    corners = np.array([[0., 0.], [48., 44.], [48., 60.], [0., 44.]])
    coordinates = []
    for eta in np.linspace(-1, 1, ny+1):
        for xi in np.linspace(-1, 1, nx+1):
            values = np.array([(1-xi)*(1-eta), (1+xi)*(1-eta), (1+xi)*(1+eta), (1-xi)*(1+eta)])/4
            coordinates.append(values @ corners)
    connectivity = []
    for j in range(ny):
        for i in range(nx):
            first = j*(nx+1)+i+1
            connectivity.append([first, first+1, first+nx+2, first+nx+1])
    mesh = Mesh(np.asarray(coordinates), np.asarray(connectivity), "Q4")
    left = np.flatnonzero(np.abs(mesh.coordinates[:, 0]) < 1e-8)
    constraints = np.array([[node+1, component, 0.] for node in left for component in (1, 2)])
    return mesh, constraints


def straight_edge_loads(coordinates, node_ids, traction: float, component: int):
    """Original uniform-edge nodal loading: half at endpoints, full inside.

    This assumes an axis-aligned, uniformly spaced straight edge, as in the
    course examples; it is not a general curved/high-order traction integrator.
    """
    ids = np.asarray(node_ids, dtype=int)
    if len(ids) < 2:
        raise ValueError("a loaded edge requires at least two nodes")
    points = coordinates[ids-1]
    axis = int(np.argmax(np.ptp(points, axis=0)))
    order = np.argsort(points[:, axis])
    ids = ids[order]
    parameters = points[order, axis]
    spacing = np.diff(parameters)
    if np.any(spacing <= 0) or not np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-10):
        raise ValueError("edge nodes must be uniformly spaced")
    load = np.full(len(ids), traction*(parameters[-1]-parameters[0])/(len(ids)-1))
    load[[0, -1]] *= .5
    return np.column_stack([ids, np.full(len(ids), component), load])


def plate_with_hole_mesh(height: float, length: float, diameter: float, nx: int, ny: int):
    """Reproduce the original cell-removal approximation and radial-band sampling.

    Returned sampling IDs are the legacy `HoleNodes` set. They need not be
    on the ideal circle or the actual stair-step boundary. The example reports
    this geometric discrepancy and does not claim validated circular-hole SCFs.
    """
    _count(nx, 2)
    _count(ny, 2)
    if not 0 < diameter < height or length <= diameter:
        raise ValueError("require 0 < diameter < height and length > diameter")
    xx, yy = np.meshgrid(np.linspace(0, length, nx), np.linspace(-height/2, height/2, ny))
    original = np.column_stack([xx.ravel(), yy.ravel()])
    center = np.array([length/2, 0.])
    inside = np.hypot(original[:, 0]-center[0], original[:, 1]-center[1]) < diameter/2
    connectivity = []
    for j in range(ny-1):
        for i in range(nx-1):
            first = j*nx+i
            nodes = np.array([first, first+1, first+nx+1, first+nx])
            if not np.any(inside[nodes]):
                connectivity.append(nodes+1)
    if not connectivity:
        raise ValueError("all cells removed; change geometry or resolution")
    connectivity = np.asarray(connectivity)
    used = np.unique(connectivity)
    renumber = np.zeros(used.max()+1, dtype=int)
    renumber[used] = np.arange(1, len(used)+1)
    mesh = Mesh(original[used-1], renumber[connectivity], "Q4")
    coordinates = mesh.coordinates
    left = np.flatnonzero(np.abs(coordinates[:, 0]) < 1e-10*max(1., length))+1
    right = np.flatnonzero(np.abs(coordinates[:, 0]-length) < 1e-10*max(1., length))+1
    centers = coordinates[mesh.connectivity-1].mean(axis=1)
    sampled_radius = np.hypot(centers[:, 0]-center[0], centers[:, 1]-center[1]).min()
    tolerance = .8*min(length/(nx-1), height/(ny-1))
    radius = np.hypot(coordinates[:, 0]-center[0], coordinates[:, 1]-center[1])
    sampling_ids = np.flatnonzero(np.abs(radius-sampled_radius) <= tolerance)+1
    if len(sampling_ids) < 2:
        raise ValueError("legacy radial-band sampling found too few nodes; refine the mesh")
    return mesh, left, right, sampling_ids, center
