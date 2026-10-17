"""Exploratory SCF study retaining the original approximate geometry and sampling."""

import numpy as np

from fem import Material, solve
from fem.meshes import plate_with_hole_mesh, straight_edge_loads
from fem.stress_recovery import nodal_stresses
from ._common import options, plt, save_figure, save_result


def run_case(ratio: float, nx: int = 81, ny: int = 41) -> dict:
    """Reproduce one legacy geometry, with the approved shear-stress correction.

    The coarse cell-removal mesh and radial-band sampling are preserved. Their
    limitations prevent treating these values as validated circular-hole SCFs.
    """
    diameter, length = .2, 10.
    height = diameter/ratio
    mesh, left, right, sampled, center = plate_with_hole_mesh(height, length, diameter, nx, ny)
    sampled_coordinates = mesh.coordinates[sampled-1]
    node_left = sampled[np.argmin(sampled_coordinates[:, 0])]
    node_right = sampled[np.argmax(sampled_coordinates[:, 0])]
    constraints = np.array([[node_left, 1, 0.], [node_left, 2, 0.], [node_right, 2, 0.]])
    nominal = 1e4/(15e-3)
    loads = np.vstack([
        straight_edge_loads(mesh.coordinates, left, -nominal, 1),
        straight_edge_loads(mesh.coordinates, right, nominal, 1),
    ])
    material = Material.from_young_poisson(200e6, 0.)
    displacement = solve(mesh, material, constraints, point_loads=loads)
    stress = nodal_stresses(mesh, displacement, material)[sampled-1]
    angle = np.arctan2(sampled_coordinates[:, 1]-center[1], sampled_coordinates[:, 0]-center[0])
    sine, cosine = np.sin(angle), np.cos(angle)
    hoop = stress[:, 0]*sine**2 + stress[:, 1]*cosine**2 - 2*stress[:, 2]*sine*cosine
    index = int(np.argmax(hoop))
    maximum = float(hoop[index])
    return dict(ratio=ratio, gross_factor=maximum/nominal, net_factor=maximum/nominal*(1-ratio),
                maximum_hoop_stress=maximum, maximum_location=sampled_coordinates[index],
                maximum_sample_radius=float(np.linalg.norm(sampled_coordinates[index]-center)),
                ideal_hole_radius=diameter/2, sampled_nodes=len(sampled),
                stress_at_maximum=stress[index], node_count=len(mesh.coordinates))


def run(full: bool = False) -> dict:
    """Use the original valid 81 x 41 study grid in both execution modes.

    The archived 41 x 21 single-case demo can select duplicate restraints;
    it is excluded from the maintained examples instead of silently changing BCs.
    """
    nx, ny = 81, 41
    rows = []
    for ratio in [.1, .2, .3, .4, .5, .6]:
        print(f"Exploratory SCF: d/H={ratio:g}, nodal grid {nx} x {ny}", flush=True)
        rows.append(run_case(ratio, nx, ny))
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    axes.plot([row["ratio"] for row in rows], [row["gross_factor"] for row in rows], "o-", label="Gross-section ratio")
    axes.plot([row["ratio"] for row in rows], [row["net_factor"] for row in rows], "s--", label="Net-section ratio")
    axes.set(xlabel="Nominal d/H", ylabel="Computed stress / nominal stress",
             title=f"Exploratory approximate-hole study: {nx} x {ny} nodal grid")
    axes.grid(alpha=.25)
    axes.legend()
    figure_path = save_figure(figure, "elasticity/stress_concentration_exploratory.png")
    return dict(rows=rows, full=full, nx=nx, ny=ny, figures=[figure_path],
                correction="sigma_xy = mu * engineering_gamma_xy",
                limitation="Stair-step hole; legacy radial-band sampling is not guaranteed to select hole-boundary nodes. Not a validated SCF benchmark.")


if __name__ == "__main__":
    save_result("stress_concentration", run(options(__doc__).full))
