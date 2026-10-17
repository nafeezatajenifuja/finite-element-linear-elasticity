"""Cook's membrane: mesh sensitivity, near-incompressibility, and recovered trace."""

import numpy as np

from fem import Material, solve
from fem.meshes import cook_mesh, straight_edge_loads
from fem.stress_recovery import nodal_stresses, plane_strain_trace
from ._common import options, plt, save_figure, save_result


def problem(resolution: int, poisson: float):
    """Solve the original Q4 plane-strain Cook model with edge traction V=100."""
    mesh, constraints = cook_mesh(resolution)
    right = np.flatnonzero(np.isclose(mesh.coordinates[:, 0], 48.))+1
    loads = straight_edge_loads(mesh.coordinates, right, 100., 2)
    material = Material.from_young_poisson(100., poisson)
    displacement = solve(mesh, material, constraints, point_loads=loads)
    tip = np.argmin(np.linalg.norm(mesh.coordinates-[48., 60.], axis=1))
    return mesh, displacement, material, float(displacement[tip, 1])


def run(full: bool = False) -> dict:
    """Use original refinements in full mode; quick mode explicitly uses smaller grids."""
    counts = [2, 4, 8, 12, 16, 32, 64] if full else [2, 4, 8, 12, 16]
    tips = []
    for count in counts:
        print(f"Cook: {count} x {count}, nu=0.4999", flush=True)
        mesh, displacement, material, tip = problem(count, .4999)
        tips.append(tip)
    finest = counts[-1]
    stresses = nodal_stresses(mesh, displacement, material)
    trace = plane_strain_trace(stresses, material)
    on_line = (np.abs(mesh.coordinates[:, 0]-24.) < 1e-6) & (mesh.coordinates[:, 1] >= 22.) & (mesh.coordinates[:, 1] <= 52.)
    line_nodes = np.flatnonzero(on_line)
    line_nodes = line_nodes[np.argsort(mesh.coordinates[line_nodes, 1])]
    poisson_values = [0., .1, .25, .4, .45, .49, .4999]
    sensitivity = []
    for poisson in poisson_values:
        print(f"Cook sensitivity: {finest} x {finest}, nu={poisson:g}", flush=True)
        if poisson == .4999:
            sensitivity.append(tips[-1])
        else:
            sensitivity.append(problem(finest, poisson)[3])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(counts, tips, "o-")
    axes[0].set(xlabel="Elements per direction", ylabel="Vertical tip displacement", title="Cook: Q4 mesh sensitivity, nu=0.4999")
    axes[1].plot(poisson_values, sensitivity, "o-")
    axes[1].set(xlabel="Poisson ratio", ylabel="Vertical tip displacement", title=f"Compressibility sensitivity: {finest} x {finest}")
    for axis in axes:
        axis.grid(alpha=.25)
    sensitivity_figure = save_figure(figure, "elasticity/cook_mesh_and_poisson_sensitivity.png")
    figure, axes = plt.subplots(figsize=(6.5, 4))
    axes.plot(mesh.coordinates[line_nodes, 1], trace[line_nodes], ".-")
    axes.set(xlabel="y along x=24", ylabel="sigma_xx + sigma_yy + sigma_zz",
             title=f"Cook: averaged stress trace, {finest} x {finest}, nu=0.4999")
    axes.grid(alpha=.25)
    trace_figure = save_figure(figure, "stress_recovery/cook_trace.png")
    return dict(mesh_counts=counts, tip_displacement=tips, poisson_values=poisson_values,
                poisson_tip_displacement=sensitivity, sensitivity_mesh=finest, full=full,
                stress_line_y=mesh.coordinates[line_nodes, 1], stress_trace=trace[line_nodes],
                figures=[sensitivity_figure, trace_figure], validation="mesh/material sensitivity; no analytical Cook solution")


if __name__ == "__main__":
    save_result("cooks_membrane", run(options(__doc__).full))
