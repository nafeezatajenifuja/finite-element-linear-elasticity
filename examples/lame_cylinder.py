"""Displacement-driven modified Lamé cylinder: analytical field and refinement."""

import numpy as np
from matplotlib.collections import PolyCollection

from fem import Material, solve
from fem.meshes import cylinder_mesh
from fem.postprocessing import error_norms, strain_energy
from ._common import options, plt, save_figure, save_result


def analytical_displacement(location, inner=.1, outer=1., imposed=.05):
    """Return the original u_r=C(r-ri^2/r) expressed in Cartesian coordinates."""
    radius = np.linalg.norm(location)
    radial = imposed*outer/(outer**2-inner**2)*(radius-inner**2/radius)
    return radial*np.asarray(location)/radius


def run(full: bool = False) -> dict:
    """Reproduce the original plane-stress cylinder model and error estimator."""
    counts = [4, 8, 12, 16, 24, 40] if full else [4, 8, 12]
    material = Material.from_young_poisson(1e5, .3, "plane_stress")
    sizes, energies, errors = [], [], []
    for radial in counts:
        print(f"Cylinder: {radial} radial x {4*radial} angular elements", flush=True)
        mesh, constraints = cylinder_mesh(radial, 4*radial)
        displacement = solve(mesh, material, constraints)
        energies.append(strain_energy(mesh, displacement, material))
        errors.append(error_norms(mesh, displacement, analytical_displacement)[0])
        sizes.append(.9/radial)
    slope = float(np.polyfit(np.log(sizes), np.log(errors), 1)[0])
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].loglog(sizes, errors, "o-", label=f"Measured slope {slope:.3f}")
    axes[0].set(xlabel="Radial element size h", ylabel="Displacement L2 estimate", title="Cylinder error refinement")
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[1].plot(sizes, energies, "o-")
    axes[1].set(xlabel="Radial element size h", ylabel="Strain energy", title="Computed strain energy")
    axes[1].invert_xaxis()
    for axis in axes:
        axis.grid(alpha=.25)
    convergence_figure = save_figure(figure, "convergence/cylinder_error_and_energy.png")
    mesh, constraints = cylinder_mesh(8, 40)
    displacement = solve(mesh, material, constraints)
    radius = np.linalg.norm(mesh.coordinates, axis=1)
    directions = mesh.coordinates/radius[:, None]
    radial_fem = np.sum(displacement*directions, axis=1)
    radial_exact = np.array([analytical_displacement(point) for point in mesh.coordinates])
    radial_exact = np.sum(radial_exact*directions, axis=1)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    order = np.argsort(radius)
    axes[0].plot(radius[order], radial_exact[order], "k-", label="Analytical")
    axes[0].plot(radius, radial_fem, ".", alpha=.5, label="Q4 FEM")
    axes[0].set(xlabel="Radius", ylabel="Radial displacement", title="Modified Lame: prescribed displacement")
    axes[0].legend()
    polygons = mesh.coordinates[mesh.connectivity-1]
    element_error = (radial_fem-radial_exact)[mesh.connectivity-1].mean(axis=1)
    collection = PolyCollection(polygons, array=element_error, cmap="coolwarm", edgecolors="none")
    axes[1].add_collection(collection)
    axes[1].autoscale_view()
    axes[1].set_aspect("equal")
    axes[1].set(xlabel="x", ylabel="y", title="Cell-mean radial displacement error")
    figure.colorbar(collection, ax=axes[1], label="FEM - analytical")
    field_figure = save_figure(figure, "elasticity/cylinder_analytical_comparison.png")
    return dict(radial_elements=counts, angular_elements=[4*n for n in counts], h=sizes,
                energies=energies, l2=errors, fitted_l2_slope=slope, full=full,
                figures=[convergence_figure, field_figure], mode="plane_stress",
                boundary_conditions="inner fixed; outer radial displacement 0.05; no pressure load")


if __name__ == "__main__":
    save_result("lame_cylinder", run(options(__doc__).full))
