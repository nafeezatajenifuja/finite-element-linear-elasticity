"""L2 rod: analytical displacement, error convergence, and stress recovery."""

import numpy as np

from fem import Material, solve
from fem.meshes import rod_mesh
from fem.postprocessing import error_norms
from fem.stress_recovery import rod_stresses
from ._common import options, plt, save_figure, save_result


def run(full: bool = False) -> dict:
    """Run the original mesh sequence with consistent body force rho*g."""
    length, young, density, gravity = 1., 250e6, 1500., 10.
    body = density*gravity
    material = Material.from_young_poisson(young, 0.)
    counts = [5, 10, 20, 50, 100, 200, 500]
    def exact(location):
        x = location[0]
        return np.array([body/young*(length*x-.5*x*x)])
    def exact_gradient(location):
        return np.array([[body/young*(length-location[0])]])
    l2, h1, sizes = [], [], []
    for count in counts:
        mesh, constraints = rod_mesh(count, length)
        displacement = solve(mesh, material, constraints, body_force=[body])
        e_l2, e_h1 = error_norms(mesh, displacement, exact, exact_gradient)
        l2.append(e_l2)
        h1.append(e_h1)
        sizes.append(length/count)
    slopes = [float(np.polyfit(np.log(sizes), np.log(errors), 1)[0]) for errors in (l2, h1)]
    figure, axes = plt.subplots(figsize=(7, 4.5))
    axes.loglog(sizes, l2, "o-", label=f"L2 estimate (slope {slopes[0]:.2f})")
    axes.loglog(sizes, h1, "s-", label=f"H1 seminorm (slope {slopes[1]:.2f})")
    axes.set(xlabel="Element length h", ylabel="Quadrature-estimated error", title="Rod self-weight: mesh refinement")
    axes.invert_xaxis()
    axes.grid(True, which="both", alpha=.25)
    axes.legend()
    convergence_figure = save_figure(figure, "convergence/rod_error_convergence.png")
    mesh, constraints = rod_mesh(8, length)
    displacement = solve(mesh, material, constraints, body_force=[body])
    grid = np.linspace(0, length, 200)
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(grid, body/young*(length*grid-.5*grid**2), "k-", label="Analytical")
    axes[0].plot(mesh.coordinates[:, 0], displacement[:, 0], "o", label="L2 FEM")
    axes[0].set(xlabel="x [m]", ylabel="Axial displacement [m]", title="Eight-element displacement")
    axes[0].legend()
    axes[1].plot(grid, body*(length-grid), "k-", label="Analytical")
    for mode, style in [("EleNodes", ".--"), ("AvgNodes", "o-"), ("GPT", "+")]:
        recovered = rod_stresses(mesh, displacement, material, mode)
        axes[1].plot(recovered[:, 0], recovered[:, 1], style, label=mode)
    axes[1].set(xlabel="x [m]", ylabel="Axial stress [Pa]", title="Consistent rho*g loading; area cancels")
    axes[1].legend(fontsize=8)
    stress_figure = save_figure(figure, "stress_recovery/rod_displacement_and_stress.png")
    return dict(elements=counts, h=sizes, l2=l2, h1_seminorm=h1, slopes=slopes,
                tip_displacement=float(displacement[-1, 0]), body_force=body,
                figures=[convergence_figure, stress_figure],
                correction="Stress example uses rho*g, not rho*g*A; originals are archived.")


if __name__ == "__main__":
    save_result("rod_self_weight", run(options(__doc__).full))
