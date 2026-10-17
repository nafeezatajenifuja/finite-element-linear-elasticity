"""Controlled Q4/Q8 bending study with a common Poisson ratio."""

import numpy as np

from fem import Material, solve
from fem.meshes import cantilever_mesh
from ._common import options, plt, save_figure, save_result


def run(full: bool = False, poisson: float = .1) -> dict:
    """Use the original beam/load settings and correct the mixed-nu comparison.

    Euler-Bernoulli theory is a one-dimensional engineering reference for this
    two-dimensional plane-strain calculation, not an exact 2D elasticity solution.
    """
    length, height, young, density, gravity = 1., .1, 200e6, 1000., 10.
    material = Material.from_young_poisson(young, poisson)
    counts = [2, 4, 8, 16, 32, 64]
    curves = {}
    for element, layers in [("Q4", 1), ("Q4", 4), ("Q8", 1)]:
        values = []
        for nx in counts:
            mesh, constraints = cantilever_mesh(nx, element, layers, length, height)
            displacement = solve(mesh, material, constraints, body_force=[0., -density*gravity])
            tip_nodes = np.flatnonzero(np.isclose(mesh.coordinates[:, 0], length))
            node = tip_nodes[np.argmin(np.abs(mesh.coordinates[tip_nodes, 1]))]
            values.append(float(displacement[node, 1]))
        curves[f"{element}, {layers} layer(s)"] = values
    beam_reference = -(density*gravity*height)*length**4/(8*young*(height**3/12))
    figure, axes = plt.subplots(figsize=(7.5, 4.5))
    for (label, values), marker in zip(curves.items(), ["s-", "d-", "o-"]):
        axes.plot(counts, values, marker, label=label)
    axes.axhline(beam_reference, color="black", linestyle="--", label="Euler-Bernoulli reference")
    axes.set(xlabel="Elements along length", ylabel="Vertical tip displacement [m]",
             title=f"Controlled beam comparison: common nu={poisson:g}")
    axes.grid(alpha=.25)
    axes.legend(fontsize=8)
    figure_path = save_figure(figure, "elasticity/cantilever_q4_q8.png")
    return dict(elements_along_length=counts, poisson=poisson, curves=curves,
                euler_bernoulli_reference=beam_reference, figures=[figure_path],
                correction="All curves use the same nu; the archived one-layer Q4 curve used nu=0.3.")


if __name__ == "__main__":
    save_result("cantilever_q4_q8", run(options(__doc__).full))
