"""Run all experiments and compare reproducible quantities with saved notebook outputs."""

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from examples import cantilever_q4_q8, cooks_membrane, lame_cylinder, rod_self_weight, stress_concentration
from examples._common import save_result


def historical_output(notebook, index):
    return "\n".join("".join(out.get("text", [])) for out in notebook["cells"][index].get("outputs", []))


def verify(results: dict, full: bool) -> list[str]:
    """Check printed values at their saved precision, with explicit rounding tolerances."""
    original = ROOT/"archive/original_course_structure/FEM_LE/Tests/Test_LE/Final_Project.ipynb"
    notebook = json.loads(original.read_bytes())
    checks = []
    def compare(label, actual, expected, relative=2e-6, absolute=1e-12):
        if len(np.atleast_1d(expected)) == 0:
            raise AssertionError(f"historical output not found: {label}")
        np.testing.assert_allclose(actual, expected, rtol=relative, atol=absolute, err_msg=label)
        checks.append(label)
    rod_rows = re.findall(r"^\s*(\d+)\s+[\d.]+\s+([\d.eE+-]+)\s+([\d.eE+-]+)\s*$", historical_output(notebook, 10), re.M)
    compare("Rod L2 estimates match saved notebook output", results["rod"]["l2"], [float(row[1]) for row in rod_rows])
    compare("Rod H1 estimates match saved notebook output", results["rod"]["h1_seminorm"], [float(row[2]) for row in rod_rows])
    compare("Rod slopes remain 2 and 1", results["rod"]["slopes"], [2., 1.], relative=1e-4)
    q8 = re.findall(r"Tip displacement = ([\d.eE+-]+)", historical_output(notebook, 15))
    compare("Unchanged Q8 beam sequence matches saved output", results["beam"]["curves"]["Q8, 1 layer(s)"], list(map(float, q8)))
    count = len(results["cylinder"]["radial_elements"])
    energies = re.findall(r"Strain energy = ([\d.eE+-]+)", historical_output(notebook, 21))
    errors = re.findall(r"L2 displacement error = ([\d.eE+-]+)", historical_output(notebook, 22))
    compare("Cylinder strain energies match original meshes", results["cylinder"]["energies"], list(map(float, energies[:count])))
    compare("Cylinder L2 estimates match original meshes", results["cylinder"]["l2"], list(map(float, errors[:count])))
    if full:
        compare("Full cylinder fitted slope matches recorded 1.7438", [results["cylinder"]["fitted_l2_slope"]], [1.7438], absolute=5e-5)
    cook_rows = re.findall(r"^\s*(\d+)x\d+: uy\(A\) = ([\d.eE+-]+)", historical_output(notebook, 27), re.M)
    cook_reference = {int(n): float(value) for n, value in cook_rows}
    compare("Cook mesh sequence matches saved output", results["cook"]["tip_displacement"], [cook_reference[n] for n in results["cook"]["mesh_counts"]])
    if full:
        sensitivity = re.findall(r"^nu =\s*[\d.]+\s*->\s*uy\(A\) = ([\d.eE+-]+)", historical_output(notebook, 33), re.M)
        compare("Cook 64x64 Poisson sweep matches saved output", results["cook"]["poisson_tip_displacement"], list(map(float, sensitivity)))
    scf_rows = re.findall(r"^\s*(0\.\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([-\d.]+)\s*$", historical_output(notebook, 48), re.M)
    compare("Exploratory SCF gross values match rounded legacy table", [row["gross_factor"] for row in results["scf"]["rows"]], [float(row[1]) for row in scf_rows], relative=0, absolute=6e-4)
    compare("Exploratory SCF net values match rounded legacy table", [row["net_factor"] for row in results["scf"]["rows"]], [float(row[2]) for row in scf_rows], relative=0, absolute=6e-4)
    checks.append("Corrected cases are labelled separately; no claim that reproduced SCF values establish accuracy")
    manifest = json.loads((ROOT/"archive/manifest.json").read_text())
    for item in manifest["artifacts"]:
        if hashlib.sha256((ROOT/item["path"]).read_bytes()).hexdigest() != item["sha256"]:
            raise AssertionError(f"archive changed: {item['path']}")
    checks.append("All 18 archived student-artifact SHA-256 hashes match")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="reproduce all original finest meshes")
    parser.add_argument("--output", type=Path, default=Path("results/reproduction_results.json"))
    args = parser.parse_args()
    results = {}
    for name, module in [("rod", rod_self_weight), ("beam", cantilever_q4_q8), ("cylinder", lame_cylinder), ("cook", cooks_membrane), ("scf", stress_concentration)]:
        print(f"Starting {name}", flush=True)
        results[name] = module.run(args.full)
        save_result(name, results[name])
    checks = verify(results, args.full)
    for check in checks:
        print("PASS:", check, flush=True)
    record = dict(
        full_original_meshes=args.full,
        environment=dict(python=platform.python_version(), platform=platform.platform(),
                         packages={name: importlib.metadata.version(name) for name in ["numpy", "scipy", "matplotlib", "sympy"]},
                         openblas_num_threads=os.environ.get("OPENBLAS_NUM_THREADS")),
        checks=checks, results=results,
        qualifications=[
            "Archived notebook outputs are the numerical reference; the report's rod table is inconsistent with those outputs.",
            "The rod stress loading, beam comparison nu, and engineering-shear recovery have approved corrections.",
            "Euler-Bernoulli is an engineering beam reference, not an exact 2D elasticity solution.",
            "The SCF geometry and sampling limitations remain; this is not a validated circular-hole benchmark.",
        ],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item(), indent=2, allow_nan=False)+"\n", encoding="utf-8")
    print(f"Saved verification record: {args.output}")


if __name__ == "__main__":
    main()
