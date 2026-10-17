# Numerical methodology and changes from the course submission

## Preserved formulation

The maintained solver implements the student's Galerkin discretization of quasi-static, isotropic, linearized elasticity. With G containing the physical gradients of the scalar shape functions, the element stiffness has the same three contributions as `calculate_local_matrices_LE.py`: the lambda outer product of vec(G.T), a mu Kronecker term, and a mu transpose-permutation term. Shape gradients use the original Jacobian convention. The local body-force integral is unchanged.

Free equations are numbered positively and prescribed equations negatively, both starting at one. Assembly retains the free/free and free/prescribed partitions. The solve uses NumPy's dense direct linear solver, and prescribed displacements enter as `f_F - K_FP @ u_P`. Original reaction partitions were allocated and discarded; the maintained implementation omits those unused allocations and does not claim to report reactions.

The L2, Q4, and Q8 interpolation polynomials and reference-node order are unchanged. Symbolic derivatives are generated once rather than recreated at every integration point. Quadrature remains two-point by default for L2, 2x2 for Q4, and fixed 3x3 for Q8. The two-point rod L2 error calculation is the original quadrature estimator; squared quadratic displacement errors are not integrated exactly by that rule. Its observed rate remains a useful refinement check.

## Original-to-current mapping

| Student source | Maintained implementation |
| --- | --- |
| `Driver_LE.py` | `src/fem/solver.py` |
| `Calculate_global_matrices_LE.py`, `Assemble.py` | `src/fem/assembly.py` |
| `calculate_local_matrices_LE.py` | `src/fem/elements.py` |
| `Shape_functions.py`, `gauss_points.py` | L2/Q4/Q8 subset in `elements.py`, `quadrature.py` |
| `Get_Elastic_Moduli.py`, `get_rhob.py`, `Vector.py`, `TMatrix.py` | `elasticity.py`, the body-force interface, and explicit tensor/vector operations |
| `create_id.py`, `Create_constraints_vector.py` | `boundary_conditions.py` |
| `PostProcessing.py`, `Calculate_Error.py`, `Calculate_Error_LE.py` | `postprocessing.py` |
| `Stress_Recovery.py`; final-notebook 2D recovery functions | `stress_recovery.py` |
| Final-notebook mesh and experiment cells | `meshes.py`, `examples/` |

## Approved numerical corrections

The author reviewed and approved these three corrections before they were made. The original files and outputs are archived unchanged.

1. **Engineering shear.** In zero-based notebook cell 44, the B matrix constructs `gamma_xy = u_x,y + u_y,x`, while recovery multiplies it by `2*mu`. The maintained relation is `sigma_xy = mu*gamma_xy`, consistent with the original solver and strain-energy calculation. A pure-shear test independently verifies the factor.
2. **Rod body force.** Cell 11 uses `rho*g*A` as the body-force density, although the solver and earlier rod cells use `rho*g` after constant cross-sectional area cancels from both sides of the one-dimensional equation. The maintained stress example uses `rho*g`, making it the same physical problem as the displacement/refinement example. Its stress magnitudes therefore differ from the archived stress plot.
3. **Controlled beam comparison.** Cell 17 hardcodes nu=0.3 for the one-layer Q4 curve, while the other curves inherit nu=0.1 from cell 15. The maintained comparison explicitly uses nu=0.1 for all three curves, retaining the unchanged Q8 sequence as a numerical regression reference. The old comparison is not presented as a controlled element benchmark.

## Additional diagnosed limitations

- **Invalid coarse SCF restraints:** in the archived 41x21 nodal-grid demo, radial-band sampling can place every sampled node at the same x coordinate. The leftmost/rightmost selector then picks the same node twice. Only two distinct displacement constraints remain, leaving rigid rotation unconstrained. The maintained examples use the original 81x41 study grid, where the selected restraint nodes are distinct with different x coordinates. Duplicate constraints are rejected. No new boundary conditions were silently substituted for the coarse demo.
- **SCF geometry/sampling:** the code removes any rectangular element containing a node inside the nominal circular hole. This produces a stepwise opening, not a body-fitted circular boundary. The legacy `HoleNodes` set uses a radial band based on the nearest element-center radius. It is not guaranteed to select the actual free boundary, and sampled radii differ from the nominal radius. These choices are retained to reproduce the original study. Reproducing its numbers does not validate circular-hole SCFs or the report's accuracy claims.
- **Rod report discrepancy:** the report's printed error table differs from the canonical final-notebook output. The reproducibility checks use the notebook outputs at their saved precision, rather than claiming to reproduce both conflicting tables.
- **Cylinder description:** the executed cylinder is displacement-driven: the inner boundary is fixed and the outer boundary has prescribed radial displacement. It does not apply an internal/external pressure load. Its effective Lamé constants are plane stress even where some original comments/report passages say plane strain.
- **Cook interpretation:** the model is a standard displacement Q4 calculation with substantial mesh and compressibility sensitivity. There is no independent reference or stabilized formulation here that isolates physical material sensitivity from discretization locking or certifies a converged Cook solution.
- **Original 1D recovery scope:** the supplied-reference-style Python recovery module is specialized to L2 chains despite its generic-looking interface. The final notebook provides separate Q4 recovery functions. The maintained API makes those scopes explicit; it does not claim general 3D stress recovery.
- **Plot geometry:** original unconstrained Delaunay triangulations can fill the annular hole. The new cylinder field plot uses actual element polygons, with no change to the numerical solution.
- **Compatibility and input checks:** the maintained code extracts scalar values explicitly rather than calling `float` on length-one arrays, which fails with current NumPy. It rejects inverted elements, duplicate constraints, fractional node IDs, and invalid dimensions. These checks do not change valid experiment inputs.

## Verification and reproduction

`tests/` includes independent mathematical checks and comparisons against the archived student implementation. `scripts/reproduce.py --full` runs the complete original mesh sequences, compares supported quantities with the saved notebook outputs, and checks every archived artifact hash. Three corrected cases are described separately; exact equality to their old outputs is not an objective.

The full cylinder sequence is Nr = [4, 8, 12, 16, 24, 40] and Ntheta = 4*Nr. The full Cook sequence is [2, 4, 8, 12, 16, 32, 64] elements per direction, with its Poisson sweep on 64x64. The beam uses [2, 4, 8, 16, 32, 64] elements along its length; the rod uses [5, 10, 20, 50, 100, 200, 500]. Both reproduction modes use the original 81x41 nodal-grid SCF sweep. The default mode truncates only the cylinder and Cook sequences for convenience and labels that choice in its output.

Numerical results, comparisons, environment versions, and qualifications are stored in `docs/reproduction_results.json` for the release validation. Deterministic inputs are used throughout the examples, and randomized test points use seed 7336. Numerical equality across BLAS implementations and platforms is not promised. For the recorded single-thread BLAS setting, set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before starting Python.

## Replacing supplied inputs

No course input file is needed by this repository. `fem.meshes` constructs the examples procedurally from scalar geometric parameters. For another problem, provide:

- `coordinates`: finite floating-point array `(n_nodes, dim)`.
- `connectivity`: integer array `(n_elements, n_local)` with one-based node IDs and positive Jacobians.
- `element`: `L2`, `Q4`, or `Q8`, in the documented reference-node order.
- `constraints`: rows `[one_based_node, one_based_component, prescribed_displacement]`; each DOF appears once.
- `point_loads`: rows `[one_based_node, one_based_component, force]`; repeated free-DOF loads accumulate.
- `body_force`: a vector of length dim or a callable returning that vector at a physical point.
- `material`: effective Lamé constants, or E/nu converted with an explicit plane-stress/plane-strain choice.

For a supplied mesh whose redistribution rights are unclear, retain it privately and convert it locally to these arrays, or generate your own mesh. No importer requires the instructor's MATLAB scripts, quadrature tables, or Gmsh file. The straight-edge load helper is restricted to uniformly spaced straight edges; it is not a general traction-boundary integration routine.

## Scope of claims

The work demonstrates an implementation of established methods. It does not establish new FEM theory, broad superiority of Q8, a validated stress-concentration design chart, production suitability, or a runtime speedup. The measured convergence statements concern the explicit examples and quadrature estimators only. The course permission statement is based on the author's report of verbal permission and covers the student's implementation; it is not treated as permission to redistribute course materials.
