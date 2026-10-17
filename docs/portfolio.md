# Repository metadata and application copy

## GitHub description

Python Galerkin FEM implementation for quasi-static linear elasticity with L2/Q4/Q8 elements, analytical checks, mesh studies, and stress recovery.

## Recommended topics

`finite-element-method`, `linear-elasticity`, `scientific-computing`, `computational-mechanics`, `numerical-methods`, `simulation`, `engineering-software`, `python`, `numpy`, `scipy`, `sympy`.

## One-line CV bullet

Implemented and verified a Python Galerkin FEM solver for linear elasticity with L2/Q4/Q8 elements, analytical comparisons, mesh-refinement studies, and stress recovery; documented numerical corrections and reproducible course-project provenance.

## Website project summary

This graduate course project implements a Python finite-element solver for quasi-static linear elasticity, including element integration, global assembly, displacement constraints, and stress recovery. Rod, cantilever, and modified Lamé examples support analytical and mesh-refinement checks, while Cook's membrane and a plate-with-hole example explore numerical sensitivity and limitations. The repository preserves the student implementation and separates course references from the maintained, reproducible code.

## Three strongest figures for recruiters

1. **Rod error convergence** — `figures/convergence/rod_error_convergence.png`. Shows quantitative verification and the distinction between L2 error and the H1 seminorm.
2. **Controlled Q4/Q8 cantilever comparison** — `figures/elasticity/cantilever_q4_q8.png`. Shows element/discretization sensitivity with a common material parameter, and labels beam theory as an engineering reference.
3. **Modified Lamé analytical comparison and error field** — `figures/elasticity/cylinder_analytical_comparison.png`. Connects an analytical field, a two-dimensional simulation, and spatial error visualization.

The rod stress-recovery plot is a useful additional figure for a methods-focused discussion. The exploratory SCF chart is not one of the recommended accuracy figures because its geometry and sampling limitations remain unresolved.
