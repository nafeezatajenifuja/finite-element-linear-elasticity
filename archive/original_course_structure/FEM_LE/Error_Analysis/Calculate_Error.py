# ==========================================================
#  Calculate_Error.py
#  Purpose: A posteriori error analysis for steady diffusion
#           Computes ||e||_L2 and |e|_H1 using Gauss quadrature
#  Notes   : Scalar diffusion (dofs_per_node = 1)
#  Author  : Nafeezat Ajenifuja
# ==========================================================

from typing import Callable, Tuple
import numpy as np

from gauss_points import GaussPoints
from Shape_functions import ShapeFunctions


def Calculate_Error(Connectivity: np.ndarray,
                    Coord: np.ndarray,
                    EleType: str,
                    NGPTS: int,
                    U: np.ndarray,
                    get_exact_solution: Callable[[np.ndarray],
                                                 Tuple[float, np.ndarray]],
                    verbose: bool = True) -> Tuple[float, float]:
    """
    Compute L2 error norm and H1 seminorm of the error.

    *********************************
    Parameters
    *********************************
    Connectivity : (Nele x nnode) int
        Element connectivity (1-based or 0-based OK; this routine expects 1-based
        for MATLAB parity; convert outside if needed).
    Coord : (NumNodes x dim) float
        Nodal coordinates.
    EleType : str
        Element type (e.g., "L2", "Q4", "T3", "Q8", "B8", ...).
    NGPTS : int
        Number of Gauss points (per direction for tensor-product rules).
    U : (NumNodes x 1) or (NumNodes,) float
        Total nodal solution (both free & prescribed DOFs).
    get_exact_solution : callable
        Function handle:  get_exact_solution(x) -> (u_exact, du_exact)
        where:
          - x : (dim,) physical coordinates
          - u_exact : float, exact solution at x
          - du_exact: (dim,) exact gradient at x
    verbose : bool
        Print status banners if True.

    *********************************
    Returns
    *********************************
    error_in_L2 : float
        L2 norm of error ||u_h - u||_L2.
    error_in_H1_seminorm : float
        H1 seminorm of error ||grad(u_h - u)||_L2.

    *********************************
    Assumptions
    *********************************
    - Scalar diffusion (dofs_per_node = 1).
    - Uses your existing ShapeFunctions and GaussPoints utilities.
    """

    if verbose:
        print("\n" + "=" * 65)
        print(" ERROR ANALYSIS")
        print("=" * 65)

    # -------------------------
    # Basic dimensions
    # -------------------------
    Nele = Connectivity.shape[0]
    nnode = Connectivity.shape[1]
    dim = Coord.shape[1]

    # Normalize U to (NumNodes,)
    U = np.asarray(U).reshape(-1)
    if U.shape[0] != Coord.shape[0]:
        raise ValueError("U length must equal number of nodes in Coord.")

    # -------------------------
    # Gauss points & weights
    # -------------------------
    r, w = GaussPoints(dim, EleType, NGPTS)  # r: (ngauss x dim), w: (ngauss,)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    ngauss = r.shape[0]

    if verbose:
        print(f"→ dim={dim}, nnode={nnode}, Nele={Nele}, NGPTS={NGPTS}, ngauss={ngauss}")

    # ---------------------------------------
    # Accumulators for squared error measures
    # ---------------------------------------
    err_L2_sq = 0.0
    err_H1_sq = 0.0

    # -------------------------
    # Element loop
    # -------------------------
    for ele in range(Nele):
        # Connectivity is assumed 1-based (MATLAB-style). If yours is 0-based, remove the -1.
        EleNodes = Connectivity[ele, :].astype(int)
        xCap = Coord[EleNodes - 1, :]              # (nnode x dim)
        uCap = U[EleNodes - 1]                     # (nnode,)

        # -------------
        # Gauss loop
        # -------------
        for g in range(ngauss):
            zeta = r[g, :]                     # (dim,)

            # Shape functions on reference element
            N, DN = ShapeFunctions(EleType, zeta)   # N: (nnode,), DN: (nnode x dim)

            # Geometry mapping
            J = xCap.T @ DN                    # (dim x dim)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Negative/zero detJ encountered in element {ele+1}.")
            invJ = np.linalg.inv(J)
            B = DN @ invJ                      # (nnode x dim)

            # Physical location, numerical field and gradient
            # x = sum_i N_i * x_i
            x = N @ xCap                        # (dim,)
            u_h = N @ uCap                      # scalar
            grad_u_h = B.T @ uCap               # (dim,)

            # Exact solution and gradient at x
            u_exact, du_exact = get_exact_solution(np.asarray(x))
            du_exact = np.asarray(du_exact).reshape(-1)
            if du_exact.shape[0] != dim:
                raise ValueError("get_exact_solution must return gradient of length 'dim'.")

            # Quadrature accumulation
            err_L2_sq += w[g] * (u_h - u_exact) ** 2 * detJ
            err_H1_sq += w[g] * np.dot(grad_u_h - du_exact, grad_u_h - du_exact) * detJ

    error_in_L2 = float(np.sqrt(err_L2_sq))
    error_in_H1_seminorm = float(np.sqrt(err_H1_sq))

    if verbose:
        print(f"→ ||e||_L2           = {error_in_L2:.6e}")
        print(f"→ |e|_H1 (seminorm) = {error_in_H1_seminorm:.6e}")
        print("=" * 65 + "\n")

    return error_in_L2, error_in_H1_seminorm

