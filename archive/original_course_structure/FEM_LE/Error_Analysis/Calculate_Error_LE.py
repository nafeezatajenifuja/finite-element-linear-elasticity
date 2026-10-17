# ==========================================================
#  Calculate_Error_Elasticity.py
#  Purpose: A posteriori error analysis for 2D linear elasticity
#           Computes ||u_h - u_exact||_L2 using Gauss quadrature
#  Notes   : Vector field (dofs_per_node = dim = 2)
#  Author  : adapted from Nafeezat's scalar diffusion routine
# ==========================================================

from typing import Callable
import numpy as np

from gauss_points import GaussPoints
from Shape_functions import ShapeFunctions


def Calculate_Error_Elasticity(
    Connectivity: np.ndarray,
    Coord: np.ndarray,
    EleType: str,
    NGPTS: int,
    U: np.ndarray,
    get_exact_disp: Callable[[np.ndarray], np.ndarray],
    verbose: bool = True
) -> float:
    """
    Compute an L2-type error norm of the displacement field:

        || u_h - u_exact ||_L2

    using Gauss quadrature on each element.

    Parameters
    ----------
    Connectivity : (Nele x nnode) int
        Element connectivity (1-based, MATLAB style).
    Coord : (NumNodes x dim) float
        Nodal coordinates (x,y).
    EleType : str
        Element type ("Q4", "Q8", ...), must match ShapeFunctions.
    NGPTS : int
        Number of Gauss points per direction (2 for Q4, 3 for Q8, etc.).
    U : (NumNodes x 2) or (2*NumNodes,) float
        Nodal displacement solution (both DOFs per node).
    get_exact_disp : callable
        get_exact_disp(x) = u_exact
        where:
          - x : (dim,) physical coordinates
          - u_exact : (dim,) exact displacement at x (ux, uy)
    verbose : bool
        Print banner and error if True.

    Returns
    -------
    error_L2 : float
        L2 norm of displacement error.
    """

    if verbose:
        print("\n" + "=" * 65)
        print(" ELASTICITY ERROR ANALYSIS (DISPLACEMENT L2 NORM)")
        print("=" * 65)

    Nele = Connectivity.shape[0]
    nnode = Connectivity.shape[1]
    dim = Coord.shape[1]
    assert dim == 2, "This routine is written for 2D problems."

    # Normalize U to (NumNodes, 2)
    U = np.asarray(U)
    if U.ndim == 1:
        # assume [ux1, uy1, ux2, uy2, ...] ordering
        if U.size != 2 * Coord.shape[0]:
            raise ValueError("Flat U length must be 2*NumNodes.")
        U = U.reshape(-1, 2)
    else:
        if U.shape[0] != Coord.shape[0] or U.shape[1] != 2:
            raise ValueError("U must be (NumNodes x 2) or flat of length 2*NumNodes.")

    # Gauss points and weights
    r, w = GaussPoints(dim, EleType, NGPTS)  # r: (ngauss x dim), w: (ngauss,)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    ngauss = r.shape[0]

    if verbose:
        print(f"→ dim={dim}, nnode={nnode}, Nele={Nele}, NGPTS={NGPTS}, ngauss={ngauss}")

    err_L2_sq = 0.0

    # -----------------------------------
    # Element loop
    # -----------------------------------
    for ele in range(Nele):
        EleNodes = Connectivity[ele, :].astype(int)
        xCap = Coord[EleNodes - 1, :]     # (nnode x 2)
        uCap = U[EleNodes - 1, :]         # (nnode x 2)

        # Gauss loop
        for g in range(ngauss):
            zeta = r[g, :]                # (dim,)

            # Shape functions on reference element
            N, DN = ShapeFunctions(EleType, zeta)  # N: (nnode,), DN: (nnode x dim)

            # Geometry mapping
            J = xCap.T @ DN               # (2 x 2)
            detJ = np.linalg.det(J)
            if detJ <= 0:
                raise ValueError(f"Negative/zero detJ in element {ele+1} (detJ={detJ}).")

            # Physical location
            x = N @ xCap                  # (2,)

            # FEM displacement at this Gauss point
            # N: (nnode,), uCap: (nnode x 2) -> (2,)
            u_h = N @ uCap

            # Exact displacement at this point
            u_ex = np.asarray(get_exact_disp(np.asarray(x))).reshape(-1)
            if u_ex.shape[0] != dim:
                raise ValueError("get_exact_disp must return displacement of length 2.")

            # Error vector
            e = u_h - u_ex
            err_L2_sq += w[g] * np.dot(e, e) * detJ

    error_L2 = float(np.sqrt(err_L2_sq))

    if verbose:
        print(f"||u_h - u_exact||_L2 = {error_L2:.6e}")
        print("=" * 65 + "\n")

    return error_L2


