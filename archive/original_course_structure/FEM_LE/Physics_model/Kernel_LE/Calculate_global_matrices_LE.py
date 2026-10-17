# ======================================================================
#  CalculateGlobalMatrices.py
#  Author: Nafeezat Ajenifuja
# ----------------------------------------------------------------------
#  PURPOSE:
#  Calculates partitioned global stiffness matrices (K_FF, K_FP, R_F)
#  and loops over each element, computing local K and r, then assembling.
# ======================================================================

import numpy as np
from gauss_points import GaussPoints
from calculate_local_matrices_LE import CalculateLocalMatrices
from Assemble import Assemble

def CalculateGlobalMatricesLE(medium_set, load_type,
                            dofs_per_node, Connectivity,
                            EleType, NGPTS, Coord, GlobalID):
    """
    Calculates the global stiffness and load matrices for the
    Linearized Elasticity (LE) problem.

    ----------------------------------------------------------------------
    PARAMETERS
    ----------------------------------------------------------------------
    medium_set : dict
        Material property returning (lambda, mu) at coordinates x.
    load_type : dict
        Body-force structure (vector-valued) or a function returning f(x).
    dofs_per_node : int
        Degrees of freedom per node. For elasticity:
            dofs_per_node = dim  (e.g., 2 in 2D, 3 in 3D).
    Connectivity : (Nele × NodesPerEle) ndarray
        Element connectivity matrix (1-based node numbering, as in MATLAB).
    EleType : str
        Element type (e.g., 'Q4', 'T3').
    NGPTS : int
        Number of Gauss points per direction.
    Coord : (Nnode × dim) ndarray
        Nodal coordinate matrix.
    GlobalID : (Nnode × dofs_per_node) ndarray
        Global ID array labeling free (+) and prescribed (–) DOFs.

    ----------------------------------------------------------------------
    RETURNS
    ----------------------------------------------------------------------
    K_FF : ndarray
        Free–free portion of the global stiffness matrix.
    K_FP : ndarray
        Free–prescribed coupling stiffness matrix.
    R_F : ndarray
        Global load vector corresponding to free DOFs.
    """

    Nele, NodesPerEle = Connectivity.shape
    dim = Coord.shape[1]

    # ----------------------------------------------------------
    # 1. INITIALIZATION (sparse → dense arrays in Python)
    # ----------------------------------------------------------
    NEqns = int(np.max(GlobalID[GlobalID > 0])) if np.any(GlobalID > 0) else 0
    NCons = int(np.max(np.abs(GlobalID[GlobalID < 0]))) if np.any(GlobalID < 0) else 0

    K_FF = np.zeros((NEqns, NEqns))
    K_FP = np.zeros((NEqns, NCons))
    R_F  = np.zeros((NEqns, 1))

    # ----------------------------------------------------------
    # 2. GET GAUSS POINTS AND WEIGHTS
    # ----------------------------------------------------------
    r, w = GaussPoints(dim, EleType, NGPTS)
    r = np.asarray(r)
    if r.ndim == 1:
        r = r[:, None]
    w = np.asarray(w).reshape(-1)

    # ----------------------------------------------------------
    # 3. LOOP OVER ALL ELEMENTS
    # ----------------------------------------------------------
    for ele in range(Nele):
        # Nodes in this element (1-based indexing convert to 0-based)
        EleNodes = Connectivity[ele, :]
        node_ids = EleNodes.astype(int) - 1

        # Compute xCap (NodesPerEle x dim)
        xCap = Coord[node_ids, :]

        # --- Calculate local stiffness matrix and load vector ---
        Klocal, rlocal = CalculateLocalMatrices(
            medium_set,
            dofs_per_node,
            EleNodes,             # preserve 1-based form for consistency
            EleType,
            load_type,
            r, w,
            xCap
        )

        # --- Assemble into global partitions ---
        K_FF, K_FP, _, _, R_F = Assemble(
            dofs_per_node,
            EleNodes,
            GlobalID[node_ids, :],
            Klocal, rlocal,
            K_FF, K_FP,
            np.zeros((NCons, NEqns)),  # placeholder for K_PF 
            np.zeros((NCons, NCons)),  # placeholder for K_PP 
            R_F
        )

    return K_FF, K_FP, R_F

