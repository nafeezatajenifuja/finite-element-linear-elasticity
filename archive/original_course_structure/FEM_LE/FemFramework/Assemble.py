# ======================================================================
#  Assemble.py
#  Author : Nafeezat Ajenifuja
#  Course : CIVE 7336 - MP3
# ----------------------------------------------------------------------
#  PURPOSE:
#     Vectorized assembly of local element matrices and vectors into
#     the correct global partitions (K_FF, K_FP, K_PF, K_PP, R_F).
#
#  CONVENTION:
#     > 0 : Free DOF (active)
#     < 0 : Prescribed DOF (constrained)
# ======================================================================

import numpy as np

def Assemble(dofs_per_node,
                        EleNodes, GlobalID, 
                        Klocal, Rlocal,
                        K_FF, K_FP, K_PF, K_PP,
                        R_F):
    """
    Vectorized Assemble (no explicit for loops).

    Parameters
    ----------
    dofs_per_node : int
        Degrees of freedom per node.
    GlobalID : (NodesPerEle, dofs_per_node) ndarray
        Global equation numbers (+ve free, -ve prescribed).
    EleNodes : (NodesPerEle,) ndarray
        Element node indices (for size reference).
    Klocal : (ndofs_e, ndofs_e) ndarray
        Local element stiffness matrix.
    Rlocal : (ndofs_e, 1) ndarray
        Local element load vector.
    K_FF, K_FP, K_PF, K_PP : ndarray
        Global stiffness partitions.
    R_F : (n_free, 1) ndarray
        Global right-hand-side vector for free DOFs.

    Returns
    -------
    K_FF, K_FP, K_PF, K_PP, R_F : updated arrays
    """

    # ---------------- Build flattened vector of global IDs ----------------
    NodesPerEle = len(EleNodes)
    ndofs_e = NodesPerEle * dofs_per_node
    v_vector = GlobalID.reshape(ndofs_e, 1)

    # ---------------- Identify categories ----------------
    row_ids = v_vector.repeat(ndofs_e, axis=1)
    col_ids = v_vector.T.repeat(ndofs_e, axis=0)

    mask_FF = (row_ids > 0) & (col_ids > 0)
    mask_FP = (row_ids > 0) & (col_ids < 0)
    mask_PF = (row_ids < 0) & (col_ids > 0)
    mask_PP = (row_ids < 0) & (col_ids < 0)

    # ---------------- Convert to 0-based integer indices ----------------
    row_FF = (row_ids[mask_FF].astype(int) - 1)
    col_FF = (col_ids[mask_FF].astype(int) - 1)

    row_FP = (row_ids[mask_FP].astype(int) - 1)
    col_FP = (np.abs(col_ids[mask_FP]).astype(int) - 1)

    row_PF = (np.abs(row_ids[mask_PF]).astype(int) - 1)
    col_PF = (col_ids[mask_PF].astype(int) - 1)

    row_PP = (np.abs(row_ids[mask_PP]).astype(int) - 1)
    col_PP = (np.abs(col_ids[mask_PP]).astype(int) - 1)

    # ---------------- Accumulate Klocal entries ----------------
    np.add.at(K_FF, (row_FF, col_FF), Klocal[mask_FF])
    np.add.at(K_FP, (row_FP, col_FP), Klocal[mask_FP])
    np.add.at(K_PF, (row_PF, col_PF), Klocal[mask_PF])
    np.add.at(K_PP, (row_PP, col_PP), Klocal[mask_PP])

    # ---------------- Assemble load vector ----------------
    mask_R = v_vector[:, 0] > 0
    row_R = (v_vector[mask_R, 0].astype(int) - 1)
    val_R = Rlocal[mask_R, 0]
    np.add.at(R_F[:, 0], row_R, val_R)

    return K_FF, K_FP, K_PF, K_PP, R_F


