# ======================================================================
#  PostProcessing.py
#  Author : Nafeezat Ajenifuja
#  Course : CIVE 7336 - MP3
# ----------------------------------------------------------------------
#  PURPOSE:
#     Vectorized reconstruction of the nodal solution matrix (U)
#     from free (U_F) and prescribed (U_P) nodal vectors using GlobalID.
#
#  CONVENTION:
#     > 0 : Free DOF (active)
#     < 0 : Prescribed DOF (constrained)
#
# ======================================================================

import numpy as np

def PostProcessing(GlobalID: np.ndarray,
                   U_F: np.ndarray,
                   U_P: np.ndarray) -> np.ndarray:
    """
    Vectorized reconstruction of total nodal field matrix U.

    ------------------------------------------------------------------
    PARAMETERS
    ------------------------------------------------------------------
    GlobalID : (NumNodes × dofs_per_node) ndarray
        Global equation numbers (+ve free, -ve prescribed).
    U_F : (NEqns × 1) ndarray
        Vector of free nodal values.
    U_P : (NCons × 1) ndarray
        Vector of prescribed nodal values.

    ------------------------------------------------------------------
    RETURNS
    ------------------------------------------------------------------
    U : (NumNodes × dofs_per_node) ndarray
        Matrix of total nodal values of the field variable.
    """

    # ----------------------------------------------------------
    # STEP 1: Initialize the result matrix
    # ----------------------------------------------------------
    NumNodes, dofs_per_node = GlobalID.shape
    U = np.zeros((NumNodes, dofs_per_node))

    # ----------------------------------------------------------
    # STEP 2: Handle Free DOFs
    # ----------------------------------------------------------
    mask_F = GlobalID > 0
    idx_F = GlobalID[mask_F].astype(int) - 1        # convert to 0-based
    U[mask_F] = U_F[idx_F, 0]

    # ----------------------------------------------------------
    # STEP 3: Handle Prescribed DOFs
    # ----------------------------------------------------------
    mask_P = GlobalID < 0
    idx_P = np.abs(GlobalID[mask_P]).astype(int) - 1
    U[mask_P] = U_P[idx_P, 0]

    return U
