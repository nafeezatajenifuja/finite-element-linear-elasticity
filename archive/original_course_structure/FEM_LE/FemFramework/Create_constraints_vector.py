# ======================================================================
#  Create_ConstraintsVector_Vectorized.py
#  Author : Nafeezat Ajenifuja
#  Course : CIVE 7336 - MP3
# ----------------------------------------------------------------------
#  PURPOSE:
#     Vectorized implementation of Create_ConstraintsVector.
#     Constructs the global prescribed DOF vector (U_P)
#     directly using NumPy indexing (no explicit loops).
#
#  CONVENTION:
#     > 0 : Free DOF
#     < 0 : Prescribed DOF
# ======================================================================

import numpy as np

def Create_ConstraintsVector(Constraints: np.ndarray, GlobalID: np.ndarray) -> np.ndarray:
    """
    Vectorized construction of the global prescribed DOF vector (U_P).

    ------------------------------------------------------------------
    PARAMETERS
    ------------------------------------------------------------------
    Constraints : (NCons × 3) ndarray
        Each row = [node#, dof#, prescribed_value]
        Node# and DOF# are 1-based.
    GlobalID : (NumNodes × dofs_per_node) ndarray
        Global equation numbers (+ve free, -ve prescribed).

    ------------------------------------------------------------------
    RETURNS
    ------------------------------------------------------------------
    U_P : (NCons × 1) ndarray
        Prescribed displacement/temperature vector.
        Entry U_P[k] = prescribed value for |GlobalID| = k + 1.
    """

    # ----------------------------------------------------------
    # STEP 1: Basic setup
    # ----------------------------------------------------------
    NCons = Constraints.shape[0]
    if NCons == 0:
        return np.zeros((0, 1))

    # ----------------------------------------------------------
    # STEP 2: Extract node, dof, and prescribed values
    # ----------------------------------------------------------
    nodes = Constraints[:, 0].astype(int) - 1   # 0-based indices
    dofs  = Constraints[:, 1].astype(int) - 1
    vals  = Constraints[:, 2].astype(float)

    # ----------------------------------------------------------
    # STEP 3: Get corresponding constraint IDs from GlobalID
    # ----------------------------------------------------------
    constraint_ids = GlobalID[nodes, dofs]

    if np.any(constraint_ids > 0):
        raise ValueError("Error: One or more constraints correspond to free DOFs (positive IDs).")

    # ----------------------------------------------------------
    # STEP 4: Map constraint IDs to U_P positions
    # ----------------------------------------------------------
    NConsTotal = int(np.max(np.abs(GlobalID[GlobalID < 0]))) if np.any(GlobalID < 0) else 0
    U_P = np.zeros((NConsTotal, 1))

    idx = np.abs(constraint_ids) - 1     # convert to 0-based index positions
    U_P[idx, 0] = vals

    return U_P
