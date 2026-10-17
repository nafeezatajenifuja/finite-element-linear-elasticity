"""
# ======================================================================
#  CIVE 7336 – MP3 : Create_ID
#  Author : Nafeezat Ajenifuja
# ----------------------------------------------------------------------
#  PURPOSE :
#     Generates a global ID matrix assigning constraint numbers (negative)
#     and equation numbers (positive) to each degree of freedom (DOF) for
#     a finite element model.
# ======================================================================
"""

import numpy as np
from typing import Tuple

def Create_ID_Matrix(Constraints: np.ndarray, dofs_per_node: int, NumNodes: int) -> Tuple[np.ndarray, int]:

    """
    *********************************
    Create_ID_Matrix
    *********************************
    Generates a global ID matrix assigning constraint numbers (negative)
    and equation numbers (positive) to each degree of freedom (DOF) for
    a finite element model.

    *********************************
    Parameters
    *********************************
    Constraints : np.ndarray
        2D array of shape (NCons, 3) containing:
        [node_number, dof_number, constraint_value]
    dofs_per_node : int
        Number of degrees of freedom per node.
    NumNodes : int
        Total number of nodes in the model.

    *********************************
    Returns
    *********************************
    GlobalID : np.ndarray
        Matrix (NumNodes x dofs_per_node) containing:
            +ve IDs → free DOFs (equation numbers)
            -ve IDs → constrained DOFs (constraint numbers)
    Neqns : int
        Total number of free equations.

    *********************************
    RAISES:
    *********************************
    ValueError
        • If `Constraints` is not a 2D array with 3 columns.
        • If node or DOF indices in `Constraints` are out of range.
        • If `NumNodes` or `dofs_per_node` are non-positive.

    TypeError
        • If `dofs_per_node` or `NumNodes` are not integers.

    Notes
    -----
    * Negative values in GlobalID correspond to Dirichlet constraints.
    * The numbering of equations follows a row-major order.
    """

    # --- Input validation ---
    if not isinstance(Constraints, np.ndarray):
        raise ValueError("Constraints must be a NumPy array.")
    if Constraints.ndim != 2 or Constraints.shape[1] != 3:
        raise ValueError("Constraints must have 3 columns: [node, dof, value].")
    if not (isinstance(dofs_per_node, int) and isinstance(NumNodes, int)):
        raise TypeError("dofs_per_node and NumNodes must be integers.")
    if dofs_per_node <= 0 or NumNodes <= 0:
        raise ValueError("dofs_per_node and NumNodes must be positive integers.")

    # --- Initialize Global ID ---
    GlobalID = np.zeros((NumNodes, dofs_per_node), dtype=int)
    NCons = Constraints.shape[0]

    # --- Assign constraints ---
    if NCons > 0:
        nodes = Constraints[:, 0].astype(int)
        dofs  = Constraints[:, 1].astype(int)
        if np.any((nodes < 1) | (nodes > NumNodes)):
            raise ValueError(f"Constraint node indices out of range (1–{NumNodes}).")
        if np.any((dofs < 1) | (dofs > dofs_per_node)):
            raise ValueError(f"Constraint DOF indices out of range (1–{dofs_per_node}).")
        GlobalID[nodes - 1, dofs - 1] = -np.arange(1, NCons + 1)

    # --- Assign equation numbers ---
    flat = GlobalID.ravel()
    mask = flat == 0
    flat[mask] = np.arange(1, np.sum(mask) + 1)
    Neqns = np.sum(mask)

    return GlobalID, Neqns

