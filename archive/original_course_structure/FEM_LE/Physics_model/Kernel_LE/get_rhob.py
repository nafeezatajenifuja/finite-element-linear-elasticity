# ==========================================================
#  get_rhob.py
#  Purpose: Provide the body-force vector f(x) for use in
#           the linear elasticity formulation.
#
#  Author : Nafeezat Ajenifuja
#  Course : CIVE 7336 - Finite Element Methods
# ==========================================================

"""
This module defines the body-force vector f(x) for linear elasticity.
It typically represents gravitational loading.
"""

import numpy as np


# ----------------------------------------------------------
#  get_rhob
# ----------------------------------------------------------
def Get_Rhob(load_type: dict, x=None) -> np.ndarray:
    """
    Returns the body-force vector f(x) for linear elasticity problems.

    --------------------------------------------------------------
    PARAMETERS
    --------------------------------------------------------------
    load_type : dict
        Keys:
            "case" : str
                Either "2D_gravity" or "3D_gravity"
            "rhob" : float
                Effective weight density, typically rho * g

    x : array_like, optional
        Spatial coordinate. Not used here, included only for compatability.

    --------------------------------------------------------------
    RETURNS
    --------------------------------------------------------------
    f : numpy.ndarray, shape (dim,1)
        Body-force vector.
        For 2D_gravity -> [[0.0],[-rhob]]
        For 3D_gravity -> [[0.0],[0.0],[-rhob]]
    """

    case = load_type.get("case", None)
    rhob = float(load_type.get("rhob", 0.0))
    
    # ------------------------------------------------------
    # 1D gravity loading (for 1D bar problems)
    # ------------------------------------------------------
    if case == "1D_gravity":
        f = np.zeros((1, 1))
        f[0, 0] = rhob
        return f
    
    # ------------------------------------------------------
    # 2D gravity loading
    # ------------------------------------------------------
    if case == "2D_gravity":
        f = np.zeros((2, 1))
        f[1, 0] = -rhob
        return f

    # ------------------------------------------------------
    # 3D gravity loading
    # ------------------------------------------------------
    if case == "3D_gravity":
        f = np.zeros((3, 1))
        f[2, 0] = -rhob
        return f

    # ------------------------------------------------------
    # Unknown case
    # ------------------------------------------------------
    raise ValueError(
        "Unknown load_type case '{}'. Expected '2D_gravity' or '3D_gravity'.".format(case)
    )
