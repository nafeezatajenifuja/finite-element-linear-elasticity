# ==========================================================
#  medium_set_LE.py
#  Purpose: Provide elastic material constants (Lame parameters)
#           for use in the 2D linear elasticity formulation.
#
#  Author : Nafeezat Ajenifuja
#  Course : CIVE 7336 - Finite Element Methods
# ==========================================================
# ----------------------------------------------------------
#  Get Elastic Moduli (Lame Parameters)
# ----------------------------------------------------------
def Get_Elastic_Moduli(medium_set, x=None):
    """
    Returns Lame parameters (lambda, mu) for linear elasticity.

    --------------------------------------------------------------
    PARAMETERS
    --------------------------------------------------------------
    medium_set : dict
        Dictionary describing the material. Must have the form:

            medium_set = {
                "type"  : "Lame_params",
                "lambda": <float>,   # First Lame parameter
                "mu"    : <float>    # Second Lame parameter (shear modulus)
            }

    x : ndarray (optional)
        Spatial coordinate 
    --------------------------------------------------------------
    RETURNS
    --------------------------------------------------------------
    lambda_val : float
        First Lame parameter

    mu_val : float
        Second Lame parameter (shear modulus)
    """

    mtype = medium_set.get("type", None)

    # ------------------------------------------------------
    # Validate material model type
    # ------------------------------------------------------
    if mtype == "Lame_params":
        lambda_val = float(medium_set["lambda"])
        mu_val     = float(medium_set["mu"])
        return lambda_val, mu_val

    # ------------------------------------------------------
    # Unknown or incorrectly specified material type
    # ------------------------------------------------------
    raise ValueError(
        "Unknown medium_set type '{}'. Expected 'Lame_params'.".format(mtype)
    )
