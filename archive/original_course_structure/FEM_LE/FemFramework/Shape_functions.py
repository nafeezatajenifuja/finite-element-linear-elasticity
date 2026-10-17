"""
# ======================================================================
#  CIVE 7336 – MP3 : Shape_functions
#  Author : Nafeezat Ajenifuja
# ----------------------------------------------------------------------
#  PURPOSE :
#      Computing element shape functions and their derivatives
#      at a given reference coordinate.
# ======================================================================
"""

import numpy as np
from typing import Tuple, Union, Sequence
import sympy as sp

"""## Shape functions

********************************

### #1D

********************************

==============

#### L2

==============
"""

def Shape_L2(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D shape functions and derivatives symbolically
    for a 2-node linear element.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        The reference coordinate array containing one zeta value.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (2,) array of shape-function values evaluated at zata.
    DN : np.ndarray
        (2,1) array of derivatives dN/d(zeta) evaluated at zeta.
    """

    # --- Validate input ---
    if not isinstance(zeta, np.ndarray):
        raise TypeError("Input zeta must be a NumPy array.")
    if zeta.size != 1:
        raise ValueError("Shape_L2 expects a single reference coordinate [zeta].")

    # --- Define symbolic variable and shape functions ---
    zet = sp.Symbol("zet")
    N_sym = [(1 - zet) / 2, (1 + zet) / 2]
    DN_sym = [sp.diff(Ni, zet) for Ni in N_sym]

    # --- Evaluate numerically at provided zeta ---
    xi_val = float(zeta[0])
    N = np.array([float(Ni.subs(zet, xi_val)) for Ni in N_sym], dtype=float)
    DN_vals = np.array([float(DNi.subs(zet, xi_val)) for DNi in DN_sym], dtype=float).reshape(-1, 1)

    # Transpose to make DN shape (2, 1)
    DN = DN_vals.reshape(-1,1)   # (2,1)

    return N, DN

"""==============

#### L3

==============
"""

def Shape_L3(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute 1D shape functions and derivatives symbolically
    for a 3-node linear element.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        The reference coordinate array containing one zeta value.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (3,) array of shape-function values evaluated at zeta.
    DN : np.ndarray
        (3,1) array of derivatives dN/d(zeta) evaluated at zeta.

    """

    # --- Define symbolic variable and shape functions ---
    zet = sp.Symbol("zet")
    N_sym = [
        0.5 * zet * (zet - 1),
        0.5 * zet * (zet + 1),
        1 - zet**2

    ]
    DN_sym = [sp.diff(Ni, zet) for Ni in N_sym]

    # --- Evaluate numerically at provided zeta ---
    xi_val = float(zeta[0])
    N = np.array([float(Ni.subs(zet, xi_val)) for Ni in N_sym], dtype=float)
    DN_vals = np.array([float(DNi.subs(zet, xi_val)) for DNi in DN_sym], dtype=float).reshape(-1, 1)

    # Transpose to make DN shape (3, 1)
    DN = DN_vals.reshape(-1, 1)   # (3,1)

    return N, DN

"""==============

#### 1D Dispatcher

==============
"""

def ShapeFunctions1D(EleType: str, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher for 1D element shape functions.

    *********************************
    Parameters
    *********************************
    EleType : str
        Element type ('L2' or 'L3').
    zeta : np.ndarray
        Reference coordinate [zeta] where shape functions are evaluated.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        Shape-function values at zeta.
    DN : np.ndarray
        Derivatives of shape functions at zeta.

    *********************************
    Raises
    *********************************
    ValueError
        If the element type is unsupported or zeta invalid.
    """

    # --- Validate input ---
    if not isinstance(zeta, np.ndarray):
        raise TypeError("Input zeta must be a NumPy array.")
    if zeta.size != 1:
        raise ValueError("zeta must contain exactly one reference coordinate [ξ].")

    if EleType == "L2":
        return Shape_L2(zeta)
    elif EleType == "L3":
        return Shape_L3(zeta)
    else:
        raise ValueError(f"Unsupported 1D element type: {EleType}")

"""********************************

### #2D

********************************

==============

#### T3

==============
"""

def Shape_T3(zeta: np.ndarray):
    """
    Computes shape functions and their derivatives for a
    3-node triangular element (T3) in the reference domain.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        1D array with two components [zet1, zet2].
        Must satisfy: zet1 >= 0, zet2 >= 0, zet1 + zet2 <= 1.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        Shape function values [N1, N2, N3] at the given (zet1, zet2).
    DN : np.ndarray
        Derivative matrix of size (3 x 2):
            columns → [∂/∂zet1, ∂/∂zet2]
            rows    → nodes 1..3

    *********************************
    Raises
    *********************************
    TypeError
        If zeta is not a NumPy array.
    ValueError
        If zeta is not length 2 or lies outside the reference triangle.
    """
    zet1, zet2 = zeta
    # --- Symbolic setup ---
    zet1_s, zet2_s = sp.symbols("zet1 zet2")

    # --- Define shape functions (multiplicative form) ---
    L1 = 1 - zet1_s - zet2_s
    L2 = zet1_s
    L3 = zet2_s

    N_expr = [L1, L2, L3]

    # --- Symbolic derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],
        [sp.diff(Ni, zet2_s) for Ni in N_expr]
    ]

    # --- Substitute actual numerical values ---
    subs_dict = {zet1_s: zet1, zet2_s: zet2}
    N = np.array([float(Ni.subs(subs_dict)) for Ni in N_expr], dtype=float)
    DN_temp = np.array([[float(d.subs(subs_dict)) for d in row] for row in DN_expr], dtype=float)

    # Convert to (3 x 2): rows = nodes, cols = derivatives
    DN = DN_temp.T

    return N, DN

"""==============

#### T6

==============
"""

def Shape_T6(zeta: np.ndarray):
    """
    Computes shape functions and their derivatives for a 6-node
    triangular element (T6) in the reference domain.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        1D array with two components [zet1, zet2].
        Must satisfy: zet1 >= 0, zet2 >= 0, and zet1 + zet2 <= 1.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        Shape function values [N1, ..., N6] at the given (zet1, zet2).
    DN : np.ndarray
        Derivatives matrix (6 x 2)

    *********************************
    Raises
    *********************************
    TypeError
        If `zeta` is not a NumPy array.
    ValueError
        If `zeta` does not have exactly two coordinates or lies
        outside the reference triangle.
    """
    zet1, zet2 = zeta
    # --- Define symbolic variables ---
    zet1_s, zet2_s = sp.symbols("zet1 zet2")

    # --- Define shape functions in multiplicative (factored) form ---
    L1 = 1 - zet1_s - zet2_s
    L2 = zet1_s
    L3 = zet2_s

    N1 = L1 * (2*L1 - 1)
    N2 = L2 * (2*L2 - 1)
    N3 = L3 * (2*L3 - 1)
    N4 = 4 * L1 * L2
    N5 = 4 * L2 * L3
    N6 = 4 * L3 * L1

    N_expr = [N1, N2, N3, N4, N5, N6]

    # --- Symbolic derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],
        [sp.diff(Ni, zet2_s) for Ni in N_expr]
    ]

     # --- Substitute numerical values ---
    subs_dict = {zet1_s: zet1, zet2_s: zet2}
    N = np.array([float(Ni.subs(subs_dict)) for Ni in N_expr], dtype=float)
    DN_temp = np.array([[float(d.subs(subs_dict)) for d in row] for row in DN_expr], dtype=float)

    # Transpose to make DN shape (6, 2)
    DN = DN_temp.T
    return N, DN

"""==============

#### Q4

==============
"""

def Shape_Q4(zeta: np.ndarray):
    """
    Computes shape functions and their derivatives for a
    4-node bilinear quadrilateral (Q4) element.

    The formulation reuses the 1D L2 shape functions along
    each coordinate direction to build the 2D tensor-product
    interpolation functions.

    Parameters
    ----------
    zeta : np.ndarray
        1D array with two components [zet1, zet2].
        Each component must lie within [-1, 1].

    Returns
    -------
    N : np.ndarray
        Shape function values [N1, N2, N3, N4] at the given (zet1, zet2).

    DN : np.ndarray
        Derivative matrix of size (4 x 2):
            columns → [∂/∂zet1, ∂/∂zet2]
            rows    → nodes 1..4

    Raises
    ------
    TypeError
        If zeta is not a NumPy array.
    ValueError
        If zeta does not have exactly two coordinates or lies outside [-1, 1].
    """
    zet1, zet2 = zeta
    # --- Get 1D shape functions and derivatives ---
    N_zet1, dN_zet1 = Shape_L2(np.array([zet1]))  # Along ζ1
    N_zet2, dN_zet2 = Shape_L2(np.array([zet2]))  # Along ζ2

    # --- 2D tensor product formulation ---
    # Shape functions
    N = np.array([
        N_zet1[0] * N_zet2[0],  # Node 1
        N_zet1[1] * N_zet2[0],  # Node 2
        N_zet1[1] * N_zet2[1],  # Node 3
        N_zet1[0] * N_zet2[1]   # Node 4
    ], dtype=float)

    # Derivatives (chain rule)
    DN = np.zeros((4, 2))
    DN[:, 0] = [
        dN_zet1[0, 0] * N_zet2[0],
        dN_zet1[1, 0] * N_zet2[0],
        dN_zet1[1, 0] * N_zet2[1],
        dN_zet1[0, 0] * N_zet2[1]
    ]

    DN[:, 1] = [
        N_zet1[0] * dN_zet2[0, 0],
        N_zet1[1] * dN_zet2[0, 0],
        N_zet1[1] * dN_zet2[1, 0],
        N_zet1[0] * dN_zet2[1, 0]
    ]
    # After computing DN
    DN[np.isclose(DN, 0.0, atol=1e-12)] = 0.0

    return N, DN

"""==============

#### Q8 (Serendipity element)

==============
"""

def Shape_Q8(zeta: np.ndarray):
    """
    Computes shape functions and derivatives for an
    8-node quadratic quadrilateral (Q8) element using
    tensor products of 1D L3 shape functions.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        2-component array [zet1, zet2] in [-1, 1].

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (8,) array of shape-function values at (zet1, zet2).

    DN : np.ndarray
        (8, 2) array of derivatives of shape functions.
        Columns: [∂/∂zet1, ∂/∂zet2]

    *********************************
    Raises
    *********************************
    TypeError
        If input is not a NumPy array.
    ValueError
        If zeta does not have 2 components or lies outside [-1, 1].
    """
    zet1, zet2 = zeta
     # --- Symbolic variables ---
    zet1_s, zet2_s = sp.symbols("zet1 zet2")

    # --- Shape function expressions (page 23 of slides) ---
    N_expr = [
        0.25 * (1 - zet1_s) * (1 - zet2_s) * (-zet1_s - zet2_s - 1),
        0.25 * (1 + zet1_s) * (1 - zet2_s) * ( zet1_s - zet2_s - 1),
        0.25 * (1 + zet1_s) * (1 + zet2_s) * ( zet1_s + zet2_s - 1),
        0.25 * (1 - zet1_s) * (1 + zet2_s) * (-zet1_s + zet2_s - 1),
        0.5  * (1 - zet1_s**2) * (1 - zet2_s),
        0.5  * (1 + zet1_s) * (1 - zet2_s**2),
        0.5  * (1 - zet1_s**2) * (1 + zet2_s),
        0.5  * (1 - zet1_s) * (1 - zet2_s**2)
    ]

    # --- Symbolic derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],
        [sp.diff(Ni, zet2_s) for Ni in N_expr]
    ]

    # --- Substitute numerical values ---
    subs = {zet1_s: zet1, zet2_s: zet2}
    N = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN_temp = np.array([[float(d.subs(subs)) for d in row] for row in DN_expr], dtype=float)
    DN = DN_temp.T  # shape (8×2): rows = nodes, cols = [∂/∂zet1, ∂/∂zet2]

    return N, DN

"""==============

#### Q9

==============
"""

def Shape_Q9(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shape functions and derivatives for a
    9-node quadratic quadrilateral (Q9) element.

    Based directly on the multiplicative symbolic form
    shown in the slides (page 24).

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        2-component array [zet1, zet2] in [-1, 1].

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (9,) array of shape-function values.
    DN : np.ndarray
        (9,2) array of derivatives.
    """
    zet1, zet2 = zeta

    # --- Define symbolic variables ---
    zet1_s, zet2_s = sp.symbols("zet1 zet2")

    # --- Define 9 shape functions (direct from slides) ---
    N_expr = [
        +0.25 * zet1_s * zet2_s * (1 - zet1_s) * (1 - zet2_s),   # Node 1
        -0.25 * zet1_s * zet2_s * (1 + zet1_s) * (1 - zet2_s),   # Node 2
        +0.25 * zet1_s * zet2_s * (1 + zet1_s) * (1 + zet2_s),   # Node 3
        -0.25 * zet1_s * zet2_s * (1 - zet1_s) * (1 + zet2_s),   # Node 4
        -0.5  * (1 - zet1_s**2) * zet2_s * (1 - zet2_s),         # Node 5
        +0.5  * zet1_s * (1 + zet1_s) * (1 - zet2_s**2),         # Node 6
        +0.5  * (1 - zet1_s**2) * zet2_s * (1 + zet2_s),         # Node 7
        -0.5  * zet1_s * (1 - zet1_s) * (1 - zet2_s**2),         # Node 8
        (1 - zet1_s**2) * (1 - zet2_s**2)                        # Node 9 (center)
    ]

    # --- Symbolic derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],
        [sp.diff(Ni, zet2_s) for Ni in N_expr]
    ]

    # --- Substitute numerical values ---
    subs = {zet1_s: zet1, zet2_s: zet2}
    N = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN_temp = np.array([[float(d.subs(subs)) for d in row] for row in DN_expr], dtype=float)
    DN = DN_temp.T  # (9 × 2): rows = nodes, columns

    return N, DN

"""==============

#### 2D Dispatcher

==============
"""

def ShapeFunctions2D(EleType: str, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher for 2D element shape functions with centralized input validation.

    Routes the element type string (e.g., 'T3', 'Q9') to the appropriate
    shape-function routine and validates domain constraints according to
    element type (triangular or quadrilateral).

    *********************************
    Parameters
    *********************************
    EleType : str
        Element type:
        'T3', 'T6'  → Triangular elements
        'Q4', 'Q8', 'Q9' → Quadrilateral elements

    zeta : np.ndarray
        2-component reference coordinate array [zet1, zet2].

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (n_nodes,) array of shape-function values.

    DN : np.ndarray
        (n_nodes, 2) array of shape-function derivatives.

    *********************************
    Raises
    *********************************
    TypeError
        If zeta is not a NumPy array.
    ValueError
        If zeta violates domain constraints or EleType is invalid.
    """

    # --- Input type validation ---
    if not isinstance(zeta, np.ndarray):
        raise TypeError("Input zeta must be a NumPy array.")
    if zeta.size != 2:
        raise ValueError("ShapeFunctions2D expects two reference coordinates [zet1, zet2].")

    zet1, zet2 = zeta

    # --- Domain constraint validation (moved here) ---
    if EleType in ["T3", "T6"]:
        # Triangular domain: ζ1 ≥ 0, ζ2 ≥ 0, ζ1 + ζ2 ≤ 1
        if zet1 < 0 or zet2 < 0 or (zet1 + zet2) > 1:
            raise ValueError(
                f"Invalid (zet1, zet2)=({zet1}, {zet2}) for {EleType}: "
                "must satisfy zet1 >= 0, zet2 >= 0, and zet1 + zet2 <= 1."
            )

    elif EleType in ["Q4", "Q8", "Q9"]:
        # Quadrilateral domain: ζ1, ζ2 ∈ [-1, 1]
        if not (-1 <= zet1 <= 1 and -1 <= zet2 <= 1):
            raise ValueError(
                f"Invalid (zet1, zet2)=({zet1}, {zet2}) for {EleType}: "
                "both coordinates must lie in [-1, 1]."
            )

    else:
        raise ValueError(
            f"Unsupported 2D element type '{EleType}'. "
            "Supported types are ['T3', 'T6', 'Q4', 'Q8', 'Q9']."
        )

    # --- Dispatch to correct element ---
    if EleType == "T3":
        N, DN = Shape_T3(zeta)
    elif EleType == "T6":
        N, DN = Shape_T6(zeta)
    elif EleType == "Q4":
        N, DN = Shape_Q4(zeta)
    elif EleType == "Q8":
        N, DN = Shape_Q8(zeta)
    elif EleType == "Q9":
        N, DN = Shape_Q9(zeta)

    return N, DN

"""********************************

### #3D

********************************

==============

#### B8

==============
"""

def Shape_Brick8(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shape functions and derivatives for an 8-node
    trilinear hexahedral (Brick8 / Hex8) element.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        3-component array [zet1, zet2, zet3] in [−1, 1].

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (8,) array of shape-function values evaluated at (zet1, zet2, zet3).
    DN : np.ndarray
        (8, 3) array of derivatives.
    """

    zet1, zet2, zet3 = zeta

    # --- Symbolic variables ---
    zet1_s, zet2_s, zet3_s = sp.symbols("zet1 zet2 zet3")

    # --- Shape function definitions (direct trilinear form) ---
    N_expr = [
        1/8 * (1 - zet1_s) * (1 - zet2_s) * (1 - zet3_s),  # Node 1
        1/8 * (1 + zet1_s) * (1 - zet2_s) * (1 - zet3_s),  # Node 2
        1/8 * (1 + zet1_s) * (1 + zet2_s) * (1 - zet3_s),  # Node 3
        1/8 * (1 - zet1_s) * (1 + zet2_s) * (1 - zet3_s),  # Node 4
        1/8 * (1 - zet1_s) * (1 - zet2_s) * (1 + zet3_s),  # Node 5
        1/8 * (1 + zet1_s) * (1 - zet2_s) * (1 + zet3_s),  # Node 6
        1/8 * (1 + zet1_s) * (1 + zet2_s) * (1 + zet3_s),  # Node 7
        1/8 * (1 - zet1_s) * (1 + zet2_s) * (1 + zet3_s)   # Node 8
    ]

    # --- Derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],
        [sp.diff(Ni, zet2_s) for Ni in N_expr],
        [sp.diff(Ni, zet3_s) for Ni in N_expr]
    ]

    # --- Evaluate numerically ---
    subs = {zet1_s: zet1, zet2_s: zet2, zet3_s: zet3}
    N = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    DN_temp = np.array([[float(d.subs(subs)) for d in row] for row in DN_expr], dtype=float)
    DN = DN_temp.T   # (8×3)

    return N, DN

"""==============

#### B27

==============
"""

def Shape_Brick27(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shape functions and derivatives symbolically
    for a 27-node quadratic hexahedral (Brick27) element.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        3-component array [zet1, zet2, zet3] with -1 <= zet_i <= 1.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (27,) array of shape-function values at (zet1, zet2, zet3).
    DN : np.ndarray
        (27, 3) array of derivatives:
    """

    zet1, zet2, zet3 = zeta
    # --- Symbolic variables ---
    zet1_s, zet2_s, zet3_s = sp.symbols("zet1 zet2 zet3")

    # --- 1D L3 shape functions (symbolic) ---
    N1_1D = 0.5 * zet1_s * (zet1_s - 1)
    N2_1D = 1 - zet1_s**2
    N3_1D = 0.5 * zet1_s * (zet1_s + 1)
    N_L3_zet1 = [N1_1D, N2_1D, N3_1D]

    N1_2D = 0.5 * zet2_s * (zet2_s - 1)
    N2_2D = 1 - zet2_s**2
    N3_2D = 0.5 * zet2_s * (zet2_s + 1)
    N_L3_zet2 = [N1_2D, N2_2D, N3_2D]

    N1_3D = 0.5 * zet3_s * (zet3_s - 1)
    N2_3D = 1 - zet3_s**2
    N3_3D = 0.5 * zet3_s * (zet3_s + 1)
    N_L3_zet3 = [N1_3D, N2_3D, N3_3D]

    # --- Tensor product shape functions ---
    N_expr = []
    for i in range(3):
        for j in range(3):
            for k in range(3):
                N_expr.append(N_L3_zet1[i] * N_L3_zet2[j] * N_L3_zet3[k])

    # --- Symbolic derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],  # d/dzet1
        [sp.diff(Ni, zet2_s) for Ni in N_expr],  # d/dzet2
        [sp.diff(Ni, zet3_s) for Ni in N_expr]   # d/dzet3
    ]

    # --- Substitute numeric values ---
    subs = {zet1_s: zet1, zet2_s: zet2, zet3_s: zet3}
    N = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    dN1 = np.array([float(Di.subs(subs)) for Di in DN_expr[0]], dtype=float)
    dN2 = np.array([float(Di.subs(subs)) for Di in DN_expr[1]], dtype=float)
    dN3 = np.array([float(Di.subs(subs)) for Di in DN_expr[2]], dtype=float)

    DN = np.vstack([dN1, dN2, dN3]).T  # (27 x 3)

    return N, DN

"""==============

#### TET4

==============
"""

def Shape_Tet4(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shape functions and derivatives symbolically
    for a 4-node linear tetrahedral (Tet4) element.

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        3-component array [zet1, zet2, zet3] satisfying:
        zet1 >= 0, zet2 >= 0, zet3 >= 0,
        and zet1 + zet2 + zet3 <= 1.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (4,) array of shape-function values evaluated at (zet1, zet2, zet3).
    DN : np.ndarray
        (4, 3) array of derivatives

    """

    zet1, zet2, zet3 = zeta
    # --- Symbolic variables ---
    zet1_s, zet2_s, zet3_s = sp.symbols("zet1 zet2 zet3")

    # --- Symbolic shape functions (from slides / barycentric form) ---
    N_expr = [
        1 - zet1_s - zet2_s - zet3_s,  # Node 1
        zet1_s,                        # Node 2
        zet2_s,                        # Node 3
        zet3_s                         # Node 4
    ]

    # --- Symbolic derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],  # d/dzet1
        [sp.diff(Ni, zet2_s) for Ni in N_expr],  # d/dzet2
        [sp.diff(Ni, zet3_s) for Ni in N_expr]   # d/dzet3
    ]

    # --- Substitute numeric values ---
    subs = {zet1_s: zet1, zet2_s: zet2, zet3_s: zet3}
    N = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    dN1 = np.array([float(Di.subs(subs)) for Di in DN_expr[0]], dtype=float)
    dN2 = np.array([float(Di.subs(subs)) for Di in DN_expr[1]], dtype=float)
    dN3 = np.array([float(Di.subs(subs)) for Di in DN_expr[2]], dtype=float)

    DN = np.vstack([dN1, dN2, dN3]).T  # shape (4 x 3)

    return N, DN

"""==============

#### W6

==============
"""

def Shape_W6(zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute shape functions and derivatives symbolically
    for a 6-node wedge (prism) element (W6).

    *********************************
    Parameters
    *********************************
    zeta : np.ndarray
        3-component array [zet1, zet2, zet3] satisfying:
        zet1 >= 0, zet2 >= 0, zet1 + zet2 <= 1, and -1 <= zet3 <= 1.

    *********************************
    Returns
    *********************************
    N : np.ndarray
        (6,) array of shape-function values at (zet1, zet2, zet3).
    DN : np.ndarray
        (6, 3) array of derivatives:
    """

    zet1, zet2, zet3 = zeta

    # --- Symbolic variables ---
    zet1_s, zet2_s, zet3_s = sp.symbols("zet1 zet2 zet3")

    # --- Base 2D triangular shape functions ---
    N_T3 = [
        1 - zet1_s - zet2_s,
        zet1_s,
        zet2_s
    ]

    # --- 1D linear shape functions (vertical direction) ---
    N_L2 = [
        (1 - zet3_s) / 2,
        (1 + zet3_s) / 2
    ]

    # --- Wedge shape functions (tensor product) ---
    N_expr = [
        N_T3[0] * N_L2[0],  # N1
        N_T3[1] * N_L2[0],  # N2
        N_T3[2] * N_L2[0],  # N3
        N_T3[0] * N_L2[1],  # N4
        N_T3[1] * N_L2[1],  # N5
        N_T3[2] * N_L2[1],  # N6
    ]

    # --- Symbolic derivatives ---
    DN_expr = [
        [sp.diff(Ni, zet1_s) for Ni in N_expr],  # d/dzet1
        [sp.diff(Ni, zet2_s) for Ni in N_expr],  # d/dzet2
        [sp.diff(Ni, zet3_s) for Ni in N_expr]   # d/dzet3
    ]

    # --- Substitute numeric values ---
    subs = {zet1_s: zet1, zet2_s: zet2, zet3_s: zet3}
    N = np.array([float(Ni.subs(subs)) for Ni in N_expr], dtype=float)
    dN1 = np.array([float(Di.subs(subs)) for Di in DN_expr[0]], dtype=float)
    dN2 = np.array([float(Di.subs(subs)) for Di in DN_expr[1]], dtype=float)
    dN3 = np.array([float(Di.subs(subs)) for Di in DN_expr[2]], dtype=float)

    DN = np.vstack([dN1, dN2, dN3]).T  # shape (6 x 3)

    return N, DN

"""==============

#### 3D Dispatcher

==============
"""

def ShapeFunctions3D(EleType: str, zeta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher function for 3D shape functions.

    Routes to the correct subroutine for:
        - TET4   (4-node tetrahedron)
        - W6     (6-node wedge / prism)
        - BRICK8 (8-node hexahedron)
        - BRICK27 (27-node hexahedron)

    *********************************
    Parameters
    *********************************
    EleType : str
        3D element type identifier (case-insensitive).

    zeta : np.ndarray
        3-component coordinate vector [zet1, zet2, zet3].

    *********************************
    Returns
    *********************************
    N : np.ndarray
        Shape function values evaluated at (zet1, zet2, zet3).
    DN : np.ndarray
        Shape function derivatives with respect to (zet1, zet2, zet3).

    *********************************
    Raises
    *********************************
    TypeError
        If zeta is not a NumPy array.
    ValueError
        If zeta has invalid dimension, domain, or unsupported element type.
    """

    # --- Input validation ---
    if not isinstance(zeta, np.ndarray):
        raise TypeError("Input zeta must be a NumPy array.")
    if zeta.size != 3:
        raise ValueError("ShapeFunctions3D expects a 3-component coordinate vector [zet1, zet2, zet3].")

    zet1, zet2, zet3 = zeta

    # --- Domain checks ---
    if EleType == "TET4":
        if zet1 < 0 or zet2 < 0 or zet3 < 0 or (zet1 + zet2 + zet3) > 1:
            raise ValueError(
                f"Invalid (zet1, zet2, zet3)=({zet1}, {zet2}, {zet3}) for {EleType}: "
                "must satisfy zet_i ≥ 0 and zet1 + zet2 + zet3 ≤ 1."
            )

    elif EleType == "W6":
        if zet1 < 0 or zet2 < 0 or (zet1 + zet2) > 1 or not (-1 <= zet3 <= 1):
            raise ValueError(
                f"Invalid (zet1, zet2, zet3)=({zet1}, {zet2}, {zet3}) for {EleType}: "
                "must satisfy zet1 ≥ 0, zet2 ≥ 0, zet1 + zet2 ≤ 1, and -1 ≤ zet3 ≤ 1."
            )

    elif EleType in ["B8", "B27"]:
        if not (-1 <= zet1 <= 1 and -1 <= zet2 <= 1 and -1 <= zet3 <= 1):
            raise ValueError(
                f"Invalid (zet1, zet2, zet3)=({zet1}, {zet2}, {zet3}) for {EleType}: "
                "must lie within [-1, 1]^3."
            )
    else:
        raise ValueError(f"Unsupported 3D element type: {EleType}")

    # --- Element dispatching ---
    if EleType == "TET4":
        N, DN = Shape_Tet4(zeta)

    elif EleType == "W6":
        N, DN = Shape_W6(zeta)

    elif EleType == "B8":
        N, DN = Shape_Brick8(zeta)

    elif EleType == "B27":
        N, DN = Shape_Brick27(zeta)

    return N, DN

"""********************************

### Master Dispatcher

********************************
"""

def ShapeFunctions(EleType: str, zeta: Union[Sequence[float], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    ===============================================================
    ShapeFunctions
    ===============================================================
    Central dispatcher for computing element shape functions and their derivatives at a given reference coordinate.

    *********************************
    PARAMETERS:
    *********************************
    EleType : str
        Element type keyword (case-insensitive). Supported types:
            1D:  L2, L3
            2D:  Q4, Q8, Q9, T3, T6
            3D:  B8, B27, TET4, W6
    zeta : array-like of float
        Reference coordinate(s) where shape functions are evaluated.
        - For 1D elements: [ξ]
        - For 2D elements: [ξ, η]
        - For 3D elements: [ξ, η, ζ]

    *********************************
    RETURNS:
    *********************************
    N  : np.ndarray
        Vector of shape functions evaluated at `zeta`.
    DN : np.ndarray
        Matrix of derivatives of shape functions w.r.t. reference
        coordinates.

    *********************************
    RAISES:
    *********************************
    ValueError
        • If EleType is not recognized.
        • If zeta dimensionality does not match the expected element type.
    TypeError
        • If zeta is not a list, tuple, or NumPy array of numeric values.

    *********************************
    NOTES:
    *********************************
    • Acts as the main entry point; delegates to:
          - ShapeFunctions1D for L2, L3
          - ShapeFunctions2D for Q4, Q8, Q9, T3, T6
          - ShapeFunctions3D for B8, B27, TET4, W6
    • All sub-dispatchers return (N, DN) arrays.

    """


    # --- Validate input types ---
    if not isinstance(EleType, str):
        raise TypeError("EleType must be a string such as 'L2', 'Q4', or 'B8'.")

    if not isinstance(zeta, (list, tuple, np.ndarray)):
        raise TypeError("zeta must be a list, tuple, or NumPy array of floats.")

    # Convert zeta to a NumPy array for internal consistency
    zeta = np.array(zeta, dtype=float).flatten()
    EleType = EleType.replace(" ", "").strip().upper()

    # --- Dimensional dispatching ---
    if EleType in ['L2', 'L3']:
        N, DN = ShapeFunctions1D(EleType, zeta)

    elif EleType in ['Q4', 'Q8', 'Q9', 'T3', 'T6']:
        N, DN = ShapeFunctions2D(EleType, zeta)

    elif EleType in ['B8', 'B27', 'TET4', 'W6']:
        N, DN = ShapeFunctions3D(EleType, zeta)

    else:
        raise ValueError(
            f"Unrecognized element type '{EleType}'. "
            "Supported types: L2, L3, Q4, Q8, Q9, T3, T6, B8, B27, TET4, W6."
        )

    # --- Output validation
    if not (isinstance(N, np.ndarray) and isinstance(DN, np.ndarray)):
        raise TypeError("Returned N and DN must both be NumPy arrays.")

    return N, DN

