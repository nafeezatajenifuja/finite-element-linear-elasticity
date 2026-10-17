# -*- coding: utf-8 -*-
"""
CIVE 7336 – MP3 : Guass_Points
Author: Nafeezat Ajenifuja

Purpose: Develop a function that provides Gauss quadrature points and corresponding weights in dimensions $n = 1, 2, 3$, and for various finite elements.
"""

import numpy as np
from typing import Tuple
from scipy.special import roots_legendre

"""#### Helper functions

*************************************************

#### Input Validation Helper function

*************************************************
"""

def validate_inputs(NGPTS: int, EleType: str, dim: int):
    """
    Centralized validation for NGPTS, EleType, and dim to avoid redundancy.

    ***********
    Parameters:
    ***********
    NGPTS: The number of Gauss points.
    EleType: The element type (e.g., 'L2', 'T3', 'Q4', etc.)
    dim: The dimension of the problem (1, 2, or 3).

    ***********
    Raises:
    ***********
    TypeError: If the input datatype is wrong.
    ValueError: If NGPTS is less than 1, or if EleType is invalid.
    """
    #Check for wrong input data type
    if not isinstance(dim, int) or not isinstance(EleType, str) or not isinstance(NGPTS, int):
        raise TypeError("Invalid input type. 'dim' should be an integer, 'EleType' should be a string, and 'NGPTS' should be an integer.")

    # Validate NGPTS
    if NGPTS < 1:
        raise ValueError(f"Invalid number of points. NGPTS must be >= 1, but got {NGPTS}.")

    # Validate EleType
    valid_elements = ['L2', 'Q4', 'Q8', 'T3', 'B8', 'TET4', 'W6']
    EleType = EleType.upper()  # Case-insensitive comparison
    if EleType not in valid_elements:
        raise ValueError(f"Invalid EleType: {EleType}. Valid entries are: {valid_elements}.")

"""*************************************************

#### Measure Check Helper function

*************************************************
"""

def check_weight_sum(EleType: str, weight_sum: float, expected_value: float, dim: int):
    """
    Checks that the sum of weights matches the expected value for the reference element.
    """
    #if EleType == 'T3' or EleType == 'W6':  # Special handling for T3 and W6 (rounded comparison)
    weight_sum_rounded = float(f"{weight_sum:.6g}")
    expected_value_rounded = float(f"{expected_value:.6g}")
    #else:
        #weight_sum_rounded = weight_sum
        #expected_value_rounded = expected_value

    if weight_sum_rounded != expected_value_rounded:
        raise ValueError(f"Sum of weights {weight_sum_rounded} does not match expected value {expected_value_rounded}.")
    else:
        print(f"\n{'-'*50}")
        print(f"Test Passed: Sum of weights is correct for {dim}D quadrature.")
        print(f"{'-'*50}\n")

"""*************************************************

#### Global initialization of all the different element types and their respective measures

*************************************************
"""

# Global dictionary for expected sums based on (dim, EleType) combination
expected_sums = {
    # 1D: Sum of weights should always be 2
    (1, 'L2'): 2.0,

    # 2D Triangle and Quadrilateral
    (2, 'T3'): 0.5,  # Triangle (area = 0.5)
    (2, 'Q4'): 4.0,  # Quadrilateral (area = 4.0)

    # 3D: Tetrahedron, Brick, and Wedge
    (3, 'B8'): 8.0,  # Brick (Cube [-1,1]^3)
    (3, 'TET4'): 1/6,  # Tetrahedron
    (3, 'W6'): 1.0,  # Wedge
}

"""### #1D

*************************************************

#### Gauss Points for 1 dimensional problem dispatcher function

*************************************************
"""

def GaussPoints_1D(NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the Gauss quadrature points and weights from Legendre polynomials

    ***********
    Parameters:
    ***********

    NGPTS (int): The number of Gaussian quadrature points.

    ***********
    Returns:
    ***********
    Tuple[r: np.ndarray, w: np.ndarray]:
        - r (np.ndarray): Locations of the quadrature points.
        - w (np.ndarray): Weights associated with the quadrature points.

    ***********
    Raises:
    ***********
    ValueError: If the number of points is less than 1.
    """

    # Validate inputs
    validate_inputs(NGPTS=NGPTS, EleType="L2", dim=1)

    # Return the roots, which are the Gauss points and weights for Legendre polynomial of degree `NGPTS`
    r, w = roots_legendre(NGPTS)

    # Return the roots and weights as numpy arrays
    return r, w

"""*************************************************

### #2D

*************************************************

#### Triangles (T3)

*************************************************
"""

def GaussPoints_2D_Triangle(NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Gauss quadrature points and corresponding weights for a right-angled triangle (T3) with dimensions h=b=1.

    ***********
    Parameters:
    ***********
    NGPTS (int): Total number of Gaussian quadrature points.
                 Supported values: 1, 3, 4, 7, 9, 12, 13

    ***********
    Returns:
    ***********
    Tuple[np.ndarray, np.ndarray]:
        - r: Locations of quadrature points (NGPTS x 2(dim) array)
        - w: Weights associated with the quadrature points (NGPTS array)

    ***********
    Raises:
    ***********
    ValueError: If NGPTS is not supported.
    """

    # 1-point quadrature
    if NGPTS == 1:
        r = np.array([[1/3, 1/3]])
        w = np.array([0.5])

    # 3-point quadrature
    elif NGPTS == 3:
        r = np.array([[0.5, 0.0],
                      [0.5, 0.5],
                      [0.0, 0.5]])
        w = np.array([1/6, 1/6, 1/6])

    # 4-point quadrature
    elif NGPTS == 4:
        r = np.array([[1/3, 1/3],
                      [0.6, 0.2],
                      [0.2, 0.6],
                      [0.2, 0.2]])
        w = np.array([-0.5625, 0.520833333333333, 0.520833333333333, 0.520833333333333]) * 0.5

    # 7-point quadrature
    elif NGPTS == 7:
        r = np.array([[0.333333333333333, 0.333333333333333],
                      [0.797426985353087, 0.101286507323456],
                      [0.101286507323456, 0.797426985353087],
                      [0.101286507323456, 0.101286507323456],
                      [0.470142064105115, 0.059715871789770],
                      [0.059715871789770, 0.470142064105115],
                      [0.470142064105115, 0.470142064105115]])
        w = np.array([0.225000000000000,
                      0.125939180544827,
                      0.125939180544827,
                      0.125939180544827,
                      0.132394152788506,
                      0.132394152788506,
                      0.132394152788506]) * 0.5

    # 9-point quadrature
    elif NGPTS == 9:
        r = np.array([[0.124949503233232, 0.437525248383384],
                      [0.437525248383384, 0.124949503233232],
                      [0.437525248383384, 0.437525248383384],
                      [0.797112651860071, 0.165409927389841],
                      [0.797112651860071, 0.037477420750088],
                      [0.165409927389841, 0.797112651860071],
                      [0.165409927389841, 0.037477420750088],
                      [0.037477420750088, 0.797112651860071],
                      [0.037477420750088, 0.165409927389841]])
        w = np.array([0.205950504760887,
                      0.205950504760887,
                      0.205950504760887,
                      0.063691414286223,
                      0.063691414286223,
                      0.063691414286223,
                      0.063691414286223,
                      0.063691414286223,
                      0.063691414286223]) * 0.5

    # 12-point quadrature
    elif NGPTS == 12:
        r = np.array([[0.873821971016996, 0.063089014491502],
                      [0.063089014491502, 0.873821971016996],
                      [0.063089014491502, 0.063089014491502],
                      [0.501426509658179, 0.249286745170910],
                      [0.249286745170910, 0.501426509658179],
                      [0.249286745170910, 0.249286745170910],
                      [0.636502499121399, 0.310352451033785],
                      [0.636502499121399, 0.053145049844816],
                      [0.310352451033785, 0.636502499121399],
                      [0.310352451033785, 0.053145049844816],
                      [0.053145049844816, 0.636502499121399],
                      [0.053145049844816, 0.310352451033785]])
        w = np.array([0.050844906370207,
                      0.050844906370207,
                      0.050844906370207,
                      0.116786275726379,
                      0.116786275726379,
                      0.116786275726379,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374,
                      0.082851075618374]) * 0.5

    # 13-point quadrature
    elif NGPTS == 13:
        r = np.array([[0.333333333333333, 0.333333333333333],
                      [0.479308067841923, 0.260345966079038],
                      [0.260345966079038, 0.479308067841923],
                      [0.260345966079038, 0.260345966079038],
                      [0.869739794195568, 0.065130102902216],
                      [0.065130102902216, 0.869739794195568],
                      [0.065130102902216, 0.065130102902216],
                      [0.638444188569809, 0.312865496004875],
                      [0.638444188569809, 0.086903154253160],
                      [0.312865496004875, 0.638444188569809],
                      [0.312865496004875, 0.086903154253160],
                      [0.086903154253160, 0.638444188569809],
                      [0.086903154253160, 0.312865496004875]])
        w = np.array([-0.149570044467670,
                      0.175615257433204,
                      0.175615257433204,
                      0.175615257433204,
                      0.053347235608839,
                      0.053347235608839,
                      0.053347235608839,
                      0.077113760890257,
                      0.077113760890257,
                      0.077113760890257,
                      0.077113760890257,
                      0.077113760890257,
                      0.077113760890257]) * 0.5

    else:
        raise ValueError(f"Triangle quadrature with NGPTS={NGPTS} not implemented.")

    return r, w

"""*************************************************

#### Quadilaterals (Q4)

*************************************************
"""

def GaussPoints_2D_Quad(NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D Gauss quadrature points and weights for a quadrilateral element (Q4)

    ***********
    Parameters:
    ***********
    NGPTS (int): Number of Gauss points per dimension.

    ***********
    Returns:
    ***********
    Tuple[np.ndarray, np.ndarray]:
        - r: Locations of quadrature points (NGPTS^2 x 2 (dim) array).
        - w: Weights associated with the quadrature points (NGPTS^2, array).
    """
    # 1D Gauss points and weights
    r1d, w1d = GaussPoints_1D(NGPTS)

    # Generate all combinations of 1D points using meshgrid and reshape to the desired array
    X, Y = np.meshgrid(r1d, r1d, indexing='ij')  # NGPTS x NGPTS arrays
    r = np.column_stack([X.ravel(), Y.ravel()])   # NGPTS^2 x 2 array

    # Generate 2D weights as tensor product
    W = np.outer(w1d, w1d)  # NGPTS x NGPTS array
    w = W.ravel()            # Flatten to 1D array

    return r, w

"""
*************************************************
#### Quadrilaterals (Q8)
*************************************************
"""

def GaussPoints_2D_Q8() -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes 2D Gauss quadrature points and weights for a Q8 element
    using the 3-point Gauss rule (3x3 = 9 points).

    --------------------------------------------
    Returns:
    --------------------------------------------
    r : (9 x 2) array
        Quadrature point coordinates in reference space.
    w : (9,) array
        Quadrature weights.
    """

    # Q8 requires 3 Gauss points per dimension
    NGPTS = 3

    # Use the same general quadrilateral Gauss generator as Q4
    r, w = GaussPoints_2D_Quad(NGPTS)

    return r, w

"""*************************************************

#### Gauss Points for 2 dimensional problem dispatcher function

*************************************************
"""

def GaussPoints_2D(EleType: str, NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher function for 2D Gaussian quadrature. Selects the appropriate 2D function for calculating the Gauss
    points and corresponding weights based on the element type (either triangle (T3) or quadrilaterals (Q4)).

    ***********
    Parameters:
    ***********
    EleType (str): The element type. 'T3' for triangle, 'Q4' for quadrilateral.
    NGPTS (int): Number of Gaussian quadrature points: Total number for T3 and per dimension for Q4.

    ***********
    Returns:
    ***********
    Tuple[np.ndarray, np.ndarray]:
        - r: Locations of the quadrature points
        - w: Quadrature weights

    ***********
    Raises:
    ***********
    ValueError: If the element type is not recognized.
    """

    # Validate inputs once, before calling element-specific function
    validate_inputs(NGPTS=NGPTS, EleType=EleType, dim=2)

    # Dispatcher logic
    if EleType == 'T3':
        return GaussPoints_2D_Triangle(NGPTS)
    elif EleType == 'Q4':
        return GaussPoints_2D_Quad(NGPTS)  # or GaussPoints_2D_Quad_Tile
    elif EleType == "Q8":
        return GaussPoints_2D_Q8()
    else:
        raise ValueError(f"Invalid element type '{EleType}' for 2D. Supported types include T3, Q4, Q8.")

"""*************************************************

### #3D

*************************************************

#### Tetrahedra (TET4)

*************************************************
"""

def GaussPoints_3D_Tetrahedra(NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Gauss quadrature points and corresponding weights for a 4-node tetrahedron (TET4) element.

    ***********
    Parameters:
    ***********
    NGPTS (int): Total number of quadrature points. Supported number of points include 1, 4, 5, 11, 15.

    ***********
    Returns:
    ***********
    Tuple[np.ndarray, np.ndarray]:
        - r: Locations of quadrature points (NGPTS x 3(dim) array)
        - w: Weights associated with the quadrature points (NGPTS, array)

    ***********
    Raises:
    ***********
    ValueError: If NGPTS is not supported
    """
    if NGPTS == 1:
        r = np.array([[1/4, 1/4, 1/4]])
        w = np.array([1/6])

    elif NGPTS == 4:
        p1 = 0.5854101966249638
        p2 = 0.1381966011250105
        r = np.array([
            [p1, p2, p2],
            [p2, p1, p2],
            [p2, p2, p1],
            [p1, p1, p1]
        ])
        w = np.array([1/(6*4)]*4)

    elif NGPTS == 5:
        r = np.array([
            [1/4, 1/4, 1/4],
            [1/2, 1/6, 1/6],
            [1/6, 1/2, 1/6],
            [1/6, 1/6, 1/2],
            [1/6, 1/6, 1/6]
        ])
        w = np.array([-4/5, 9/20, 9/20, 9/20, 9/20]) / 6

    elif NGPTS == 11:
        p1, p2, p3, p4, p5 = 0.25, 0.785714285714286, 0.071428571428571, 0.399403576166799, 0.100596423833201
        q1, q2, q3 = -0.013155555555556, 0.007622222222222, 0.024888888888889
        r = np.array([
            [p1, p1, p1],
            [p2, p3, p3],
            [p3, p2, p3],
            [p3, p3, p2],
            [p3, p3, p3],
            [p4, p5, p5],
            [p5, p4, p5],
            [p5, p5, p4],
            [p5, p4, p4],
            [p4, p5, p4],
            [p4, p4, p5]
        ])
        w = np.array([q1, q2, q2, q2, q2, q3, q3, q3, q3, q3, q3])

    elif NGPTS == 15:
        p1, p2, p3, p4, p5, p6, p7 = 0.25, 0.0, 0.333333333333333, 0.727272727272727, 0.090909090909091, 0.066550153573664, 0.433449846426336
        q1, q2, q3, q4 = 0.030283678097089, 0.006026785714286, 0.011645249086029, 0.010949141561386
        r = np.array([
            [p1, p1, p1],
            [p2, p3, p3],
            [p3, p2, p3],
            [p3, p3, p2],
            [p3, p3, p3],
            [p4, p5, p5],
            [p5, p4, p5],
            [p5, p5, p4],
            [p5, p5, p5],
            [p6, p7, p7],
            [p7, p6, p7],
            [p7, p7, p6],
            [p7, p6, p6],
            [p6, p7, p6],
            [p6, p6, p7]
        ])
        w = np.array([q1, q2, q2, q2, q2, q3, q3, q3, q3, q4, q4, q4, q4, q4, q4])

    else:
        raise ValueError(f"Tetrahedral quadrature with NGPTS={NGPTS} not implemented.")

    return r, w

"""*************************************************

#### Brick (B8)

*************************************************
"""

def GaussPoints_3D_Brick(NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Gauss quadrature points and corresponding weights for an 8-node brick (B8) element.

    ***********
    Parameters:
    ***********
    NGPTS (int): Number of Gauss points per dimension.

    ***********
    Returns:
    ***********
    Tuple[np.ndarray, np.ndarray]:
        - r: Locations of quadrature points (NGPTS^3 x 3(dim) array).
        - w: Weights associated with the quadrature points(NGPTS^3, array)
    """
    # 1D Gauss points and weights
    r1d, w1d = GaussPoints_1D(NGPTS)

    # Generate all combinations of 1D points using meshgrid and reshape to the desired array
    X, Y, Z = np.meshgrid(r1d, r1d, r1d, indexing='ij') # NGPTS x NGPTS x NGPTS arrays

    # Flatten to get Nx3 array of points
    r = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()]) #NGPTS^3 X 3 array

    # Compute 3D weights as tensor product
    W = np.outer(np.outer(w1d, w1d).ravel(), w1d).ravel()
    w = W

    return r, w

"""*************************************************

#### Wedge (W6)

*************************************************
"""

def GaussPoints_3D_Wedge(NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes Gauss quadrature points and corresponding weights for a 6-node wedge (W6) element.

    ***********
    Parameters
    ***********
    NGPTS: int
        Total number of Gaussian quadrature points. Supported values: 6 and 9

    ***********
    Returns
    ***********
    Tuple[np.ndarray, np.ndarray]:
        - r: Locations of quadrature points ((num_1D_pts *num_2D_pts) x 3(dim) array).
        - w: Weights associated with the quadrature points((num_1D_pts *num_2D_pts), array)
    """

    # Get 2D Gauss points and weights for triangular face
    r2D, w2D =  GaussPoints_2D_Triangle(NGPTS) # shape (NGPTS, 2)

    # Get 1D Gauss points and weights along the eta2 direction
    r1D, w1D = GaussPoints_1D(NGPTS) # shape (NGPTS,)

    # Vectorized combination of points.
    r_tri_rep = np.repeat(r2D, NGPTS, axis=0)  # repeat each 2D point for each 1D point
    r_line_tile = np.tile(r1D, NGPTS)          # tile 1D points to match repetitions

    #Combine x, z from the triangle and y from the line
    eta1 = r_tri_rep[:, 0]
    eta3 = r_tri_rep[:, 1]
    eta2 = r_line_tile
    r = np.column_stack([eta1, eta2, eta3])

    # Vectorized combination of weights
    w_tri_rep = np.repeat(w2D, NGPTS) # ((num_1D_pts *num_2D_pts),)
    w_line_tile = np.tile(w1D, NGPTS) # ((num_1D_pts *num_2D_pts),)
    w = w_tri_rep * w_line_tile

    return r, w

"""*************************************************

#### Gauss Points for 3 dimensional problems dispatcher function

*************************************************
"""

def GaussPoints_3D(EleType: str, NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Dispatcher function for 3D Gaussian quadrature. Selects the appropriate 3D function for calculating the Gauss
    points and corresponding weights based on the element type (either Brick (B8), tetrahedra (TET4), or wedge (W6)).

    ***********
    Parameters:
    ***********
    EleType (str): The element type. 'B8' for brick, 'TET4' for tetrahedral, and 'W6' for wedge.
    NGPTS (int): Number of Gaussian quadrature points per dimension (or total points for tetrahedral/wedge).

    ***********
    Returns:
    ***********
    Tuple[np.ndarray, np.ndarray]:
        - r: Locations of the quadrature points
        - w: Quadrature weights

    ***********
    Raises:
    ***********
    ValueError: If the element type is not recognized.
    """
    # Validate inputs
    EleType = EleType.upper()
    validate_inputs(NGPTS=NGPTS, EleType=EleType, dim=3)

    # Dispatcher logic
    if EleType == 'B8':
        return GaussPoints_3D_Brick(NGPTS)
    elif EleType == 'TET4':
        return GaussPoints_3D_Tetrahedra(NGPTS)
    elif EleType == 'W6':
        return GaussPoints_3D_Wedge(NGPTS)
    else:
        raise ValueError(f"Invalid element type '{EleType}' for 3D. Supported types include B8, TET4, W6.")

"""### #Overall Dispatcher: GaussPoints"""

def GaussPoints (dim: int, EleType: str, NGPTS: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    A dispatcher function for performing Gauss quadrature. Selects the appropriate function for calculating the Gauss
    points and corresponding weights based on the input dimension and element type.

    ***********
    Parameters:
    ***********
    dim (int): The dimension of the problem (1, 2, or 3).
    EleType (str): The element type which includes ('L2' for 1D, 'Q4' for 2D square, 'T3' for 2D triangle, 'B8' for 3D brick, 'TET4' for 3D tetrahedron, 'W6' for 3D wedge).
    NGPTS (int): The number of Gaussian quadrature points.

    ***********
    Returns:
    ***********
    Tuple[r: np.ndarray, w: np.ndarray]:
        - r (np.ndarray): Locations of the quadrature points.
        - w (np.ndarray): Weights associated with the quadrature points.

    ***********
    Raises:
    ***********
    ValueError: If the element type is not recognized.
    """
    # Normalize the element type once here
    EleType = EleType.strip().upper()

    # Validate the input types and range
    validate_inputs(NGPTS=NGPTS, EleType=EleType, dim=dim)

    #dispatcher logic
    if dim == 1:
        return GaussPoints_1D(NGPTS)
    elif dim == 2:
        return GaussPoints_2D(EleType, NGPTS)
    elif dim == 3:
        return GaussPoints_3D(EleType, NGPTS)
    else:
        raise ValueError("Unsupported dimension. Only 1, 2, and 3 are supported.")

