"""
# ======================================================================
#  CIVE 7336 – MP3 : Vector
#  Author : Nafeezat
# ----------------------------------------------------------------------
#  PURPOSE :
#     Stack the columns of a matrix into a single column vector..
# ======================================================================
"""

import numpy as np

def Vector(A: np.ndarray) -> np.ndarray:
    """
    Stack the columns of a matrix into a single column vector.

    *********************************
    Parameters
    *********************************
    A : np.ndarray
        Input matrix of any shape (m x n)

    *********************************
    Returns
    *********************************
    vecA : np.ndarray
        Column vector (mn x 1) formed by stacking
        columns of A sequentially (column-major order).

    *********************************
    Raises
    *********************************
    TypeError
        If A is not a NumPy array.
    ValueError
        If A is empty.
    """
    if not isinstance(A, np.ndarray):
        raise TypeError("Input A must be a NumPy array.")
    if A.size == 0:
        raise ValueError("Input matrix A cannot be empty.")

    # Stack columns (column-major / Fortran order)
    vecA = A.flatten(order='F').reshape(-1, 1)
    return vecA