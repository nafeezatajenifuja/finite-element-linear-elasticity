# ======================================================================
#  CIVE 7336 – MP3 : TMatrix
#  Author : Professor Kalyana B. Nakshatrala  (translated to Python)
# ----------------------------------------------------------------------
#  PURPOSE :
#     Creates TMatrix (matrix of transposer) used in linear elasticity.
#
#     This is a *direct transcription* of the MATLAB code:
#
#         function [T] = TMatrix(m,n)
#
# ======================================================================

import numpy as np

def TMatrix(m, n):
    """
    Creates the T-matrix used in class linear elasticity formulations.

    Direct transcription of the MATLAB code:

        function [T] = TMatrix(m,n)

        Im = eye(m);  In = eye(n);
        T  = zeros(m*n , m*n);

        for i = 1:n
            T((i-1)*m + (1:m), 1:m*n) = kron(Im, In(i,:));
        end
    """

    # ===========================;
    #   Initializing the arrays  ;
    # ===========================;
    Im = np.eye(m)
    In = np.eye(n)
    T  = np.zeros((m * n, m * n))

    # ========================;
    #   Forming the T matrix  ;
    # ========================;
    for i in range(1, n+1):   # MATLAB: 1:n
        row_idx = (i-1) * m + np.arange(0, m)    # (i-1)*m + (1:m)
        T[row_idx, :] = np.kron(Im, In[i-1, :])  # kron(Im, In(i,:))

    return T
