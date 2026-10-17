# ======================================================================
#  CIVE 7336 – Project : CalculateLocalMatrices (Linear Elasticity Version)
#  Author : Nafeezat Ajenifuja
# ----------------------------------------------------------------------
#  PURPOSE :
#     Compute Klocal and rlocal for a single element (Linear Elasticity).
#     Conventions :
#       - zeta is treated as a column vector
#       - J = xCap.T * DN
#       - x = xCap.T @ N
#       - B = DN @ invJ
# ======================================================================

import numpy as np
from typing import Dict, Tuple
from Shape_functions import ShapeFunctions
from Vector import Vector
from TMatrix import TMatrix
from Get_Elastic_Moduli import Get_Elastic_Moduli
from get_rhob import Get_Rhob


def CalculateLocalMatrices(medium_set: Dict,
                           dofs_per_node: int,
                           EleNodes: np.ndarray,
                           EleType: str,
                           load_type: Dict,
                           r: np.ndarray,
                           w: np.ndarray,
                           xCap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

    # --------------------------------------------------------------
    # Basic geometric information
    # --------------------------------------------------------------
    NodesPerEle = EleNodes.size
    dim         = xCap.shape[1]
    ndofs_e     = NodesPerEle * dofs_per_node

    Klocal = np.zeros((ndofs_e, ndofs_e))
    rlocal = np.zeros((ndofs_e, 1))

    r = np.asarray(r)
    w = np.asarray(w).flatten()

    if r.ndim == 1:
        r = r[:, np.newaxis]

    ngauss, dim_ref = r.shape
    xCap_T = xCap.T

    # ==============================================================
    # Loop over Gauss points (class-style)
    # ==============================================================
    for gpt in range(ngauss):

        # --------------------------------------
        # Sample Gauss point
        # --------------------------------------
        zeta = r[gpt, :]

        # --------------------------------------
        # Compute N and DN
        # --------------------------------------
        N, DN = ShapeFunctions(EleType, zeta)

        # --------------------------------------
        # Compute Jacobian J, detJ, invJ
        # --------------------------------------
        J = xCap_T @ DN
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)

        # --------------------------------------
        # Compute B matrix (gradients)
        # --------------------------------------
        B = DN @ invJ     # (NodesPerEle × dim)

        # --------------------------------------
        # Compute spatial location x
        # --------------------------------------
        x = xCap_T @ N    # (dim,)

        # --------------------------------------
        # Get elastic moduli (lambda, mu)
        # --------------------------------------
        lambda_, mu_ = Get_Elastic_Moduli(medium_set, x)

        # --------------------------------------
        # === Stiffness Matrix Contributions ===
        # --------------------------------------

        # 1. Volumetric (lambda) term
        vB = Vector(B.T)                         # Vector(B')
        Klocal += w[gpt] * lambda_ * (vB @ vB.T) * detJ

        # 2. First mu term: kron(B, I) * kron(B', I)
        kron1 = np.kron(B, np.eye(dim))
        kron2 = np.kron(B.T, np.eye(dim))
        Klocal += w[gpt] * mu_ * (kron1 @ kron2) * detJ

        # 3. Second mu term: kron(B, I) * TMatrix * kron(B', I)
        TM = TMatrix(dim, dim)
        Klocal += w[gpt] * mu_ * (kron1 @ TM @ kron2) * detJ

        # --------------------------------------
        # === Load Vector ===
        # --------------------------------------
        kronN = np.kron(N[:, None], np.eye(dim))   # kron(N', I)
        body_force = Get_Rhob(load_type, x)        # dim×1
        rlocal += w[gpt] * (kronN @ body_force) * detJ

    return Klocal, rlocal
