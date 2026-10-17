# ======================================================================
#  CIVE 7336 – Project : Stress_Recovery
#  Author : Nafeezat Ajenifuja (Python transcription of KBN's MATLAB code)
# ----------------------------------------------------------------------
#  PURPOSE :
#     Calculates stress field using several methods:
#       - Gauss points        ("GPT")
#       - At element nodes    ("EleNodes")
#       - Averaged at nodes   ("AvgNodes")
# ======================================================================

import numpy as np
from gauss_points import GaussPoints
from Shape_functions import ShapeFunctions
from Get_Elastic_Moduli import Get_Elastic_Moduli


def Stress_Recovery(Connectivity: np.ndarray,
                    Coord: np.ndarray,
                    EleType: str,
                    medium_set: dict,
                    NGPTS: int,
                    recovery_type: str,
                    U: np.ndarray) -> np.ndarray:
    """
    Calculates stress field using several methods.

    Parameters
    ----------
    Connectivity : (Nele x NodesPerEle) array
        Element connectivity matrix (1-based node numbers).
    Coord : (NumNodes x dim) array
        Nodal coordinates.
    EleType : str
        Element type (e.g., 'L2').
    medium_set : dict
        Medium properties (e.g., Lame parameters).
    NGPTS : int
        Number of Gauss points per element.
    recovery_type : str
        One of {'GPT', 'EleNodes', 'AvgNodes'}.
    U : (NumNodes x dim) array
        Nodal displacement vector.

    Returns
    -------
    sigma : array
        Stress field:
          - For 'GPT':    shape (Nele*NGPTS, 2)  [x, sigma]
          - For 'EleNodes': shape (Nele*NodesPerEle, 2)
          - For 'AvgNodes': shape (NumNodes, 2)
    """

    rt = recovery_type.lower()

    if rt == "gpt":
        return Stress_Recovery_GPT(Connectivity, Coord, EleType,
                                   medium_set, NGPTS, U)

    if rt == "elenodes":
        return Stress_Recovery_EleNodes(Connectivity, Coord, EleType,
                                        medium_set, U)

    if rt == "avgnodes":
        return Stress_Recovery_AvgNodes(Connectivity, Coord, EleType,
                                        medium_set, U)

    raise ValueError(f"Unknown recovery_type '{recovery_type}'. "
                     "Use 'GPT', 'EleNodes', or 'AvgNodes'.")


# =================================================
#  Subfunction: Calculate stress at Gauss points
# -------------------------------------------------
def Stress_Recovery_GPT(Connectivity: np.ndarray,
                        Coord: np.ndarray,
                        EleType: str,
                        medium_set: dict,
                        NGPTS: int,
                        U: np.ndarray) -> np.ndarray:
    """
    Calculates stress at Gauss points.

    Returns
    -------
    stress_GPT : (Nele*NGPTS x 2) array
        [x, sigma] at each Gauss point.
    """
    dim = Coord.shape[1]
    Nele = Connectivity.shape[0]

    # Get Gauss points and weights
    r, w = GaussPoints(dim, EleType, NGPTS)
    r = np.asarray(r).flatten()
    w = np.asarray(w).flatten()

    # Initialization
    stress_GPT = np.zeros((Nele * len(w), 2))

    # Loop over elements
    for ele in range(Nele):
        # EleNodes: as MATLAB, 1-based in Connectivity
        EleNodes = Connectivity[ele, :].astype(int)
        node_ids = EleNodes - 1  # to 0-based

        xCap = Coord[node_ids, :]      # (NodesPerEle x dim)
        uCap = U[node_ids, :]          # (NodesPerEle x dim)

        xCap_T = xCap.T                # (dim x NodesPerEle)

        for gpt in range(len(w)):
            zeta = r[gpt]

            N, DN = ShapeFunctions(EleType, [zeta])   # N: (NodesPerEle,), DN: (NodesPerEle x dim_ref)

            # Spatial coordinate x (1D bar: scalar)
            x = float(xCap_T @ N)

            # Jacobian and B
            J = xCap_T @ DN                        # (dim x dim_ref)
            invJ = np.linalg.inv(J)
            B = DN @ invJ                          # (NodesPerEle x dim)

            # Gradient of displacement: gradu = B' * uCap
            # For 1D, this is scalar; we take the (0,0) entry.
            gradu = (B.T @ uCap)                   # (dim x dim)
            gradu_scalar = float(gradu[0, 0])

            # Strain (1D: symmetric gradient)
            strain_GPT = 0.5 * (gradu_scalar + gradu_scalar)  # scalar

            # Young's modulus from lambda,mu
            lambda_, mu_ = Get_Elastic_Moduli(medium_set, np.array([x]))
            E = mu_ * (3 * lambda_ + 2 * mu_) / (lambda_ + mu_)

            # Stress
            sigma_val = E * strain_GPT

            row = NGPTS * ele + gpt
            stress_GPT[row, 0] = x
            stress_GPT[row, 1] = sigma_val

    return stress_GPT


# =============================================
#  Subfunction: Calculate stress at EleNodes
# ---------------------------------------------
def Stress_Recovery_EleNodes(Connectivity: np.ndarray,
                             Coord: np.ndarray,
                             EleType: str,
                             medium_set: dict,
                             U: np.ndarray) -> np.ndarray:
    """
    Calculates stress at element nodes.

    Returns
    -------
    stress_EleNodes : (Nele*NodesPerEle x 2) array
        [x, sigma] at each element node.
    """
    Nele, NodesPerEle = Connectivity.shape

    stress_EleNodes = np.zeros((Nele * NodesPerEle, 2))

    # Location of nodes in reference element (for L2)
    r = np.array([-1.0, 1.0])

    for ele in range(Nele):
        EleNodes = Connectivity[ele, :].astype(int)
        node_ids = EleNodes - 1

        xCap = Coord[node_ids, :]      # (NodesPerEle x dim)
        uCap = U[node_ids, :]          # (NodesPerEle x dim)
        xCap_T = xCap.T

        for gpt in range(NodesPerEle):
            zeta = r[gpt]
            N, DN = ShapeFunctions(EleType, [zeta])

            # Spatial coordinate
            x = float(xCap_T @ N)

            # Jacobian and B
            J = xCap_T @ DN
            invJ = np.linalg.inv(J)
            B = DN @ invJ

            gradu = (B.T @ uCap)       # (dim x dim)
            gradu_scalar = float(gradu[0, 0])

            strain_EleNodes = 0.5 * (gradu_scalar + gradu_scalar)

            lambda_, mu_ = Get_Elastic_Moduli(medium_set, np.array([x]))
            E = mu_ * (3 * lambda_ + 2 * mu_) / (lambda_ + mu_)

            sigma_val = E * strain_EleNodes

            row = NodesPerEle * ele + gpt
            stress_EleNodes[row, 0] = x
            stress_EleNodes[row, 1] = sigma_val

    return stress_EleNodes


# ==========================================
#  Subfunction: Averaging stress at nodes
# ------------------------------------------
def Stress_Recovery_AvgNodes(Connectivity: np.ndarray,
                             Coord: np.ndarray,
                             EleType: str,
                             medium_set: dict,
                             U: np.ndarray) -> np.ndarray:
    """
    Calculates nodal stress by averaging over connected elements.

    Returns
    -------
    stress_Nodes : (NumNodes x 2) array
        [x, sigma_avg] at each node.
    """
    # First get stresses at element nodes
    stress_EleNodes = Stress_Recovery_EleNodes(Connectivity, Coord,
                                               EleType, medium_set, U)

    NumNodes = Coord.shape[0]
    Nele = Connectivity.shape[0]

    stress_Nodes = np.zeros((NumNodes, 2))

    # Fill coordinates
    # (Coord is (NumNodes x 1) in 1D)
    stress_Nodes[:, 0] = Coord[:, 0]

    # Accumulate contributions from each element, as in MATLAB
    for ele in range(Nele):
        # Maps directly from MATLAB indexing:
        # stress_Nodes(ele,2)   += stress_EleNodes(2*ele-1,2)
        # stress_Nodes(ele+1,2) += stress_EleNodes(2*ele,2)
        stress_Nodes[ele, 1]   += stress_EleNodes[2 * ele, 1]
        stress_Nodes[ele + 1, 1] += stress_EleNodes[2 * ele + 1, 1]

    # Average interior nodes (2..NumNodes-1 in MATLAB)
    if NumNodes > 2:
        stress_Nodes[1:NumNodes - 1, 1] *= 0.5

    return stress_Nodes
