# ==========================================================
#  Driver_LE.py
#  Purpose: Run the Linearized Elasticity FEM simulation
#  Author : Nafeezat Ajenifuja
#  Course : CIVE 7336 – Finite Element Methods
# ==========================================================

import numpy as np
from create_id import Create_ID_Matrix
from Create_constraints_vector import Create_ConstraintsVector
from Calculate_global_matrices_LE import CalculateGlobalMatricesLE
from PostProcessing import PostProcessing


def Driver_LE(Connectivity, Constraints, Coord,
                            medium_set, dim, dofs_per_node,
                            EleType, load_type, NCons, Nele,
                            NGPTS, NumNodes, ptload):
    """
    Runs the full linearized elasticity simulation.
    
    Parameters
    ----------
    Connectivity : (Nele x nnode)
        Element connectivity matrix.
    
    Constraints : (NCons x 3)
        Dirichlet conditions [node, dof, value].
    
    Coord : (NumNodes x dim)
        Nodal coordinates.
    
    medium_set : dict
        Material properties. Must define Lame parameters.
        Supported forms:
            medium_set = {
                "type": "Lame_params",
                "lambda": float,
                "mu": float
            }
    
    dim : int
        Spatial dimension (2 or 3).
    
    dofs_per_node : int
        Must equal dim for elasticity (2 in 2D, 3 in 3D).
    
    EleType : str
        Element type (e.g. "Q4", "T3").
    
    load_type : dict
        Body force description.
        Supported forms:
            {"case": "2D_gravity", "rhob": float}
            {"case": "3D_gravity", "rhob": float}
    
    NCons, Nele, NGPTS, NumNodes : int
        Problem sizes.
    
    Returns
    -------
    U : (NumNodes x dofs_per_node)
        Complete nodal displacement field.
    """

    print("\n" + "=" * 65)
    print(" LINEAR ELSATICITY SIMULATION STATUS REPORT")
    print("=" * 65 + "\n")
    
    # ------------------------------------------------------
    # Check DOFs-per-node vs dimension consistency
    # ------------------------------------------------------
    if dofs_per_node != dim:
        raise ValueError(
            f"ERROR: Linear Elasticity requires dofs_per_node == dim.\n"
            f"Received dofs_per_node={dofs_per_node}, dim={dim}."
        )

    # ------------------------------------------------------
    # Step 1 – Create Global ID Matrix
    # ------------------------------------------------------
    print("Step 1: Creating Global ID Matrix ...")
    GlobalID, NEqns = Create_ID_Matrix(Constraints, dofs_per_node, NumNodes)
    print(f"   → GlobalID shape: {GlobalID.shape},   NEqns = {NEqns}\n")

    # ------------------------------------------------------
    # Step 2 – Create Constraints Vector
    # ------------------------------------------------------
    print("Step 2: Creating Constraint Vector U_P ...")
    U_P = Create_ConstraintsVector(Constraints, GlobalID)
    print(f"   → U_P shape: {U_P.shape}\n")

    # ------------------------------------------------------
    # Step 3 – Assemble Global Matrices and Load Vector
    # ------------------------------------------------------
    print("Step 3: Calculating Global Stiffness Matrix and Load Vector ...")
    K_FF, K_FP, R_F = CalculateGlobalMatricesLE(
    medium_set,
    load_type,
    dofs_per_node,
    Connectivity,
    EleType,
    NGPTS,
    Coord,
    GlobalID
    )
    
    # ------------------------------------------------------
    # Step 3.5 – Apply Nodal Point Loads (if provided)
    # ------------------------------------------------------
    if ptload is not None:
        print("Step 3.5: Applying nodal point loads ...")
    
        for i in range(ptload.shape[0]):
            node = int(ptload[i, 0])        # node number
            dof_local = int(ptload[i, 1])   # 1=x, 2=y in 2D
            val = ptload[i, 2]              # load magnitude
    
            # Convert node,dof → global DOF index
            global_dof = GlobalID[node - 1, dof_local - 1]
    
            if global_dof > 0:   # free DOF
                R_F[global_dof - 1, 0] += val
            else:
                print(f"  WARNING: Load applied on prescribed DOF {global_dof}")
    
        print("   Point loads applied.\n")

    # ------------------------------------------------------
    # Step 4 – Solve Linear System
    # ------------------------------------------------------
    print("Step 4: Solving Linear System K_FF · U_F = R_F – K_FP · U_P ...")
    rhs = R_F - K_FP @ U_P
    U_F = np.linalg.solve(K_FF, rhs)
    print(f"   → U_F shape: {U_F.shape}\n")

    # ------------------------------------------------------
    # Step 5 – Post-Processing
    # ------------------------------------------------------
    print("Step 5: Reconstructing Full Nodal Vector U ...")
    U = PostProcessing(GlobalID, U_F, U_P)
    print(f"   → U shape: {U.shape}\n")

    print("Linear elasticity simulation completed successfully.")
    print("=" * 65 + "\n")

    return U


# ----------------------------------------------------------
# Quick self-test (executed only if run directly)
# ----------------------------------------------------------
if __name__ == "__main__":
    print("This module defines Driver_LE() — no standalone test.")


