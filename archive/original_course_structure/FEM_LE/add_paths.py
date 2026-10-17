# ==========================================================
#  add_paths.py — Updated Project Path Registration
#  Author : Nafeezat Ajenifuja
#  Purpose: Dynamically add all FEM project subdirectories
#           to Python's sys.path 
# ==========================================================

import sys
import os

def add_project_paths(verbose=True):
    """
    Adds all major project folders and physics kernels to Python's sys.path.

    NEW DIRECTORY STRUCTURE HANDLED:
    --------------------------------
    FEM_FRAMEWORK/
    ERROR_ANALYSIS/
    PHYSICS_MODEL/
        KERNEL_DIFFUSION/
        KERNEL_LE/
    DATA_FILES/
        DATA_DIFFUSION/
        DATA_LE/
    TEST_FILES/
        TEST_DIFFUSION/
        TEST_LE/
    """

    # --------------------------------------------------
    # Base project directory (directory containing this file)
    # --------------------------------------------------
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # --------------------------------------------------
    # All subdirectories to add
    # --------------------------------------------------
    subfolders = [
        # FEM core framework
        "FemFramework",
        "Error_Analysis",

        # Physics kernels
        os.path.join("Physics_model", "Kernel_diffusion"),
        os.path.join("Physics_model", "Kernel_LE"),

        # Data files
        os.path.join("DataFiles", "Data_file_diffusion"),
        os.path.join("DataFiles", "Data_file_LE"),

        # Test files
        os.path.join("Tests", "Test_diffusion"),
        os.path.join("Tests", "Test_LE"),
    ]

    # --------------------------------------------------
    # Register each directory
    # --------------------------------------------------
    for folder in subfolders:
        full_path = os.path.join(base_dir, folder)

        if os.path.isdir(full_path):
            if full_path not in sys.path:
                sys.path.append(full_path)
                if verbose:
                    print(f" Added to sys.path: {full_path}")
        else:
            if verbose:
                print(f" Warning: Directory not found — {full_path}")

    if verbose:
        print("\n All FEM project paths successfully added.\n")


# ----------------------------------------------------------
# Quick self-test
# ----------------------------------------------------------
if __name__ == "__main__":
    add_project_paths(verbose=True)



