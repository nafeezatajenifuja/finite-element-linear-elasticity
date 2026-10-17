"""Course implementation of small-strain, quasi-static isotropic elasticity."""

from .elasticity import Material
from .mesh import Mesh
from .solver import solve

__all__ = ["Material", "Mesh", "solve"]
