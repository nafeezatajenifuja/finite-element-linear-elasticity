"""Material constants and the original tensor / Kronecker formulation."""

from dataclasses import dataclass

import numpy as np

from .mesh import FloatArray


@dataclass(frozen=True)
class Material:
    """Effective isotropic Lamé constants for the selected problem reduction."""

    lambda_: float
    mu: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.lambda_) or not np.isfinite(self.mu):
            raise ValueError("material constants must be finite")
        if self.mu <= 0 or self.lambda_ + self.mu <= 0:
            raise ValueError("material requires mu > 0 and lambda + mu > 0")

    @classmethod
    def from_young_poisson(cls, young: float, poisson: float, mode: str = "plane_strain") -> "Material":
        """Convert E and nu using the original plane-stress / plane-strain rules.

        For the 1D rod use nu=0, giving lambda=0 and 2 mu=E. The cylinder
        example uses plane stress; Cook's membrane and the beam use plane strain.
        """
        if not np.isfinite(young) or young <= 0 or not np.isfinite(poisson) or not -1 < poisson < 0.5:
            raise ValueError("require finite E > 0 and -1 < nu < 0.5")
        if mode == "plane_stress":
            lambda_ = young * poisson / (1 - poisson**2)
        elif mode == "plane_strain":
            lambda_ = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
        else:
            raise ValueError("mode must be plane_stress or plane_strain")
        return cls(lambda_, young / (2 * (1 + poisson)))

    def matrix(self) -> FloatArray:
        """Return the 2D constitutive matrix for [eps_xx, eps_yy, gamma_xy]."""
        return np.array([
            [self.lambda_ + 2 * self.mu, self.lambda_, 0],
            [self.lambda_, self.lambda_ + 2 * self.mu, 0],
            [0, 0, self.mu],
        ])


def transpose_matrix(dimension: int) -> FloatArray:
    """Map vec(A) to vec(A.T), the standard commutation operation.

    The original student helper follows the instructor's TMatrix reference.
    This equivalent implementation constructs the transpose permutation directly.
    """
    indices = np.arange(dimension**2).reshape(dimension, dimension).T.ravel()
    return np.eye(dimension**2)[indices]
