
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core.preprocessing import Bar


@dataclass
class RigidBodyDisplacement:
    bar: Bar
    deform: np.ndarray
    n_disc: int = 10

    def __post_init__(self):
        if self.deform.shape != (6, 1):
            raise ValueError('"deform" must have shape (6, 1).')
        if self.n_disc < 1:
            raise ValueError('"n_disc" has to be greater than 0')

    @cached_property
    def x(self):
        """Discrete positions along the bar length for evaluation."""
        return np.linspace(0, self.bar.length, self.n_disc + 1)

    @cached_property
    def x_coef(self):
        """Coefficients for the rigid axial displacement u(x)."""
        length = self.bar.length
        u_i, u_j = self.deform[0, 0], self.deform[3, 0]
        coef = np.zeros((6, 1))
        coef[0] = u_i
        coef[1] = (u_j - u_i) / length
        return coef  # Only terms up to linear order

    @cached_property
    def z_coef(self):
        """Coefficients for the rigid transverse displacement w(x)."""
        length = self.bar.length
        w_i, w_j = self.deform[1, 0], self.deform[4, 0]
        coef = np.zeros((6, 1))
        coef[0] = w_i
        coef[1] = (w_j - w_i) / length
        return coef  # Only terms up to linear order

    def _eval_poly(self, coef: np.ndarray):
        """Evaluate a polynomial at all discrete positions along the bar."""
        powers = np.vander(self.x, N=len(coef), increasing=True)
        return powers @ coef

    @cached_property
    def deform_disc(self):
        """Evaluate rigid body displacements along the bar.

        Returns
        -------
        :any:`numpy.ndarray`
        A (n, 2) array, where:

                - n is the number of discretization points + 1 (`n_disc` + 1),
                - Column 0 contains the axial displacement `u(x)`,
                - Column 1 contains the transverse displacement `w(x)`.

            u(x) = deform_disc[:, 0]
            w(x) = deform_disc[:, 1]
        """
        coef = [self.x_coef[:, 0], self.z_coef[:, 0]]
        return np.vstack([self._eval_poly(c) for c in coef]).T
