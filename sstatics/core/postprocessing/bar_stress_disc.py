

from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core.preprocessing import Bar
from sstatics.core.postprocessing import (
    DifferentialEquation, CrossSectionStress)


@dataclass
class BarStressDistribution:
    r"""
    Assemble and evaluate the beam differential equation and compute stresses.

    This class constructs the differential equation of the deflection curve
    (bending line) for a given bar using the bar geometry, deformations and
    internal force vector. The differential equation is discretized into
    `n_disc` segments and evaluated at the corresponding `n_disc + 1`
    positions along the bar. The discretized internal force result
    (`force_disc`) returned by the differential equation solution is then
    used to compute section stresses at each evaluation point using
    `CrossSectionStress`.

    Parameters
    ----------
    bar : :any:`Bar`
        The bar for which the differential equation and stress distribution
        are computed.
    deform : :any:`numpy.ndarray`
        Deformation vector of the bar (shape (6, 1)). These are the nodal
        deformation values from the structural solution used to assemble the
        differential equation.
    force : :any:`numpy.ndarray`
        Internal force vector of the bar (shape (6, 1)). These are the
        nodal/internal forces from the structural solution.
    n_disc : int, default=10
        Number of discretization segments along the bar (number of evaluation
        points = n_disc + 1). The differential equation is evaluated at these
        points.

    Raises
    ------
    ValueError
        If `deform` or `force` does not have shape (6, 1), or if `n_disc < 1`.

    Examples
    --------
    >>> from sstatics.core.preprocessing import
    ...     (Bar, Node, CrossSection, Material)
    >>> # assume system solved and `bar`, `deform`, `force` are available
    >>> stress_dist = BarStressDistribution(bar, deform, force, n_disc=10)
    >>> stress_dist.x.shape
    (11, 1)
    >>> stress_disc = stress_dist.stress_disc
    >>> stress_disc.shape
    (11, 4)
    >>> stress_at_mid = stress_dist.stress_at_z(0.05)
    >>> stress_at_mid.shape
    (11, 3)
    """
    bar: Bar
    deform: np.ndarray
    force: np.ndarray
    n_disc: int = 10

    def __post_init__(self):
        if self.deform.shape != (6, 1):
            raise ValueError('"deform" must have shape (6, 1).')
        if self.force.shape != (6, 1):
            raise ValueError('"force" must have shape (6, 1).')
        if self.n_disc < 1:
            raise ValueError('"n_disc" has to be greater than 0')

        self._dgl = DifferentialEquation(
            bar=self.bar,
            deform=self.deform,
            forces=self.force,
            n_disc=self.n_disc
        )
        self._cs_stress = CrossSectionStress(self.bar.cross_section)

    @cached_property
    def x(self):
        """
        Discretized positions along the bar length.

        Returns
        -------
        :any:`numpy.ndarray`
            Array of shape (n_disc + 1,) containing positions along the bar
            from 0 to bar length.

        Examples
        --------
        >>> from sstatics.core.preprocessing import
        ...     (Bar, Node, CrossSection, Material)
        >>> bar = Bar(...)
        >>> deform = np.zeros((6, 1))
        >>> force = np.zeros((6, 1))
        >>> stress_dist = BarStressDistribution(bar, deform, force, n_disc=10)
        >>> stress_dist.x
        array([0.0, 0.1, 0.2, ..., bar.length])
        """
        return self._dgl.x

    @cached_property
    def force_disc(self):
        """
        Discretized internal forces along the bar.

        Returns
        -------
        :any:`numpy.ndarray`
            Array of shape (n_disc + 1, 3) where columns correspond to:
            0 → axial force N
            1 → shear force V
            2 → bending moment M

        Examples
        --------
        >>> from sstatics.core.preprocessing import
        ...     (Bar, Node, CrossSection, Material)
        >>> bar = Bar(...)
        >>> deform = np.zeros((6, 1))
        >>> force = np.zeros((6, 1))
        >>> stress_dist = BarStressDistribution(bar, deform, force, n_disc=10)
        >>> forces = stress_dist.internal_forces
        >>> forces[:, 0]  # axial forces
        """
        return self._dgl.forces_disc

    @cached_property
    def stress_disc(self):
        """
        Stress distribution at the extreme fibers (top and bottom) along the
        bar.

        Returns
        -------
        :any:`numpy.ndarray`
            Array of shape (n_disc + 1, 4) where columns correspond to:
            0 → normal stress σ_normal
            1 → shear stress τ_shear
            2 → bending stress σ_bending at bottom fiber
            3 → bending stress σ_bending at top fiber

        Notes
        -----
        Uses the cross-section boundary coordinates to compute bending stresses
        at top and bottom fibers.

        Examples
        --------
        >>> from sstatics.core.preprocessing import
        ...     (Bar, Node, CrossSection, Material)
        >>> bar = Bar(...)
        >>> deform = np.zeros((6, 1))
        >>> force = np.zeros((6, 1))
        >>> stress_dist = BarStressDistribution(bar, deform, force, n_disc=10)
        >>> sigma = stress_dist.stress_disc
        >>> sigma.shape
        (11, 4)
        >>> sigma[:, 2]  # bending stress at bottom fiber
        """
        f_x, f_z, f_m = self.force_disc.T
        n_points = len(f_x)
        sigma_matrix = np.zeros((n_points, 4))
        _, zb = self.bar.cross_section.boundary()

        for i in range(n_points):
            sigma_matrix[i] = [
                self._cs_stress.normal_stress(n=f_x[i]),
                self._cs_stress.shear_stress(v_z=f_z[i]),
                self._cs_stress.bending_stress(m_yy=f_m[i], z=zb[0]),
                self._cs_stress.bending_stress(m_yy=f_m[i], z=zb[1])
            ]
        return sigma_matrix

    def stress_at_z(self, z: float):
        """
        Stress distribution at a specified height `z` along the cross-section.

        Parameters
        ----------
        z : float
            Height in the local z-direction where stresses are evaluated.

        Returns
        -------
        :any:`numpy.ndarray`
            Array of shape (n_disc + 1, 3) where columns correspond to:
            0 → normal stress σ_normal
            1 → shear stress τ_shear
            2 → bending stress σ_bending

        Examples
        --------
        >>> from sstatics.core.preprocessing import
        ...     (Bar, Node, CrossSection, Material)
        >>> bar = Bar(...)
        >>> deform = np.zeros((6, 1))
        >>> force = np.zeros((6, 1))
        >>> stress_dist = BarStressDistribution(bar, deform, force, n_disc=10)
        >>> sigma_mid = stress_dist.stress_at_z(0.05)
        >>> sigma_mid.shape
        (11, 3)
        >>> sigma_mid[:, 0]  # normal stress at z = 0.05
        """
        f_x, f_z, f_m = self.force_disc.T
        n_points = len(f_x)
        sigma_matrix = np.zeros((n_points, 3))
        for i in range(n_points):
            sigma_matrix[i] = [
                self._cs_stress.normal_stress(n=f_x[i]),
                self._cs_stress.shear_stress(v_z=f_z[i], z=z),
                self._cs_stress.bending_stress(m_yy=f_m[i], z=z)
            ]
        return sigma_matrix
