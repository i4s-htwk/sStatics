
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core.preprocessing import System, Bar


@dataclass
class SystemResult:
    r"""Calculates discrete result vectors for the bars of the provided system.

    Parameters
    ----------
    system : :py:class:`System`
        The statical system that was analyzed.
    deforms : list of :any:`numpy.ndarray`
        List of deformation vectors for each segmented bar in the system.
        Each deformation array corresponds to a segmented bar in
        :py:attr:`system.segmented_bars`.
    forces : list of :any:`numpy.ndarray`
        List of force vectors for each segmented bar in the system.
        Each force array corresponds to a segmented bar in
        :py:attr:`system.segmented_bars`.
    n_disc : :any:`int`, default=10
        Number of discrete evaluation points along each bar for discretisation
        of forces and deformations.

    Raises
    ------
    ValueError
        If the length of :py:attr:`deforms` or :py:attr:`forces` does not match
        the number of segmented bars in :py:attr:`system`.

    Attributes
    ----------
    bars : list of :py:class:`BarResult`
        List of each segmented bar with discrete results.

    Examples
    --------
    >>> from sstatics.core import (
    >>>     Bar, BarLineLoad, CrossSection, FirstOrder, Material, Node, System,
    >>>     SystemResult
    >>> )
    >>> n1 = Node(0, 0, u='fixed', w='fixed')
    >>> n2 = Node(4, 0, w='fixed')
    >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
    >>> mat = Material(210000000, 0.1, 81000000, 0.1)
    >>> load = BarLineLoad(1, 1, 'z', 'bar', 'exact')
    >>> bar = Bar(n1, n2, cross, mat, line_loads=load)
    >>> system = System([bar])
    >>> fo = FirstOrder(system)
    >>> results = fo.calc
    >>> system_result = SystemResult(system, results[0], results[1], n_disc=10)
    >>> for bar in system_result.bars:
    >>>     print(bar.x_coef)
    array([[ 0. -0.  0.]
           [ 0. -0. -0.]
           [ 0. -0.  0.]
           [ 0.  0.  0.]])
    """

    system: System
    deforms: list[np.ndarray]
    forces: list[np.ndarray]
    n_disc: int = 10

    def __post_init__(self):
        if len(self.system.segmented_bars) != len(self.deforms):
            raise ValueError(
                'The number of bars in "system.segmented_bars" does not match '
                'the number of entries in "deforms".'
            )
        if len(self.system.segmented_bars) != len(self.forces):
            raise ValueError(
                'The number of bars in "system.segmented_bars" does not match '
                'the number of entries in "forces".'
            )
        self.bars = [
            BarResult(bar, self.deforms[i], self.forces[i], self.n_disc)
            for i, bar in enumerate(self.system.segmented_bars)
        ]

    @cached_property
    def length_disc(self):
        r"""Discrete evaluation points along the length of each segmented bar.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the discrete coordinate points along the
            corresponding bar where results are evaluated.

        See Also
        --------
        :py:attr:`length_disc` in :py:class:`BarResult`

        Examples
        --------
        >>> system_result = SystemResult(...)
        >>> system_result.length_disc
        """
        return [result.length_disc for result in self.bars]

    @cached_property
    def deforms_disc(self):
        r"""Discrete deformation vectors evaluated at discrete points.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the deformation vectors evaluated along
            the length of the corresponding segmented bar.

        See Also
        --------
        :py:attr:`bar_deforms_disc` in :py:class:`BarResult`
        """
        return [result.bar_deforms_disc for result in self.bars]

    @cached_property
    def forces_disc(self):
        r"""Discrete force vectors evaluated at discrete points.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the force vectors evaluated along
            the length of the corresponding segmented bar.

        See Also
        --------
        :py:attr:`bar_forces_disc` in :py:class:`BarResult`
        """
        return [result.bar_forces_disc for result in self.bars]

    @cached_property
    def system_results_disc(self):
        r"""Tuple of deformation and force vectors evaluated at discrete
        points.

        Returns
        -------
        tuple of lists
            Tuple containing two lists:
                - List of deformation arrays evaluated at discrete points.
                - List of force arrays evaluated at discrete points.

        Examples
        --------
        >>> system_result = SystemResult(...)
        >>> deforms, forces = system_result.system_results_disc
        """
        return self.deforms_disc, self.forces_disc


@dataclass
class BarResult:
    r"""Calculates discrete result vector for the provided bar.

    Parameters
    ----------
    bar : :any:`Bar`
        The bar that is to be discretised.
    bar_deforms : :any:`numpy.ndarray`
        The deformation vector of the bar with shape (6, 1).
    bar_forces : :any:`numpy.ndarray`
        The force vector of the bar with shape (6, 1).
    n_disc : :any:`int`, default=10
        Number of discrete points along the bar for discretisation.

    Raises
    ------
    ValueError
        :py:attr:`bar_deforms` and :py:attr:`bar_forces` need to have a shape
        of (6, 1)
    ValueError
        :py:attr:`n_disc` has to be greater than zero.

    Examples
    --------
    >>> from sstatics.core.preprocessing.bar import Bar
    >>> from sstatics.core.preprocessing.loads import BarLineLoad
    >>> from sstatics.core.preprocessing.cross_section import CrossSection
    >>> from sstatics.core.preprocessing.material import Material
    >>> from sstatics.core.preprocessing.node import Node
    >>> from sstatics.core.preprocessing.system import System
    >>> from sstatics.core.solution.methods import FirstOrder
    >>> from sstatics.core.postprocessing.results import BarResult
    >>> n1 = Node(0, 0, u='fixed', w='fixed')
    >>> n2 = Node(4, 0, w='fixed')
    >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
    >>> mat = Material(210000000, 0.1, 81000000, 0.1)
    >>> load = BarLineLoad(1, 1, 'z', 'bar', 'exact')
    >>> bar = Bar(n1, n2, cross, mat, line_loads=load)
    >>> system = System([bar])
    >>> fo = FirstOrder(system)
    >>> results = fo.calc
    >>> bar_result = BarResult(bar, results[0][0], results[1][0], n_disc=10)
    """

    bar: Bar
    bar_deforms: np.ndarray
    bar_forces: np.ndarray
    n_disc: int = 10

    def __post_init__(self):
        if self.bar_deforms.shape != (6, 1):
            raise ValueError('"bar_deforms" must have shape (6, 1).')
        if self.bar_forces.shape != (6, 1):
            raise ValueError('"bar_forces" must have shape (6, 1).')
        if self.n_disc < 1:
            raise ValueError('"n_disc" has to be greater than 0')

    @cached_property
    def length_disc(self):
        r"""Discrete positions along the bar length for evaluation.

        Returns
        -------
        :any:`numpy.ndarray`
            Linearly spaced coordinates from 0 to bar length with
            :py:attr:`n_disc` + 1 points.

        Examples
        --------
        >>> bar_result = BarResult(...)
        >>> bar_result.length_disc
        array([0.  0.4 0.8 1.2 1.6 2.  2.4 2.8 3.2 3.6 4. ])
        """
        return np.linspace(0, self.bar.length, self.n_disc + 1)

    @cached_property
    def x_coef(self):
        r"""Polynomial coefficients for the differential equation in the local
        x-direction.

        This method returns the coefficient matrix for evaluating internal
        quantities in the x-direction using a degree-3 polynomial. The matrix
        has shape (4, 3), where each **row** corresponds to a polynomial term
        of increasing degree (constant up to cubic), and each **column**
        represents a physical quantity.

        Returns
        -------
        :any:`numpy.ndarray`
            A (4, 3) matrix of polynomial coefficients.

        Notes
        -----
        The following variables are used:

            :math:`l`: Length of the bar.

            :math:`EA`: Axial stiffness.

            :math:`N`: Normal force at the start of bar.

            :math:`u`: Axial displacement at the start of bar.

            :math:`p0_{i,x}, p0_{j,x}`: Distributed load in local x-direction
            at the start and end of the bar.

        The difference between the load values is:

        .. math::
            dp0_x = p0_{j,x} - p0_{i,x}

        The polynomial coefficients in the local x-direction are calculated
        using the following matrix equation:

        .. math::
            a_{x} =
            \begin{bmatrix}
                p0_{i,x} & -N & u \\
                \dfrac{dp0_x}{l} & -p0_{i,x} & -\dfrac{N}{EA} \\
                0 & -\dfrac{dp0_x}{2l} & \dfrac{p0_{i,x}}{2EA} \\
                0 & 0 & \dfrac{dp0_x}{6lEA}
            \end{bmatrix}

        The matrix columns correspond to the following physical quantities:

        1. Linearly distributed load in local x-direction, :math:`p_x(x)`
        2. Normal force along the bar axis, :math:`N(x)`
        3. Axial deformation in local x-direction, :math:`u(x)`

        The resulting coefficients allow pointwise evaluation of these
        quantities along the bar axis in local x-direction.


        Examples
        --------
        Polynomial for linearly distributed load in local x-direction
        >>> bar_result = BarResult(...)
        >>> bar_result.x_coef[:, 0]
        array([ 0.  0.  0.  0.])

        This represents the polynomial: :math:`p(x) = 0`
        """
        l, EA = self.bar.length, self.bar.EA
        p0_ix, p0_jx = self.bar.line_load[0][0], self.bar.line_load[3][0]
        n, u = self.bar_forces[0][0], self.bar_deforms[0][0]
        dp0_x = p0_jx - p0_ix
        return np.array([
            [p0_ix, -n, u],
            [dp0_x / l, -p0_ix, -n / EA],
            [0, -dp0_x / (2 * l), p0_ix / (2 * EA)],
            [0, 0, dp0_x / (6 * l * EA)]
        ])

    @cached_property
    def z_coef(self):
        r"""Polynomial coefficients for the differential equation in the local
        z-direction.

        This method returns the coefficient matrix for evaluating internal
        quantities in the z-direction using a degree-5 polynomial. The
        matrix has shape (6, 5), where each **row** corresponds to a polynomial
        term of increasing degree (constant up to quintic), and each **column**
        represents a physical quantity.

        Returns
        -------
        :any:`numpy.ndarray`
            A (6, 5) matrix of polynomial coefficients.

        Notes
        -----
        The following variables are used:

            :math:`l`: Length of the bar.

            :math:`EI`: Flexural stiffness.

            :math:`V, M`: Shear force and bending moment at the start of bar.

            :math:`w, \varphi`: Transverse displacement and slope at the start
            of bar.

            :math:`p0_{i,z}, p0_{j,z}`: Distributed load in the local
            z-direction at the start and end of the bar.

        The difference between the load values is:

        .. math::
            dp0_z = p0_{j,z} - p0_{i,z}

        The polynomial coefficients in the local z-direction are calculated
        using the following matrix equation:

        .. math::
            a_{z} =
            \begin{bmatrix}
                p0_{i,z} & -V & -M & \varphi & w \\
                \dfrac{dp0_z}{l} & -p0_{i,z} & -V & -\dfrac{M}{EI} &
                -\varphi \\
                0 & -\dfrac{dp0_z}{2l} & -\dfrac{p0_{i,z}}{2} & -\dfrac{V}{2EI}
                & \dfrac{M}{2EI} \\
                0 & 0 & -\dfrac{dp0_z}{6l} & -\dfrac{p0_{i,z}}{6EI} &
                \dfrac{V}{6EI} \\
                0 & 0 & 0 & -\dfrac{dp0_z}{24lEI} & \dfrac{p0_{i,z}}{24EI} \\
                0 & 0 & 0 & 0 & \dfrac{dp0_z}{120lEI}
            \end{bmatrix}

        The matrix columns correspond to the following physical quantities:

        1. Linearly distributed load in local z-direction, :math:`p_z(x)`
        2. Shear force in local z-direction, :math:`V(x)`
        3. Bending moment about the y-axis, :math:`M(x)`
        4. Slope of the deflection curve in z-direction, :math:`\varphi(x)`
        5. Transverse deformation in local z-direction, :math:`w(x)`

        The resulting coefficients allow pointwise evaluation of these
        quantities along the bar axis in local z-direction.


        Examples
        --------
        Polynomial for the shear force function in local z-direction:

        >>> bar_result = BarResult(...)
        >>> bar_result.z_coef[:, 1]
        array([ 2. -1. -0.  0.  0.  0.])

        This represents the polynomial: :math:`V(x) = 2 - 1 x`


        Polynomial for the slope of the deflection curve in z-direction

        >>> bar_result = BarResult(...)
        >>> bar_result.z_coef[:, 3]
        array([-4.58592008e-04 -0.00000000e+00  1.71972003e-04 -2.86620005e-05
        -0.00000000e+00  0.00000000e+00])

        This represents the polynomial: :math:`\varphi(x) = -4.59 \cdot 10^{-4}
        + 1.72 \cdot 10^{-4} x^2 - 2.87 \cdot 10^{-5} x^3`

        """
        l, EI = self.bar.length, self.bar.EI
        p0_iz, p0_jz = self.bar.line_load[1][0], self.bar.line_load[4][0]
        v, m = self.bar_forces[1][0], self.bar_forces[2][0]
        w, phi = self.bar_deforms[1][0], self.bar_deforms[2][0]
        dp0_z = p0_jz - p0_iz
        return np.array([
            [p0_iz, -v, -m, phi, w],
            [dp0_z / l, -p0_iz, -v, -m / EI, -phi],
            [0, -dp0_z / (2 * l), -p0_iz / 2, -v / (2 * EI), m / (2 * EI)],
            [0, 0, -dp0_z / (6 * l), -p0_iz / (6 * EI), v / (6 * EI)],
            [0, 0, 0, -dp0_z / (24 * l * EI), p0_iz / (24 * EI)],
            [0, 0, 0, 0,  dp0_z / (120 * l * EI)]
        ])

    def _eval_poly(self, coef: np.ndarray):
        r"""
        Evaluate a vectorised polynomial using the given coefficient matrix.

        This method computes the values of result vectors (e.g., internal
        forces, deformations) along the bar by evaluating polynomials defined
        by the coefficient matrices `coef_x` and `coef_z´.

        Parameters
        ----------
        coef : np.ndarray
            Coefficient matrix of shape (n, m), where:
            - n is the polynomial degree + 1 (number of terms, i.e., constant
            up to degree n-1),
            - m is the number of different physical quantities (e.g.,
            [load, force, deformation]).

        Returns
        -------
        np.ndarray
            Array of shape (k, m), where:
            - k is the number of discretisation points + 1 along the bar
            (`len(self.length_disc)`),
            - m is the number of different physical quantities.

            Each row corresponds to a position along the bar,
            each column to a physical quantity.

        Notes
        -----
        This method uses the Vandermonde matrix to evaluate the polynomial at
        all discretized positions along the bar (given by `self.length_disc`).
        It supports multiple quantities simultaneously using matrix
        multiplication.

        Examples
        --------
        If you have a cubic polynomial (degree 3) for 3 quantities, the `coef`
        matrix has shape (4, 3). Then, for 20 evaluation points, the result
        will be a (20, 3) array.
        """
        powers = np.vander(self.length_disc, N=len(coef), increasing=True)
        return powers @ coef

    @cached_property
    def bar_deforms_disc(self):
        r"""Evaluate displacement-related result vectors along the bar.

        This method computes the axial displacement `u(x)`, transverse
        displacement `w(x)`, and slope `φ(x)` at discretized positions
        along the bar. The values are determined by analysing the
        corresponding polynomial functions from the coefficient matrices.

        Returns
        -------
        :any:`numpy.ndarray`
            A (n, 3) array, where:
                - n is the number of evaluation points + 1 (`n_disc` + 1),
                - Column 0 contains the axial displacement `u(x)`,
                - Column 1 contains the transverse displacement `w(x)`,
                - Column 2 contains the slope `φ(x)`.

        Notes
        -----
        The polynomials are evaluated using `self._eval_poly`, which uses
        a Vandermonde matrix for efficient vectorized evaluation.

        See Also
        --------
        :py:meth:`_eval_poly`

        Examples
        --------
        >>> bar_result = BarResult(...)
        >>> deforms = bar_result.bar_deforms_disc

        The values can be accessed as:

        >>> u_vals = deforms[:, 0]     # axial displacement
        >>> w_vals = deforms[:, 1]     # transverse displacement
        >>> phi_vals = deforms[:, 2]   # slope
        """
        coef = [self.x_coef[:, 2], self.z_coef[:, 4], self.z_coef[:, 3]]
        return np.vstack([self._eval_poly(c) for c in coef]).T

    @cached_property
    def bar_forces_disc(self):
        r"""
        Evaluate internal force result vectors along the bar.

        This method computes the normal force `N(x)`, shear force `V(x)`,
        and bending moment `M(x)` at discretized positions along the bar.
        The values are determined by analysing the corresponding polynomial
        functions from the coefficient matrices.

        Returns
        -------
        :any:`numpy.ndarray`
            A (n, 3) array, where:
                - n is the number of evaluation points + 1 (`n_disc` + 1),
                - Column 0 contains the normal force `N(x)`,
                - Column 1 contains the shear force `V(x)`,
                - Column 2 contains the bending moment `M(x)`.

        Notes
        -----
        The polynomials are evaluated using `self._eval_poly`, which uses
        a Vandermonde matrix for efficient vectorized evaluation.

        See Also
        --------
        :py:meth:`_eval_poly`

        Examples
        --------
        >>> bar_result = BarResult(...)
        >>> forces = bar_result.bar_forces_disc

        The values can be accessed as:

        >>> n_vals = forces[:, 0]  # normal force
        >>> v_vals = forces[:, 1]  # shear force
        >>> m_vals = forces[:, 2]  # moment
        """
        coef = [self.x_coef[:, 1], self.z_coef[:, 1], self.z_coef[:, 2]]
        return np.vstack([self._eval_poly(c) for c in coef]).T
