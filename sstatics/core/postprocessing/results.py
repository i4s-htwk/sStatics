
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

import numpy as np

from sstatics.core.preprocessing.system import System
from sstatics.core.preprocessing.bar import Bar
from sstatics.core.preprocessing.node import Node


@dataclass
class SystemResult:
    r"""Represents the post-processing results of a structural system.

    This class aggregates discrete bar deformations, internal forces,
    nodal displacements, and support reactions from a solved
    :py:class:`System`. It provides convenient access to per-bar and
    per-node results for further analysis or visualization.

    Parameters
    ----------
    system : :py:class:`System`
        The analyzed structural system.
    bar_deform_list : list of :any:`numpy.ndarray`
        List of deformation vectors for each mesh bar in the system.
        Each deformation array corresponds to a mesh bar in
        :py:attr:`FirstOrder.bar_deform_list`.
    bar_internal_forces : list of :any:`numpy.ndarray`
        List of force vectors for each mesh bar in the system.
        Each force array corresponds to a mesh bar in
        :py:attr:`FirstOrder.bar_internal_forces`.
    node_deform : :any:`numpy.ndarray`
        A vector with dimensions (dof * number of nodes, 1) containing the
        resulting displacement of each node in its local coordinate system
        :py:attr:`FirstOrder.node_deform`.
    node_support_forces : :any:`numpy.ndarray`
        A vector with dimensions (dof * number of nodes, 1) containing the
        support reactions in the nodal coordinate system
        :py:attr:`FirstOrder.node_support_forces`.
    system_support_forces : :any:`numpy.ndarray`
        A vector with dimensions (dof * number of nodes, 1) that contains
        the support reactions referenced to the global coordinate system
        :py:attr:`FirstOrder.system_support_forces`.
    dof : int, default=3
        Number of degrees of freedom per node. Must be 3 for 2D systems
        (translations ux, uz and rotation φ). The value 6 would correspond
        to 3D systems (ux, uy, uz, φx, φy, φz), but 3D is currently not
        implemented.
    n_disc : :any:`int`, default=10
        Number of discrete evaluation points along each bar for discretisation
        of forces and deformations.

    Raises
    ------
    ValueError
        If the number of deformation or force vectors does not match the
        number of bars, if the node vectors have inconsistent shapes,
        or if `dof` or `n_disc` are invalid.
    TypeError
        If any of the input arrays are not instances of `numpy.ndarray`.


    Attributes
    ----------
    bars : list of :py:class:`BarResult`
        List containing the discrete results for each bar.
    nodes : list of :py:class:`NodeResult`
        List containing the displacement and support forces for each node.


    Examples
    --------
    >>> from sstatics.core.preprocessing import (
    >>>     Bar, BarLineLoad, CrossSection, Material, Node, System
    >>> )
    >>> from sstatics.core.solution import FirstOrder
    >>> from sstatics.core.postprocessing import SystemResult
    >>>
    >>> n1 = Node(0, 0, u='fixed', w='fixed')
    >>> n2 = Node(4, 0, w='fixed')
    >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
    >>> mat = Material(210000000, 0.1, 81000000, 0.1)
    >>> load = BarLineLoad(1, 1, 'z', 'bar', 'exact')
    >>> bar = Bar(n1, n2, cross, mat, line_loads=load)
    >>> system = System([bar])
    >>> solution = FirstOrder(system)
    >>> system_result = SystemResult(
    >>>                     system=system,
    >>>                     bar_deform_list=solution.bar_deform_list,
    >>>                     bar_internal_forces=solution.internal_forces,
    >>>                     node_deform=solution.node_deform,
    >>>                     node_support_forces=solution.node_support_forces,
    >>>                     system_support_forces=
    >>>                         solution.system_support_forces
    >>> )
    >>> for bar in system_result.bars:
    >>>     print(bar.x_coef)
    array([[ 0. -0.  0.]
           [ 0. -0. -0.]
           [ 0. -0.  0.]
           [ 0.  0.  0.]])
    """

    system: System
    bar_deform_list: list[np.ndarray]
    bar_internal_forces: list[np.ndarray]
    node_deform: np.ndarray
    node_support_forces: np.ndarray
    system_support_forces: np.ndarray
    dof: Literal[3, 6] = 3
    n_disc: int = 10

    def __post_init__(self):
        self._validation()
        self.bars = [
            BarResult(bar,
                      self.bar_deform_list[i],
                      self.bar_internal_forces[i],
                      self.n_disc)
            for i, bar in enumerate(self.system.mesh)
        ]
        self.nodes = [
            NodeResult(node,
                       self.node_deform[
                            i * self.dof: i * self.dof + self.dof],
                       self.node_support_forces[
                            i * self.dof: i * self.dof + self.dof],
                       self.system_support_forces[
                            i * self.dof: i * self.dof + self.dof])
            for i, node in enumerate(self.system.nodes())
        ]

    def _validation(self):
        # DOF check
        if self.dof not in (3, 6):
            raise ValueError(
                f'"dof" must be either 3 (2D space) or 6 (3D space),'
                f'got {self.dof}.'
            )
        if self.dof == 6:
            raise ValueError(
                '"dof=6" (3D systems) is not yet implemented.'
            )
        if self.n_disc < 1:
            raise ValueError('"n_disc" must be a positive integer.')

        # Check list lengths
        n_bars = len(self.system.mesh)
        if len(self.bar_deform_list) != n_bars:
            raise ValueError(
                f'Expected {n_bars} bar deformations, got '
                f'{len(self.bar_deform_list)}.'
            )
        if len(self.bar_internal_forces) != n_bars:
            raise ValueError(
                f'Expected {n_bars} bar force vectors, got '
                f'{len(self.bar_internal_forces)}.'
            )

        # Type and shape validation for lists
        for i, arr in enumerate(self.bar_deform_list):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f'bar_deform_list[{i}] must be a numpy.ndarray.'
                )
        for i, arr in enumerate(self.bar_internal_forces):
            if not isinstance(arr, np.ndarray):
                raise TypeError(
                    f'bar_internal_forces[{i}] must be a numpy.ndarray.'
                )

        # Shape checks for node-related arrays
        shapes = [self.node_deform.shape,
                  self.node_support_forces.shape,
                  self.system_support_forces.shape]
        if not all(s == shapes[0] for s in shapes):
            raise ValueError(
                f'"node_deform", "node_support_forces", '
                f'and "system_support_forces" must all have the same shape, '
                f'got {shapes}.'
            )

        n_nodes = len(self.system.nodes(mesh_type="mesh"))
        expected_size = n_nodes * self.dof
        if self.node_deform.shape[0] != expected_size:
            raise ValueError(
                f'Expected {expected_size} rows for {n_nodes} nodes × '
                f'{self.dof} DOF, but got {self.node_deform.shape[0]}.'
            )

        if self.node_deform.ndim != 2 or self.node_deform.shape[1] != 1:
            raise ValueError(
                f'"node_deform" must be a column vector of shape '
                f'({expected_size}, 1), but got {self.node_deform.shape}.'
            )

    @cached_property
    def deforms_disc(self):
        r"""Discrete deformation vectors evaluated at discrete points.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the deformation vectors evaluated along
            the length of the corresponding mesh bar.

        See Also
        --------
        :py:attr:`BarResult.deform_disc`
        """
        return [result.deform_disc for result in self.bars]

    @cached_property
    def forces_disc(self):
        r"""Discrete force vectors evaluated at discrete points.

        Returns
        -------
        list of :any:`numpy.ndarray`
            Each array contains the force vectors evaluated along
            the length of the corresponding mesh bar.

        See Also
        --------
        :py:attr:`BarResult.forces_disc`
        """
        return [result.forces_disc for result in self.bars]


@dataclass
class BarResult:
    r"""Calculates discrete result vector for the provided bar.

    Parameters
    ----------
    bar : :any:`Bar`
        The bar that is to be discretised.
    deform : :any:`numpy.ndarray`
        The deformation vector of the bar with shape (6, 1).
    forces : :any:`numpy.ndarray`
        The force vector of the bar with shape (6, 1).
    n_disc : :any:`int`, default=10
        Number of discrete points along the bar for discretisation.

    Raises
    ------
    ValueError
        :py:attr:`deform` and :py:attr:`forces` need to have a shape
        of (6, 1)
    ValueError
        :py:attr:`n_disc` has to be greater than zero.

    Examples
    --------
    >>> from sstatics.core (
    >>>     Bar, BarLineLoad, BarResult, CrossSection, Material, Node, System,
    >>>     FirstOrder
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
    >>> bar_result = BarResult(bar, results[0][0], results[1][0], n_disc=10)
    """

    bar: Bar
    deform: np.ndarray
    forces: np.ndarray
    n_disc: int = 10

    def __post_init__(self):
        if self.deform.shape != (6, 1):
            raise ValueError('"deform" must have shape (6, 1).')
        if self.forces.shape != (6, 1):
            raise ValueError('"forces" must have shape (6, 1).')
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

            :math:`p_{i,x}, p_{j,x}`: Distributed load in local x-direction
            at the start and end of the bar.

        The difference between the load values is:

        .. math::
            dp_x = p_{j,x} - p_{i,x}

        The polynomial coefficients in the local x-direction are calculated
        using the following matrix equation:

        .. math::
            a_{x} =
            \begin{bmatrix}
                p_{i,x} & -N & u \\
                \dfrac{dp_x}{l} & -p_{i,x} & -\dfrac{N}{EA} \\
                0 & -\dfrac{dp_x}{2l} & \dfrac{p_{i,x}}{2EA} \\
                0 & 0 & \dfrac{dp_x}{6lEA}
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
        p_ix, p_jx = self.bar.line_load[0][0], self.bar.line_load[3][0]
        n, u = self.forces[0][0], self.deform[0][0]
        dp_x = p_jx - p_ix
        return np.array([
            [p_ix, -n, u],
            [dp_x / l, -p_ix, -n / EA],
            [0, -dp_x / (2 * l), p_ix / (2 * EA)],
            [0, 0, dp_x / (6 * l * EA)]
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

            :math:`p_{i,z}, p_{j,z}`: Distributed load in the local
            z-direction at the start and end of the bar.

        The difference between the load values is:

        .. math::
            dp_z = p_{j,z} - p_{i,z}

        The polynomial coefficients in the local z-direction are calculated
        using the following matrix equation:

        .. math::
            a_{z} =
            \begin{bmatrix}
                p_{i,z} & -V & -M & \varphi & w \\
                \dfrac{dp_z}{l} & -p_{i,z} & -V & -\dfrac{M}{EI} &
                -\varphi \\
                0 & -\dfrac{dp_z}{2l} & -\dfrac{p_{i,z}}{2} & -\dfrac{V}{2EI}
                & \dfrac{M}{2EI} \\
                0 & 0 & -\dfrac{dp_z}{6l} & -\dfrac{p_{i,z}}{6EI} &
                \dfrac{V}{6EI} \\
                0 & 0 & 0 & -\dfrac{dp_z}{24lEI} & \dfrac{p_{i,z}}{24EI} \\
                0 & 0 & 0 & 0 & \dfrac{dp_z}{120lEI}
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
        p_iz, p_jz = self.bar.line_load[1][0], self.bar.line_load[4][0]
        v, m = self.forces[1][0], self.forces[2][0]
        w, phi = self.deform[1][0], self.deform[2][0]
        dp_z = p_jz - p_iz
        return np.array([
            [p_iz, -v, -m, phi, w],
            [dp_z / l, -p_iz, -v, -m / EI, -phi],
            [0, -dp_z / (2 * l), -p_iz / 2, -v / (2 * EI), m / (2 * EI)],
            [0, 0, -dp_z / (6 * l), -p_iz / (6 * EI), v / (6 * EI)],
            [0, 0, 0, -dp_z / (24 * l * EI), p_iz / (24 * EI)],
            [0, 0, 0, 0,  dp_z / (120 * l * EI)]
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
    def deform_disc(self):
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
        >>> deforms = bar_result.deform_disc

        The values can be accessed as:

        >>> u_vals = deforms[:, 0]     # axial displacement
        >>> w_vals = deforms[:, 1]     # transverse displacement
        >>> phi_vals = deforms[:, 2]   # slope
        """
        coef = [self.x_coef[:, 2], self.z_coef[:, 4], self.z_coef[:, 3]]
        return np.vstack([self._eval_poly(c) for c in coef]).T

    @cached_property
    def forces_disc(self):
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
        >>> forces = bar_result.forces_disc

        The values can be accessed as:

        >>> n_vals = forces[:, 0]  # normal force
        >>> v_vals = forces[:, 1]  # shear force
        >>> m_vals = forces[:, 2]  # moment
        """
        coef = [self.x_coef[:, 1], self.z_coef[:, 1], self.z_coef[:, 2]]
        return np.vstack([self._eval_poly(c) for c in coef]).T


@dataclass
class NodeResult:
    r"""Represents the result vectors associated with a single node.

    This class stores the displacement and support reaction vectors
    for a given node, both in the nodal and global coordinate system.

    Parameters
    ----------
    node : :py:class:`Node`
        The node for which the results are stored.
    deform : numpy.ndarray
        Displacement vector of the node in its local coordinate system,
        with shape (3, 1).
    node_support : numpy.ndarray
        Support reaction vector in the nodal coordinate system,
        with shape (3, 1).
    system_support : numpy.ndarray
        Support reaction vector in the global coordinate system,
        with shape (3, 1).

    Raises
    ------
    TypeError
        If ``node`` is not an instance of :py:class:`Node` or any of the
        vectors are not ``numpy.ndarray``.
    ValueError
        If any of the vectors do not have shape (3, 1) or contain
        invalid values (NaN or infinite).
    TypeError
        If any of the vectors are not of type ``float32`` or ``float64``.

    Attributes
    ----------
    node : :py:class:`Node`
        Reference to the corresponding node object.
    deform : numpy.ndarray
        Local displacement vector of the node.
    node_support : numpy.ndarray
        Support reactions in the nodal coordinate system.
    system_support : numpy.ndarray
        Support reactions in the global coordinate system.

    Examples
    --------
    >>> from sstatics.core.preprocessing import Node
    >>> import numpy as np
    >>>
    >>> n1 = Node(0, 0, u="fixed", w="fixed", phi='fixed')
    >>> deform = np.zeros((3, 1))
    >>> node_support = np.array([[0.0], [10.0], [0.0]])
    >>> system_support = np.array([[0.0], [10.0], [0.0]])
    >>> node_result = NodeResult(
    ...     node=n1,
    ...     deform=deform,
    ...     node_support=node_support,
    ...     system_support=system_support
    ... )
    >>> node_result.deform
    array([[0.],
           [0.],
           [0.]])
    """

    node: Node
    deform: np.ndarray
    node_support: np.ndarray
    system_support: np.ndarray

    def __post_init__(self):
        self._validation()

    def _validation(self):
        if not isinstance(self.node, Node):
            raise TypeError('"node" must be an instance of Node.')

        for name, arr in [
            ("deform", self.deform),
            ("node_support", self.node_support),
            ("system_support", self.system_support),
        ]:
            if not isinstance(arr, np.ndarray):
                raise TypeError(f'"{name}" must be a numpy.ndarray.')
            if arr.shape != (3, 1):
                raise ValueError(
                    f'"{name}" must have shape (3, 1), got {arr.shape}.')
            if not np.isfinite(arr).all():
                raise ValueError(f'"{name}" contains NaN or infinite values.')
            if arr.dtype not in (np.float32, np.float64):
                raise TypeError(
                    f'"{name}" must be of type float32 or float64, '
                    f'got {arr.dtype}.')


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
    def length_disc(self):
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
        powers = np.vander(self.length_disc, N=len(coef), increasing=True)
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
