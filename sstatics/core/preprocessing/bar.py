
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np

from sstatics.core.preprocessing.cross_section import CrossSection
from sstatics.core.preprocessing.material import Material
from sstatics.core.preprocessing.node import Node
from sstatics.core.preprocessing.loads import (
    BarLineLoad, BarPointLoad
)
from sstatics.core.preprocessing.temperature import BarTemp
from sstatics.core.utils import transformation_matrix
from sstatics.core.logger_mixin import LoggerMixin


@dataclass(eq=False)
class Bar(LoggerMixin):
    """Create a bar for a statical system.

     Parameters
    ----------
    node_i : :any:`Node`
        Node at the start of the bar.
    node_j : :any:`Node`
        Node at the end of the bar.
    cross_section : :any:`CrossSection`
        Cross-sectional properties of the bar.
    material : :any:`Material`
        Material properties of the bar.
    hinge_u_i : :any:`bool`, default=False
        Normal hinge at the start of the bar.
    hinge_w_i : :any:`bool`, default=False
        Shear hinge at the start of the bar.
    hinge_phi_i : :any:`bool`, default=False
        Moment hinge at the start of the bar.
    hinge_u_j : :any:`bool`, default=False
        Normal hinge at the end of the bar.
    hinge_w_j : :any:`bool`, default=False
        Shear hinge at the end of the bar.
    hinge_phi_j : :any:`bool`, default=False
        Moment hinge at the end of the bar.
    deformations : :any:`tuple` | :any:`list`, default=('moment', 'normal')
        Deformation components considered in the calculation.
        Valid options: "moment", "normal", "shear".
    line_loads : :any:`tuple` | :any:`list`, default=()
        Distributed loads acting on the bar.
    temp : :any:`BarTemp`, default=(BarTemp(0, 0))
        Temperature loads acting on the bar.
    point_loads : :any:`tuple` | :any:`list`, default=()
        Point loads acting on the bar.

    Raises
    ------
    ValueError
        :py:attr:`node_i` and :py:attr:`node_j` need to have different
        locations.
    ValueError
        There has to be at least one deformation component.
    ValueError
        Valid deformation keywords are "moment", "normal" and "shear".
    More discription...
    """
    """ TODO """

    node_i: Node
    node_j: Node
    cross_section: CrossSection
    material: Material
    # TODO: what if all hinges are set to True? Problem for calculations or
    # TODO: representation?
    hinge_u_i: bool = False
    hinge_w_i: bool = False
    hinge_phi_i: bool = False
    hinge_u_j: bool = False
    hinge_w_j: bool = False
    hinge_phi_j: bool = False
    deformations: (
        tuple[Literal['moment', 'normal', 'shear'], ...] |
        list[Literal['moment', 'normal', 'shear']] |
        Literal['moment', 'normal', 'shear']
    ) = ('moment', 'normal')
    line_loads: tuple[BarLineLoad, ...] | list[BarLineLoad] | BarLineLoad = ()
    temp: BarTemp = field(default_factory=lambda: BarTemp(0, 0))
    point_loads: (
        tuple[BarPointLoad, ...] | list[BarPointLoad] | BarPointLoad
    ) = ()
    debug: bool = False

    # TODO: other validations? validate hinges
    def __post_init__(self):
        if self.node_i.same_location(self.node_j):
            raise ValueError(
                'node_i and node_j need to have different locations.'
            )
        # TODO: find a solution for this edge case
        if len(self.deformations) == 0:
            raise ValueError('There has to be at least one deformation.')
        if isinstance(self.line_loads, BarLineLoad):
            self.line_loads = self.line_loads,
        self.line_loads = tuple(self.line_loads)
        if isinstance(self.point_loads, BarPointLoad):
            self.point_loads = self.point_loads,
        self.point_loads = tuple(self.point_loads)
        if isinstance(self.deformations, str):
            self.deformations = self.deformations,
        self.deformations = tuple(self.deformations)
        for d in self.deformations:
            if d not in ('moment', 'normal', 'shear'):
                raise ValueError(
                    'Valid deformation key words are "moment", "normal" and '
                    '"shear".'
                )

    def transformation_matrix(self, to_node_coord: bool = True):
        r"""Create a 6x6 rotation matrix based on the bar's inclination
        and node components.

        Parameters
        ----------
        to_node_coord : :any:`bool`, default=True
            Determines whether a transformation into the node coordinate system
            takes place.
            :python:`True` if transformation to the node coordinate system is
            applied.
            :python:`False` if the transformation is not applied.

        Returns
        -------
        :any:`numpy.array`
            A 6x6 matrix for rotating 6x1 vectors.

        Notes
        -----
            If the transformation to the node coordinate system is not applied,
            the rotation angle is determined solely by the inclination of the
            bar:
            :python:`alpha = bar.inclination`.

            If the transformation to the node coordinate system is applied,
            the rotation is calculated by subtracting the node rotations from
            the bar inclination:
            :python:`alpha_i = bar.inclination - bar.node_i.rotation`
            :python:`alpha_j = bar.inclination - bar.node_j.rotation`

            The resulting matrix has the form

            .. math::
                \left[\begin{array}{cccccc}
                \cos(\alpha_i) & \sin(\alpha_i) & 0 & 0 & 0 & 0\\
                -\sin(\alpha_i) & \cos(\alpha_i) & 0 & 0 & 0 & 0\\
                0 & 0 & 1 & 0 & 0 & 0 \\
                0 & 0 & 0 & \cos(\alpha_j) & \sin(\alpha_j) & 0 \\
                0 & 0 & 0 & -\sin(\alpha_j) & \cos(\alpha_j) & 0 \\
                0 & 0 & 0 & 0 & 0 & 1
                \end{array}\right]

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
            >>> import numpy
        >>> node_1 = Node(0, 0, rotation=numpy.pi/4)
        >>> node_2 = Node(4, -3)
        >>> cross_sec = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> material = Material(210000000, 0.1, 81000000, 0.1)
        >>> bar = Bar(node_1, node_2, cross_sec, material)
        >>> bar.transformation_matrix()
        array([[0.98994949, -0.14142136, 0., 0., 0., 0.],
        [0.14142136, 0.98994949, 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0.8, 0.6, 0.],
        [0., 0., 0., -0.6, 0.8, 0.],
        [0., 0., 0., 0., 0., 1.]])
        >>> bar.transformation_matrix(False)
        array([[0.8, 0.6, 0., 0., 0., 0.],
        [-0.6, 0.8, 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0.],
        [0., 0., 0., 0.8, 0.6, 0.],
        [0., 0., 0., -0.6, 0.8, 0.],
        [0., 0., 0., 0., 0., 1.]])
        """
        alpha_i = alpha_j = self.inclination
        if to_node_coord:
            alpha_i -= self.node_i.rotation
            alpha_j -= self.node_j.rotation
        return np.vstack((
            np.hstack((transformation_matrix(alpha_i), np.zeros((3, 3)))),
            np.hstack((np.zeros((3, 3)), transformation_matrix(alpha_j))),
        ))

    # is this correct?
    def same_location(self, other):
        """ TODO """
        a = (
            self.node_i.same_location(other.node_i) and
            self.node_j.same_location(other.node_j)
        )
        b = (
            self.node_i.same_location(other.node_j) and
            self.node_j.same_location(other.node_i)
        )
        return a or b

    @cached_property
    def inclination(self):
        r""" Calculates the inclination of the beam in relation to the node
        coordinates.

        Returns
        -------
        :any:`float`
            Angle of inclination in rad.

        Notes
        -----
            The inclination is calcultated by using the following equation:

            .. math::
                \alpha = \arctan \frac{(-z_2 + z_1)}{(x_2 - x_1)}

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> node_1 = Node(0, 0)
        >>> node_2 = Node(4, -2)
        >>> cross_sec = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> material = Material(210000000, 0.1, 81000000, 0.1)
        >>> bar_1 = Bar(node_1, node_2, cross_sec, material)
        >>> bar_1.inclination
        0.7853981634
        """
        return np.arctan2(
            -self.node_j.z + self.node_i.z, self.node_j.x - self.node_i.x
        )

    @cached_property
    def length(self):
        r"""Calculates the distance between two nodes that define the bar
        element.

        Returns
        -------
        :any:`float`
            Length of bar element.

        Notes
        -----
            The length is calcultated by using the following equation:

            .. math::
                L = \sqrt{(x_2 - x_1)^2 + (z_2 - z_1)^2}

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> node_1 = Node(2, 5)
        >>> node_2 = Node(10, 6)
        >>> cross_sec = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> material = Material(210000000, 0.1, 81000000, 0.1)
        >>> bar_1 = Bar(node_1, node_2, cross_sec, material)
        >>> bar_1.length
        8.06225774829855
        """
        return np.sqrt(
            (self.node_j.x - self.node_i.x) ** 2 +
            (self.node_j.z - self.node_i.z) ** 2
        )

    @cached_property
    def hinge(self):
        """Creates a tuple containing bar hinges at both ends of a structural
        element.

        This method assembles and returns the hinge conditions of a bar
        at its start node (*i*) and end node (*j*). Hinges define how the bar
        is allowed to move or rotate at its ends.

        The tuple consists of six elements representing the hinges at each
        node:

        Returns
        -------
        tuple
            A 6-tuple containing the hinge parameters in the following order:
                * :py:attr:`hinge_u_i` (bool): Normal hinge at node *i*.
                * :py:attr:`hinge_w_i` (bool): Shear hinge at node *i*.
                * :py:attr:`hinge_phi_i` (bool): Moment hinge at node *i*.
                * :py:attr:`hinge_u_j` (bool): Normal at node *j*.
                * :py:attr:`hinge_w_j` (bool): Shear hinge at node *j*.
                * :py:attr:`hinge_phi_j` (bool): Moment at node *j*.

        Notes
        -----
        A value of ``True`` indicates the presence of a hinge,
        while ``False`` indicates a rigid connection at that degree of freedom.

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        >>> n2 = Node(4, 0, u='fixed', w='fixed', phi='fixed')
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> mat = Material(210000000, 0.1, 81000000, 0.1)
        >>> b = Bar(n1, n2, cross, mat, hinge_w_i=True, hinge_phi_j=True)
        >>> b.hinge
        (False, True, False, False, False, True)
        """
        return (
            self.hinge_u_i, self.hinge_w_i, self.hinge_phi_i,
            self.hinge_u_j, self.hinge_w_j, self.hinge_phi_j,
        )

    @cached_property
    def flexural_stiffness(self):
        r"""Calculates the flexural stiffness (:math:`EI`) of the element.

        The flexural stiffness is defined as the product of the Young's modulus
        (:math:`E`) and the second moment of area (moment of inertia,
        :math:`I`) of the cross-section.

        Returns
        -------
        :any:`float`
            The flexural stiffness :math:`EI`.

        Notes
        -----
            The flexural stiffness is given by:

            .. math::
                EI = E \cdot I

            If the bending deformation component is not to be considered in the
            calculation (:py:attr:`deformations` does not include 'moment'),
            the flexural stiffness of the bar is scaled to reduce the
            deformation component so that the bending deformation becomes
            negligibly small.

            .. math::
                EI = 1000 \cdot E \cdot I

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> n1 = Node(0, 0, u='fixed', w='fixed')
        >>> n2 = Node(4, 0, w='fixed')
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> mat = Material(210000000, 0.1, 81000000, 0.1)
        >>> b = Bar(n1, n2, cross, mat, deformations=['moment', 'normal'])
        >>> b.flexural_stiffness
        5814.900000000001
        >>> b = Bar(n1, n2, cross, mat, deformations=['normal'])
        >>> b.flexural_stiffness
        5814900.000000001
        """
        EI = self.material.young_mod * self.cross_section.mom_of_int
        return EI if 'moment' in self.deformations else 1_000 * EI

    EI = property(lambda self: self.flexural_stiffness)
    """ Alias of :py:attr:`flexural_stiffness`. """

    @cached_property
    def axial_rigidity(self):
        r"""Calculates the axial rigidity (:math:`EA`) of the bar.

        The axial rigidity is defined as the product of the Young's modulus
        (:math:`E`) and the area (:math:`A`).

        Returns
        -------
        :any:`float`
            The axial rigidity :math:`EA`.

        Notes
        -----
            The axial rigidity is given by:

            .. math::
                EA = E \cdot A

            If the axial deformation component is not to be considered in the
            calculation (:py:attr:`deformations` does not include 'normal'),
            the axial rigidity of the bar is scaled to reduce the
            deformation component so that the axial deformation becomes
            negligibly small.

            .. math::
                EA = 1000 \cdot E \cdot A

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> n1 = Node(0, 0, u='fixed', w='fixed')
        >>> n2 = Node(4, 0, w='fixed')
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> mat = Material(210000000, 0.1, 81000000, 0.1)
        >>> b = Bar(n1, n2, cross, mat, deformations=['moment', 'normal'])
        >>> b.axial_rigidity
        1613640.0
        >>> b = Bar(n1, n2, cross, mat, deformations=['moment'])
        >>> b.axial_rigidity
        1613640000.0
        """
        EA = self.material.young_mod * self.cross_section.area
        return EA if 'normal' in self.deformations else 1_000 * EA

    EA = property(lambda self: self.axial_rigidity)
    """ Alias of :py:attr:`axial_rigidity`. """

    @cached_property
    def shear_stiffness(self):
        r"""Calculates the shear stiffness (:math:`GA_s`) of the bar.

        The shear stiffness is calculated using the shear modulus (:math:`G`)
        and the shear area (:math:`A_s`). The modification of the shear area
        (:math:`A_s`) compared to the cross-sectional area (:math:`A`) is taken
        into account using the shear correction factor (:math:`\kappa`).

        Returns
        -------
        :any:`float`
            The shear stiffness :math:`GA_s`.

        Notes
        -----
            The shear stiffness is given by:

            .. math::
                GA_s = G \cdot A \cdot \kappa

            If the shear deformation component is not to be considered in the
            calculation (:py:attr:`deformations` does not include 'shear'),
            the flexural stiffness of the bar is scaled to reduce the
            deformation component so that the shear deformation becomes
            negligibly small.

            .. math::
                GA_s = 1000 \cdot EI

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> n1 = Node(0, 0, u='fixed', w='fixed')
        >>> n2 = Node(4, 0, w='fixed')
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> mat = Material(210000000, 0.1, 81000000, 0.1)
        >>> b = Bar(n1, n2, cross, mat, deformations=['moment', 'normal'])
        >>> b.shear_stiffness
        5814900.000000001
        >>> b = Bar(n1, n2, cross, mat, deformations=['moment', 'shear'])
        >>> b.shear_stiffness
        390581.97463079996
        """
        GA = self.material.shear_mod * self.cross_section.area
        GA_s = GA * self.cross_section.shear_cor
        EI = self.material.young_mod * self.cross_section.mom_of_int
        return GA_s if 'shear' in self.deformations else 1_000 * EI

    GA_s = property(lambda self: self.shear_stiffness)
    """ Alias of :py:attr:`shear_stiffness`. """

    @cached_property
    def phi(self):
        r"""Calculates the stiffness contributions of shear deformation.

        Returns
        -------
        :any:`float`
            Stiffness contributions of shear deformation.

        Notes
        -----
            .. math::
                \phi = \dfrac{12 \cdot E \cdot I}{GA_s \cdot \ell^2}

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> n1 = Node(0, 0, u='fixed', w='fixed')
        >>> n2 = Node(4, 0, w='fixed')
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> mat = Material(210000000, 0.1, 81000000, 0.1)
        >>> b = Bar(n1, n2, cross, mat)
        >>> b.phi
        0.0007499999999999999
        >>> b = Bar(n1, n2, cross, mat, deformations=['moment', 'shear'])
        >>> b.phi
        0.011165837860598733
        """
        return 12 * self.EI / (self.GA_s * self.length ** 2)

    @cached_property
    def line_load(self):
        """The overall line load on a bar element as a 6x1 vector.

        Returns
        -------
        :any:`numpy.array`
            Sum of all line loads specified in :py:attr:`line_loads`.

        See Also
        --------
        :py:class:`BarLineLoad`

        Notes
        -----
            If no line loads are specified, then a 6x1 zero vector is
            returned.
            The load components are rotated by the inclination of the beam.

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> node_1 = Node(0, 0)
        >>> node_2 = Node(3, -4)
        >>> cross_sec = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> material = Material(210000000, 0.1, 81000000, 0.1)
        >>> Bar(node_1, node_2, cross_sec, material).line_load
        array([[0], [0], [0], [0], [0], [0]])

        >>> from sstatics.core.preprocessing.loads import BarLineLoad
        >>> line_loads = (BarLineLoad(1, 1, 'z', 'bar', 'exact'),
        >>>               BarLineLoad(2, 3, 'x', 'system', 'proj'))
        >>> Bar(node_1, node_2, cross_sec, material,
        >>>     line_loads=line_loads).line_load
        array([[0.96], [2.28], [0], [1.44], [2.92], [0]])
        """
        if len(self.line_loads) == 0:
            return np.array([[0], [0], [0], [0], [0], [0]])
        return np.sum(
            [load.rotate(self.inclination) for load in self.line_loads], axis=0
        )

    @property
    def point_load(self):
        """The overall point load on a bar element as a 6x1 vector.

        Returns
        -------
        :any:`numpy.array`
            Sum of all point loads specified in :py:attr:`point_loads`.

        See Also
        --------
        :py:class:`BarPointLoad`

        Notes
        -----
            If no point loads are specified, then a 6x1 zero vector is
            returned.

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> node_1 = Node(0, 0)
        >>> node_2 = Node(3, 0)
        >>> cross_sec = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> material = Material(210000000, 0.1, 81000000, 0.1)
        >>> Bar(node_1, node_2, cross_sec, material).point_load
        array([[0], [0], [0], [0], [0], [0]])

        >>> from sstatics.core.preprocessing.loads import BarPointLoad
            >>> import numpy
        >>> point_loads = (BarPointLoad(1, 0, 0),
        >>>                BarPointLoad(0, 2, numpy.pi/4, position=1))
        >>> Bar(node_1, node_2, cross_sec, material,
        >>>     point_loads=point_loads).point_load
        array([[1], [0], [0], [1.41421356], [1.41421356], [0]])
        """
        if len(self.point_loads) == 0:
            return np.array([[0], [0], [0], [0], [0], [0]])
        return np.sum(
            [load.rotate() for load in self.point_loads], axis=0
        )

    @property
    def f0_point(self):
        r"""Calculates the internal forces due to point loads related to
        the local bar coordinate system.

        The method rotates the point load components from the system
        coordination system into the bar coordination system. Only point load
        componants at the beginning (position = 0) and at the end
        (position = 1) of the beam are included.

        Returns
        -------
        :any:`numpy.array`
            6x1 vector of the rotated point load components.

        Notes
        -----
            The transformation is performed using the following rotation
            matrix:

            .. math::
                f^{0'}=
               \begin{bmatrix}
                   \cos(\alpha- \beta_i) & \sin(\alpha - \beta_i) & 0 & 0 & 0
                   & 0\\
                   -\sin(\alpha - \beta_i) & \cos(\alpha - \beta_i) & 0 &
                   0 & 0 & 0\\
                   0 & 0 & 1 & 0 & 0 & 0 \\
                   0 & 0 & 0 & \cos(\alpha- \beta_j) & \sin(\alpha - \beta_j)
                   & 0 \\
                   0 & 0 & 0 & -\sin(\alpha - \beta_j) & \cos(\alpha - \beta_j)
                   & 0\\
                   0 & 0 & 0 & 0 & 0 & 1
               \end{bmatrix}^{T} \cdot
               f^{0}_{load}
        """
        m = np.transpose(self.transformation_matrix())
        return m @ self.point_load

    @cached_property
    def f0_temp(self):
        r"""Calculates the internal forces due to temperature loads related to
        the local bar coordinate system.

        Returns
        -------
        :any:`numpy.array`
            6x1 vector of the internal forces due to temperature loads.

        Notes
        -----
            The vector is calculated by the following mathmatical equations:

            .. math::
                f^{0'} =
                \left\lbrace\begin{array}{c}
                \alpha_T \cdot T \cdot E \cdot A \\ 0 \\
                \dfrac{\alpha_T \cdot \Delta T \cdot E \cdot I}{h} \\
                - \alpha_T \cdot T \cdot E \cdot A \\ 0 \\
                - \dfrac{\alpha_T \cdot \Delta T \cdot E \cdot I}{h}
                \end{array}\right\rbrace

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> from sstatics.core.preprocessing.temperature import BarTemp
        >>> n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        >>> n2 = Node(4, 0, u='fixed', w='fixed', phi='fixed')
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> mat = Material(210000000, 0.1, 81000000, 1.2e-5)
        >>> temp = BarTemp(15, 30)
        >>> b = Bar(n1, n2, cross, mat, temp=temp)
        >>> b.f0_temp
        array([[435.6828], [0], [5.23341], [-435.6828], [0], [-5.23341]])
        """
        factor = self.material.therm_exp_coeff * self.material.young_mod
        f0_x = factor * self.temp.temp_s * self.cross_section.area
        f0_m = factor * self.temp.temp_delta * self.cross_section.mom_of_int
        f0_m /= self.cross_section.height
        return np.array([[f0_x], [0], [f0_m], [-f0_x], [0], [-f0_m]])

    @cached_property
    def f0_displacement(self):
        r"""Calculates the internal forces due to support stresses related to
        the local bar coordinate system.

        Returns
        -------
        :any:`numpy.array`
            6x1 vector of the internal forces due to support stresses.

        Notes
        -----
            The support displacements of the initial and end nodes are combined
            into a 6x1 vector. By multiplying the stiffness matrix with the
            acting support displacements, the resulting internal forces are
            obtained. Since the support displacements are given in the node
            coordinate system, a transformation using the transformation matrix
            is required to obtain the internal forces in the bar coordinate
            system.

            The vector is calculated using the following mathematical
            equations:

            .. math::
                f^{0'} = k^{'} \cdot \Delta^{'}

            Including the transformation matrix, the following equation is
            used:

            .. math::
                f^{0'} = k^{'} \cdot T \cdot \left\lbrace\begin{array}{c}
                u_i \\ w_i \\ \varphi_i \\ u_j \\ w_j \\ \varphi_j
                \end{array}\right\rbrace

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> from sstatics.core.preprocessing.dof import NodeDisplacement
        >>> displace = NodeDisplacement(0, 0.005, 0)
        >>> n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        >>> n2 = Node(4, 0, w='fixed', displacements=displace)
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> mat = Material(210000000, 0.1, 81000000, 1.2e-5)
        >>> b = Bar(n1, n2, cross, mat)
        >>> b.f0_displacement
        array([[0], [-5.45146875], [10.9029375], [0], [5.45146875],
        [10.9029375]])
        """
        f0_displacement = np.vstack(
            (self.node_i.displacement, self.node_j.displacement)
        )
        k = self.stiffness_matrix(hinge_modification=False,
                                  to_node_coord=False)
        return k @ np.transpose(self.transformation_matrix()) @ f0_displacement

    @cached_property
    def f0_line(self):
        r"""Calculates the internal forces due to external line loads related
        to the local bar coordinate system.

        Returns
        -------
        :any:`numpy.array`
            6x1 vector of the internal forces due to external loads

        Notes
        -----

            In the calculation of the deformation method, the load
            deformation state (LDS) of the bar element is considered in
            addition to unit deformation state (UDS) of the unloaded bar.
            For common loading conditions, internal forces
            are tabulated in engineering handbooks. The implemented vector for
            internal forces due to distributed loads assumes a trapezoidal load
            distribution. The vector also takes shear deformations into
            account.

            The vector is calculated using the following mathematical
            equations:

            .. math::
                f^{0'} =
                \left\lbrace\begin{array}{c}
                f_{x,i}^{(0)'} \\\\ f_{z,i}^{(0)'} \\\\ f_{M,i}^{(0)'} \\\\
                f_{x,j}^{(0)'} \\\\ f_{z,j}^{(0)'} \\\\ f_{M,j}^{(0)'}
                \end{array}\right\rbrace = \left\lbrace\begin{array}{c}
                \dfrac{(7n_i + 3 n_j) \ell}{20} \\\\
                \dfrac{(40EIp_j + 80EIp_i + 3GA_s \ell^2 p_j + 7 G A_s \ell^2
                p_i) \ell}{240 EI + 20 GA_s \ell^2} \\\\
                -\dfrac{(30 EI p_j + 30 EI p_i + 2 GA_s \ell p_j + 3 GA_s
                \ell^2 p_i) \ell^2}{720EI + 60 GA_s \ell^2} \\\\
                -\dfrac{(3n_i + 7 n_j) \ell}{20} \\\\
                -\dfrac{(80EIp_j + 40EIp_i + 7GA_s \ell^2 p_j + 3 G A_s \ell^2
                p_i) \ell}{240 EI + 20 GA_s \ell^2} \\\\
                -\dfrac{(30 EI p_j + 30 EI p_i + 3 GA_s \ell p_j + 2 GA_s
                \ell^2 p_i) \ell^2}{720EI + 60 GA_s \ell^2}
                \end{array}\right\rbrace
        """
        p, const = self.line_load, self.GA_s * self.length ** 2
        factors = np.array([
            [7, 3, 0, 0],
            [0, 0, 40 * self.EI + 3 * const, 80 * self.EI + 7 * const],
            [0, 0, 30 * self.EI + 2 * const, 30 * self.EI + 3 * const],
        ]) * np.array([
            [1], [1 / (12 * self.EI + const)],
            [self.length / (36 * self.EI + 3 * const)]
        ]) * self.length / 20
        f0 = np.vstack((
            factors @ np.array([[p[0][0]], [p[3][0]], [p[4][0]], [p[1][0]]]),
            factors @ np.array([[p[3][0]], [p[0][0]], [p[1][0]], [p[4][0]]]),
        ))
        return f0 * np.array([[-1], [-1], [1], [-1], [-1], [-1]])

    @cached_property
    def stiffness_shear_force(self):
        r"""Computes the element stiffness matrix considering shear effects.

        Returns
        -------
        :any:`numpy.array`
            A 6x6 matrix representing the element stiffness matrix according to
            first-order theory, including shear effects.

        Notes
        -----
            The element stiffness matrix accounts for the deformation
            components of the beam due to axial force, bending, and shear.
            The factor :py:attr:`phi` describes the influence of shear
            deformation.

            The general form of the element stiffness matrix :math:`k^{'}` is
            given by:

            .. math::
                k^{'} =
                \left[\begin{array}{rrr|rrr}
                \dfrac{EA}{\ell} & 0 & 0 & -\dfrac{EA}{\ell} & 0 & 0 \\
                0 & \dfrac{12EI}{\ell^3 ( 1 + \phi)} & -\dfrac{6EI}{\ell^2
                (1 + \phi)} & 0 & -\dfrac{12EI}{\ell^3(1 + \phi)} & -
                \dfrac{6EI}{\ell^2(1 + \phi)} \\
                0 & -\dfrac{6EI}{\ell^2(1 + \phi)} & \dfrac{EI(4 + \phi)}{\ell
                (1 + \phi)} & 0 & \dfrac{6EI}{\ell^2(1 + \phi)} & \dfrac{EI(2 -
                 \phi)}{\ell(1 + \phi)} \\  \hline
                -\dfrac{EA}{\ell} & 0 & 0 & \dfrac{EA}{\ell} & 0 & 0 \\
                0 & -\dfrac{12EI}{\ell^3(1 + \phi)} & \dfrac{6EI}{\ell^2(1 +
                \phi)} & 0 & \dfrac{12EI}{\ell^3(1 + \phi)} & \dfrac{6EI}
                {\ell^2(1 + \phi)} \\
                0 & -\dfrac{6EI}{\ell^2(1 + \phi)} & \dfrac{EI(2 - \phi)}{\ell
                (1 + \phi)} & 0 & \dfrac{6EI}{\ell^2(1 + \phi)} & \dfrac{EI(4 +
                 \phi)}{\ell(1 + \phi)} \\
                \end{array}\right]

        Examples
        --------
        >>> from sstatics.core.preprocessing.bar import Bar
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> from sstatics.core.preprocessing.loads import BarLineLoad
        >>> n1, n2 = Node(0, 0), Node(0, -4)
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> material = Material(210000000, 0.1, 81000000, 0.1)
        >>> deform = ['moment', 'normal', 'shear']
        >>> b = Bar(n1, n2, cross, material, deformations=deform)
        >>> b.stiffness_shear_force
        array([[1, 0, 0, 1, 0, 0],
             [0, 0.9889574613, 0.9889574613, 0, 0.9889574613, 0.9889574613],
             [0, 0.9889574613, 0.991718096,  0, 0.9889574613, 0.983436192],
             [1, 0, 0, 1, 0, 0],
             [0, 0.9889574613, 0.9889574613, 0, 0.9889574613, 0.9889574613],
             [0, 0.9889574613, 0.983436192,  0, 0.9889574613, 0.991718096]])
    """
        return np.array([
            [1 + self.phi, 0, 0, 1 + self.phi, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 1, (4 + self.phi)/4, 0, 1, (2 - self.phi)/2],
            [1 + self.phi, 0, 0, 1 + self.phi, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 1, (2 - self.phi)/2, 0, 1, (4 + self.phi)/4],
        ]) / (1 + self.phi)

    def f0(
            self,
            hinge_modification: bool = True, to_node_coord: bool = True
    ):
        r"""Represents the vector of internal forces at the beam element ends
        in the load-deformation state (LDS).

        Due to external loads acting on the beam element, internal forces
        develop at the element ends. These internal forces are assembled in
        the vector :math:`f^{(0)'}`.

        Parameters
        ----------
        hinge_modification : :any:`bool`, default=True
            Modifies the load vector and the element stiffness matrix to
            account for hinges.
        to_node_coord : :any:`bool`, default=True
            Transforms the load vector into the nodal coordinate system.

        Returns
        -------
        :any:`numpy.array`
            The 6x1 vector of internal forces at the beam element ends due to
            external loads.

        See Also
        --------
        :py:attr:`f0_displacement`
        :py:attr:`f0_line`
        :py:attr:`f0_point`
        :py:attr:`f0_temp`

        Notes
        -----
        The beam element can be subjected to distributed loads and point loads.
        In statically indeterminate systems, internal forces may also arise due
        to temperature effects and support deformations.
        These loading cases are collected within the algorithm.

        If hinges are present in the beam, it is necessary to modify the
        stiffness matrix and load vector to account for discontinuities in
        deformations at hinge locations.

        Finally, the load vector may be transformed into the nodal coordinate
        system if required.
        """
        f0 = self.f0_line + self.f0_temp + self.f0_displacement - self.f0_point

        if hinge_modification:
            k = self.stiffness_matrix(hinge_modification=False,
                                      to_node_coord=False)
            for i, value in enumerate(self.hinge):
                if value:
                    f0 = f0 - 1 / k[i, i] * k[:, i:i + 1] * f0[i, :]
                    k = k - 1 / k[i, i] * k[:, i:i + 1] @ np.transpose(
                        k[:, i:i + 1]
                    )

        if to_node_coord:
            f0 = self.transformation_matrix() @ f0

        return f0

    def stiffness_matrix(
            self,
            hinge_modification: bool = True, to_node_coord: bool = True
    ):
        r""" Represents the element stiffness matrix for the beam element
        :math:`k^{'}`.

        The element stiffness matrix defines the relationship between nodal
        displacements and rotations of a beam element and the resulting
        internal forces at its ends.

        Parameters
        ----------
        hinge_modification : :any:`bool`, default=True
            Modifies the element stiffness matrix to account for hinges.
        to_node_coord : :any:`bool`, default=True
            Transforms the element stiffness matrix into the nodal coordinate
            system.

        Returns
        -------
        :any:`numpy.array`
            The 6x6 matrix representing the element stiffness matrix.

        See Also
        --------
        :py:attr:`stiffness_shear_force`

        Notes
        -----
        The element stiffness matrix is symmetric and depends on the order of
        calculation and the applied approach. If the beam is shear-flexible,
        the shear component is included in the matrix.

        The general form of the element stiffness matrix :math:`k^{'}` for a
        shear-rigid beam is:

        .. math::
            k^{'} =
            \left[\begin{array}{rrr|rrr}
            \dfrac{EA}{\ell} & 0 & 0 & -\dfrac{EA}{\ell} & 0 & 0 \\
            0 & \dfrac{12EI}{\ell^3} & -\dfrac{6EI}{\ell^2} & 0
            & -\dfrac{12EI}{\ell^3} & -\dfrac{6EI}{\ell^2} \\
            0 & -\dfrac{6EI}{\ell^2} & \dfrac{4EI}{\ell} & 0
            & \dfrac{6EI}{\ell^2} & \dfrac{2EI}{\ell} \\  \hline
            -\dfrac{EA}{\ell} & 0 & 0 & \dfrac{EA}{\ell} & 0 & 0 \\
            0 & -\dfrac{12EI}{\ell^3} & \dfrac{6EI}{\ell^2} & 0
            & \dfrac{12EI}{\ell^3} & \dfrac{6EI}{\ell^2} \\
            0 & -\dfrac{6EI}{\ell^2} & \dfrac{2EI}{\ell} & 0
            & \dfrac{6EI}{\ell^2} & \dfrac{4EI}{\ell} \\
            \end{array}\right]

        If hinges are present in the beam, it is necessary to modify the
        stiffness matrix to account for discontinuities in deformations at
        hinge locations.

        Finally, the stiffness matrix may be transformed into the nodal
        coordinate system if required.
        """

        EI_l = self.EI / self.length
        EI_l2 = EI_l / self.length
        k = np.array([
            [self.EA, 0, 0, -self.EA, 0, 0],
            [0, 12 * EI_l2, -6 * EI_l, 0, -12 * EI_l2, -6 * EI_l],
            [0, -6 * EI_l, 4 * self.EI, 0, 6 * EI_l, 2 * self.EI],
            [-self.EA, 0, 0, self.EA, 0, 0],
            [0, -12 * EI_l2, 6 * EI_l, 0, 12 * EI_l2, 6 * EI_l],
            [0, -6 * EI_l, 2 * self.EI, 0, 6 * EI_l, 4 * self.EI],
        ]) / self.length

        if 'shear' in self.deformations:
            k = k * self.stiffness_shear_force

        if hinge_modification:
            for i, value in enumerate(self.hinge):
                if value:
                    k = k - 1 / k[i, i] * k[:, i:i + 1] @ np.transpose(
                        k[:, i:i + 1]
                    )

        if to_node_coord:
            trans_m = self.transformation_matrix()
            k = trans_m @ k @ np.transpose(trans_m)

        return k


@dataclass(eq=False)
class BarSecond(Bar):

    approach: Literal['analytic', 'taylor', 'p_delta'] = 'analytic'
    f_axial: float = 0

    @cached_property
    def modified_flexural_stiffness(self):
        r"""Computes the modified flexural stiffness (:math:`B_s`) based on
        **second-order theory** , considering both shear deformations and
        axial force effects.

        The modified flexural stiffness :math:`B_s` accounts for shear
        deformations and the influence of axial force on the beam element.

        Returns
        -------
        :any:`float`
            The modified flexural stiffness :math:`B_s`.

        Notes
        -----
            The modified flexural stiffness is calculated a:

            .. math::
                B_s = EI \cdot ( 1 + \dfrac{L}{GA_s})

        References
        ----------
        ..  [1] Spura, Christian: Einführung in die Balkentheorie nach
            Timoshenko und Euler-Bernoulli. Springer Vieweg, 2019
            https://doi.org/10.1007/978-3-658-25216-8
        """
        return self.EI * (1 + self.f_axial / self.GA_s)

    B_s = property(lambda self: self.modified_flexural_stiffness)
    """ Alias of :py:attr:`modified_flexural_stiffness`. """

    @cached_property
    def characteristic_number(self):
        r"""Returns the characteristic number :math:`\mu` of the beam element.

        This dimensionless characteristic number integrates both the bending
        and shear stiffness as well as the axial force in the beam. It is used
        in the correction functions applied in the calculation of the load
        vector and stiffness matrix based on the analytical approach of
        second-order theory.

        Returns
        -------
        :any:`float`
            The dimensionless characteristic number :math:`\mu`.

        See Also
        --------
        :py:attr:`stiffness_second_order_analytic`
        :py:attr:`f0_line_analytic`

        Notes
        -----
            It should be noted that the correction functions are not defined at
            :math:`\mu = 0`, which may lead to numerical instabilities in this
            range.

            The characteristic number is defined by the following equation:

            .. math::
                \mu = \sqrt{\dfrac{\mid L \mid}{B_s}} \cdot \ell
        """
        return np.sqrt(abs(self.f_axial) / self.B_s) * self.length

    @property
    def f0_line(self):
        if self.approach == 'analytic':
            return self.f0_line_analytic
        elif self.approach == 'taylor':
            return self.f0_line_taylor
        return super().f0_line

    @cached_property
    def f0_line_analytic(self):
        r"""Calculates the internal forces due to external line loads related
        to the local bar coordinate system for the analytical solution of the
        second-order theory.

        To calculate the internal forces for the second-order theory, this
        function considers the particular solution while taking into account
        the axial force :math:`L` in the beam element.

        Returns
        -------
        :any:`numpy.array`
            6x1 vector of the internal forces due to external loads for the
            analytical solution of the second-order theory.

        See Also
        --------
        :py:attr:`averaged_longitudinal_force`

        Notes
        -----
        The application of continuous element loads can generate full
        constraint forces in the base element. This function calculates the
        particular solution of the deformation line according to the
        second-order theory. The particular solution is determined separately
        for positive and negative axial forces :math:`L`, which are evaluated
        based on the input parameter.

        **For negative axial forces, the internal forces are given by:**

        .. math::
             f^{0'} = \left\lbrace\begin{array}{c}
                -\dfrac{(7n_i + 3 n_j) \ell}{20} \\
                -\dfrac{B_s \mu^2}{\ell^2} \cdot c_2 - [\dfrac{\ell}{\mu^2} +
                \dfrac{EI}{GA_s \ell}] (p_j - p_i) \\
                \dfrac{B_s \mu^2}{\ell^2} \cdot c_3 - [\dfrac{\ell^2}{\mu^2} +
                \dfrac{EI}{GA_s}] \cdot p_i \\
                \dfrac{(3n_i + 7 n_j) \ell}{20} \\
                -\dfrac{B_s \mu^2}{\ell^2} \cdot c_2 - p_i \cdot \ell -
                [\dfrac{\ell}{\mu^2} + \dfrac{EI}{GA_s \ell} + \dfrac{\ell}{2}]
                (p_j - p_i) \\
                \dfrac{B_s \mu^2}{\ell^2} [c_3 \cdot \cos (\mu) + c_4 \cdot
                \sin ( \mu)] - [\dfrac{\ell^2}{\mu^2} + \dfrac{EI}{GA_s}] \cdot
                 p_j
                \end{array}\right\rbrace

        with the coefficients:

        .. math::
            \begin{array}{ll}
                c_1 = \dfrac{\ell^2}{6 B_s \mu^3} \cdot &  \bigg(\dfrac{3 \big[
                GA_s \ell^4 + 2 B_s \ell^2 \mu^2 + \frac{B_{s}^2}{GA_s} \mu^4
                \big] \sinh (\mu) (p_i + p_j)}
                {GA_s \ell^2 \mu \sin (\mu) + 2 [GA_s \ell^2 + B_s \mu^2] (\cos
                 ( \mu) - 1)}\\\\
                 & + \dfrac{6 EI \ell^2 \mu (1- \cos ( \mu)) (p_i - p_j)}{GA_s
                  \ell^2 \mu \sin (\mu) + 2 [GA_s \ell^2 + B_s \mu^2] (\cos (
                  \mu) - 1)} \\\\
                 & + \dfrac{[B_s \ell^2 \mu^3 + GA_s \ell^4 \mu] [p_i + 2 p_j +
                  (2 p_i + p_j) \cos (\mu)]}{GA_s \ell^2 \mu \sin (\mu) + 2
                  [GA_s \ell^2 + B_s \mu^2] (\cos ( \mu) - 1)}\bigg)
                \end{array}

        .. math::
             \begin{array}{ll}
                c_2 = -\dfrac{\ell^3}{6 B_s \mu^3} \cdot &  \bigg(\dfrac{12 EI
                ( 1- \cos ( \mu)) (p_i - p_j) + GA_s \ell^2 \mu \sin (\mu)
                (2 p_i + p_j)}
                {GA_s \ell^2 \mu \sin (\mu) + 2 [GA_s \ell^2 + B_s \mu^2]
                (\cos ( \mu) - 1)}\\\\
                 & - \dfrac{3 (GA_s \ell^2 + B_s \mu^2) (1 - \cos ( \mu))
                 (p_i + p_j)}{GA_s \ell^2 \mu \sin (\mu) + 2 [GA_s \ell^2 + B_s
                  \mu^2] (\cos ( \mu) - 1))}\bigg)
                \end{array}

        .. math::
            c_3 = - c_1

        .. math::
            c_4 = -\bigg[ \dfrac{B_s \mu}{G A_s \ell} + \dfrac{\ell}{\mu}
            \bigg] \cdot c_2 - \dfrac{EI \ell^2 ( p_j - p_i)}{B_s GA_s \mu^3}

        **For positive axial forces, the internal forces are given by:**

        .. math::
             f^{0'} = \left\lbrace\begin{array}{c}
                -\dfrac{(7n_i + 3 n_j) \ell}{20} \\
                \dfrac{B_s \mu^2}{\ell^2} \cdot c_2 + [\dfrac{\ell}{\mu^2} -
                \dfrac{EI}{GA_s \ell}] (p_j - p_i) \\
                -\dfrac{B_s \mu^2}{\ell^2} \cdot c_3 + [\dfrac{\ell^2}{\mu^2} -
                \dfrac{EI}{GA_s}] \cdot p_i \\
                \dfrac{(3n_i + 7 n_j) \ell}{20} \\
                -\dfrac{B_s \mu^2}{\ell^2} \cdot c_2 - \dfrac{(p_j + p_i)
                \ell}{2} + [\dfrac{\ell}{\mu^2} + \dfrac{EI}{GA_s \ell}] (p_j -
                p_i) \\
                -B_s \cdot [c_3 \cdot \dfrac{\mu^2}{\ell^2} \cosh (\mu) + c_4
                \cdot \dfrac{\mu^2}{\ell^2} \sinh ( \mu)] + [\dfrac{\ell^2}
                {\mu^2} - \dfrac{EI}{GA_s}] \cdot p_j
                \end{array}\right\rbrace


        with the coefficients:

        .. math::
            \begin{array}{ll}
            c_1 = \dfrac{\ell^2}{6 B_s \mu^3} \cdot &  \bigg(\dfrac{3 \big[GA_s
            \ell^4 - 2 B_s \ell^2 \mu^2 + \frac{B_{s}^2}{GA_s} \mu^4 \big]
            \sinh (\mu) (p_i + p_j)}{2 [GA_s \ell^2 - B_s \mu^2] (1- \cosh
            (\mu)) + GA_s \ell^2 \mu \sinh(\mu)}\\\\
             & + \dfrac{6 EI \ell^2 \mu (1- \cosh ( \mu)) (p_i - p_j)}{2 [GA_s
              \ell^2 - B_s \mu^2] (1- \cosh (\mu)) + GA_s \ell^2 \mu
              \sinh(\mu)} \\\\
             & + \dfrac{[B_s \ell^2 \mu^3 - GA_s \ell^4 \mu] [p_i + 2 p_j +
             (2 p_i + p_j) \cosh (\mu)]}{2 [GA_s \ell^2 - B_s \mu^2] (1- \cosh
             (\mu)) + GA_s \ell^2 \mu \sinh(\mu)}\bigg)
            \end{array}
        .. math::
            \begin{array}{ll}
            c_2 = -\dfrac{\ell^3}{6 B_s \mu^3} \cdot &  \bigg(\dfrac{12 EI ( 1-
             \cosh ( \mu)) (p_i - p_j) - GA_s \ell^2 \mu \sinh (\mu) (2 p_i +
             p_j)}
            {2 [GA_s \ell^2 - B_s \mu^2] (1- \cosh (\mu)) + GA_s \ell^2 \mu
            \sinh(\mu)}\\\\
             & - \dfrac{3 (GA_s \ell^2 - B_s \mu^2) (1 - \cosh ( \mu)) (p_i +
             p_j)}{2 [GA_s \ell^2 - B_s \mu^2] (1- \cosh (\mu)) + GA_s \ell^2
             \mu \sinh(\mu)}\bigg)
            \end{array}

        .. math::
            c_3 = - c_1

        .. math::
            c_4 = \bigg[ \dfrac{B_s \mu}{G A_s \ell} - \dfrac{\ell}{\mu} \bigg]
             \cdot c_2 + \dfrac{EI \ell^2 ( p_j - p_i)}{B_s GA_s \mu^3}
        """
        p_vec = self.line_load
        mu = self.characteristic_number
        B_s = self.B_s
        p_i, p_j = p_vec[1][0], p_vec[4][0]
        p_sum, p_diff = p_vec[1][0] + p_vec[4][0], p_vec[1][0] - p_vec[4][0]

        if self.f_axial < 0:
            sin_mu = np.sin(mu)
            cos_mu = np.cos(mu)
            denominator = (self.GA_s * self.length ** 2 * mu * sin_mu + (
                    2 * (self.GA_s * self.length ** 2 + B_s * mu ** 2) * (
                        cos_mu - 1)))

            c_1 = (self.length ** 2 / (6 * B_s * mu ** 3)) * (
                (3 * (self.GA_s * self.length ** 4 + (
                        2 * B_s * self.length ** 2 * mu ** 2) + (
                        B_s ** 2 / self.GA_s * mu ** 4)) * sin_mu * p_sum + (
                            6 * self.EI * self.length ** 2 * mu * (
                                1 - cos_mu)) * p_diff) / denominator - (
                    (B_s * self.length ** 2 * mu ** 3) + (
                        self.GA_s * self.length ** 4 * mu)) * (
                    p_i + 2 * p_j + (2 * p_i + p_j) * cos_mu) / denominator)

            c_2 = -(self.length ** 3 / (6 * B_s * mu ** 2)) * (
                (12 * self.EI * (1 - cos_mu) * p_diff + (
                        self.GA_s * self.length ** 2 * mu * sin_mu) * (
                            2 * p_i + p_j)) / denominator - (
                    3 * (self.GA_s * self.length ** 2 + B_s * mu ** 2) * (
                        1 - cos_mu) * p_sum) / denominator)

            c_3 = -c_1
            c_4 = -((B_s * mu / (self.GA_s * self.length)) + (
                    self.length / mu)) * c_2 - (
                    self.EI * self.length ** 2 * (p_j - p_i) /
                    (B_s * self.GA_s * mu ** 3))

            f0_z_i = - (B_s * mu ** 2 / self.length ** 2) * c_2 - (
                    (self.length / mu ** 2)
                    + (self.EI / (self.GA_s * self.length))) * (p_j - p_i)
            f0_z_j = - (
                B_s * mu ** 2 / self.length ** 2) * c_2 - p_i * self.length - (
                  (self.length / mu ** 2) + (
                    self.EI / (self.GA_s * self.length)) + (self.length / 2)
            ) * (p_j - p_i)

            f0_m_i = (B_s * mu ** 2 / self.length ** 2) * c_3 - (
                    (self.length ** 2 / mu ** 2) + (self.EI / self.GA_s)
            ) * p_i
            f0_m_j = (B_s * mu ** 2 / self.length ** 2) * (
                    c_3 * cos_mu + c_4 * sin_mu) - (
                    (self.length ** 2 / mu ** 2) + (self.EI / self.GA_s)
            ) * p_j
        else:
            sin_mu = np.sinh(mu)
            cos_mu = np.cosh(mu)
            denominator = (
                    2 * (self.GA_s * self.length ** 2 - B_s * mu ** 2) *
                    (1 - cos_mu) + self.GA_s * self.length ** 2 * mu * sin_mu)

            c_1 = (self.length ** 2 / (6 * B_s * mu ** 3)) * (
                    (3 * (self.GA_s * self.length ** 4 - (
                            2 * B_s * self.length ** 2 * mu ** 2) + (
                            (B_s ** 2 / self.GA_s) * mu ** 4)
                          ) * sin_mu * p_sum + (
                            6 * self.EI * self.length ** 2 * mu * (
                                1 - cos_mu) * p_diff) + (
                            (B_s * self.length ** 2 * mu ** 3) - (
                                self.GA_s * self.length ** 4 * mu)) * (
                                p_i + 2 * p_j + (2 * p_i + p_j) * cos_mu)
                     ) / denominator)

            c_2 = -(self.length ** 3 / (6 * B_s * mu ** 2)) * (
                    (12 * self.EI * (1 - cos_mu) * p_diff - (
                        self.GA_s * self.length ** 2 * mu * sin_mu) * (
                            2 * p_i + p_j) - (3 * (
                                (self.GA_s * self.length ** 2) - (
                                    B_s * mu ** 2)) * (
                                        1 - cos_mu) * p_sum)) / denominator)

            c_3 = -c_1

            c_4 = ((B_s * mu / (self.GA_s * self.length)) - (self.length / mu)
                   ) * c_2 + (self.EI * self.length ** 2 * (p_j - p_i)) / (
                          B_s * self.GA_s * mu ** 3)

            f0_z_i = (B_s * mu ** 2 / self.length ** 2) * c_2 + (
                    (self.length / mu ** 2) - (
                        self.EI / (self.GA_s * self.length))) * (p_j - p_i)

            f0_z_j = (B_s * mu ** 2 / self.length ** 2) * c_2 - (
                    (p_j + p_i) * self.length / 2) + (
                    (self.length / mu ** 2) - (
                        self.EI / (self.GA_s * self.length))) * (p_j - p_i)

            f0_m_i = -(B_s * mu ** 2 / self.length ** 2) * c_3 + (
                    (self.length ** 2 / mu ** 2) - (
                        self.EI / self.GA_s)) * p_i

            f0_m_j = -B_s * (
                    c_3 * (mu ** 2 / self.length ** 2) * cos_mu + c_4 * (
                        mu ** 2 / self.length ** 2) * sin_mu) + (
                    (self.length ** 2 / mu ** 2) - (self.EI / self.GA_s)) * p_j

        return (
            np.array(
                [[-(7 * p_vec[0][0] + 3 * p_vec[3][0]) * self.length / 20],
                 [-f0_z_i],
                 [-f0_m_i],
                 [-(3 * p_vec[0][0] + 7 * p_vec[3][0]) * self.length / 20],
                 [f0_z_j],
                 [f0_m_j]])
        )

    @cached_property
    def f0_line_taylor(self):
        r"""Calculates the internal forces due to external line loads related
        to the local bar coordinate system for the Taylor series expansion of
        the second-order theory.

        To calculate the internal forces for the second-order theory, this
        function considers the Taylor series expansion while taking into
        account the axial force :math:`L` in the beam element.

        Returns
        -------
        :any:`numpy.array`
            6x1 vector of the internal forces due to external loads for the
            Taylor series expansion of the second-order theory.

        See Also
        --------
        :py:attr:`averaged_longitudinal_force`

        Notes
        -----
            The application of continuous element loads can generate full
            constraint forces in the base element. This function calculates the
            internal forces by using a Taylor series expansion according to the
            second-order theory.

            **The vector is calculated by the following mathmatical
            equations:**

            .. math::
                \begin{array}{ll}
                    f_{z,i}^{(0)'} = \dfrac{\ell}{20} \cdot &  \bigg(\dfrac{720
                     B_{s}^2 (p_j + p_i) - 4 EIGA_s \ell^2 (p_j - p_i) + 20 B_s
                      GA_s \ell^2 ( 5 p_j + 7 p_i)}{(12 B_s + GA_s
                      \ell^2)^2}\\\\
                     & \dfrac{(GA_s)^2 \ell^4 (3 p_j + 7 p_i)}{(12 B_s + GA_s
                     \ell^2)^2} - \dfrac{EI (p_j - p_i)}{GA_s \ell} -
                     \dfrac{12 B_s}{L \ell} \cdot \dfrac{(EI - B_s)
                     (p_j - p_i)}{(12 B_s + GA_s \ell^2)}\bigg)
                \end{array}

            .. math::
                \begin{array}{ll}
                    f_{M,i}^{(0)'} = & \dfrac{4320 B_{s}^3(p_j + p_i) + 6 EI
                    (GA_s)^2 \ell^4 (p_j- p_i) + 60 B_s GA_s \ell^2 (12B_s p_i
                     - GA_s \ell^2 p_j)}{60 GA_s (12 B_s + GA_s \ell^2)^2}\\\\
                     & - \dfrac{(GA_s)^3 \ell^6 (2p_j + 3 p_i)}{60 GA_s (12 B_s
                      + GA_s \ell^2)^2} - \dfrac{EI}{GA_s} \cdot p_i +
                      \dfrac{6 B_s}{L} \cdot \dfrac{(EI - B_s) ( p_j - p_i)}
                      {(12 B_s + GA_s \ell^2)}
                \end{array}

            .. math::
                f_{z,j}^{(0)'} = f_{z,i}^{(0)'} - \dfrac{(p_j + p_i) \cdot
                \ell}{2}

            .. math::
                \begin{array}{ll}
                f_{M,j}^{(0)'} = & \dfrac{4320 B_{s}^3(p_j + p_i) - 6 EI
                (GA_s)^2 \ell^4 (p_j- p_i) + 60 B_s GA_s \ell^2 (12B_s p_j -
                GA_s \ell^2 p_i)}{60 GA_s (12 B_s + GA_s \ell^2)^2}\\\\
                 & - \dfrac{(GA_s)^3 \ell^6 (3p_j + 2 p_i)}{60 GA_s (12 B_s +
                 GA_s \ell^2)^2} - \dfrac{EI}{GA_s} \cdot p_j - \dfrac{6 B_s}
                 {L} \cdot \dfrac{(EI - B_s) ( p_j - p_i)}{(12 B_s + GA_s
                 \ell^2)}
                \end{array}

            .. math::
                f^{0'} = \left\lbrace\begin{array}{c}
                -\dfrac{(7n_i + 3 n_j) \ell}{20} \\\\
                -f_{z,i}^{(0)'} \\\\
                -f_{M,i}^{(0)'} \\\\
                -\dfrac{(3n_i + 7 n_j) \ell}{20}\\\\
                f_{z,j}^{(0)'}\\\\
                f_{M,j}^{(0)'}
                \end{array}\right\rbrace

        """
        p_vec = self.line_load
        B_s = self.B_s
        p_i, p_j = p_vec[1][0], p_vec[4][0]

        f0_z_i = (self.length / 20) * (720 * B_s ** 2 * (p_j + p_i) - (
                4 * self.EI * self.GA_s * self.length ** 2) * (p_j - p_i) + (
                20 * B_s * self.GA_s * self.length ** 2) * (
                5 * p_j + 7 * p_i) + self.GA_s ** 2 * self.length ** 4 * (
                3 * p_j + 7 * p_i)) / (
                12 * B_s + self.GA_s * self.length ** 2) ** 2 - (
                self.EI * (p_j - p_i)) / (self.GA_s * self.length) - (
                12 * B_s) / (self.f_axial * self.length) * (
                (self.EI - B_s) * (p_j - p_i)) / (
                12 * B_s + self.GA_s * self.length ** 2)

        f0_m_i = (4320 * B_s ** 3 * (p_j + p_i) + (
                6 * self.EI * self.GA_s ** 2 * self.length ** 4) * (
                p_j - p_i) + 60 * B_s * self.GA_s * self.length ** 2 * (
                12 * B_s * p_i - self.GA_s * self.length ** 2 * p_j) - (
                self.GA_s ** 3 * self.length ** 6) * (2 * p_j + 3 * p_i)) / (
                60 * self.GA_s * (
                    12 * B_s + self.GA_s * self.length ** 2) ** 2) - (
                self.EI * p_i) / self.GA_s + (6 * B_s) / self.f_axial * (
                self.EI - B_s) * (p_j - p_i) / (
                12 * B_s + self.GA_s * self.length ** 2)

        f0_z_j = f0_z_i - (p_j + p_i) * self.length / 2

        f0_m_j = (4320 * B_s ** 3 * (p_j + p_i) - (
                6 * self.EI * self.GA_s ** 2 * self.length ** 4) * (
                p_j - p_i) + 60 * B_s * self.GA_s * self.length ** 2 * (
                12 * B_s * p_j - self.GA_s * self.length ** 2 * p_i) - (
                self.GA_s ** 3 * self.length ** 6) * (
                3 * p_j + 2 * p_i)) / (60 * self.GA_s * (
                    12 * B_s + self.GA_s * self.length ** 2) ** 2) - (
                self.EI * p_j) / self.GA_s - (6 * B_s / self.f_axial) * (
                self.EI - B_s) * (p_j - p_i) / (
                12 * B_s + self.GA_s * self.length ** 2)

        return (
            np.array(
                [[-(7 * p_vec[0][0] + 3 * p_vec[3][0]) * self.length / 20],
                 [-f0_z_i],
                 [-f0_m_i],
                 [-(3 * p_vec[0][0] + 7 * p_vec[3][0]) * self.length / 20],
                 [f0_z_j],
                 [f0_m_j]])
        )

    @cached_property
    def stiffness_matrix_analytic(self):
        r"""Creates the element stiffness matrix according to second-order
        theory for the analytical solution.

        To establish the stiffness matrix based on second-order theory, four
        correction functions must be calculated, which account for the
        influence of the axial force (:math:`L`).

        Returns
        -------
        :any:`numpy.array`
            A 6x6 matrix representing the theoretically exact element stiffness
            matrix according to second-order theory for a shear-flexible and
            bending-flexible beam under constant negative or positive axial
            force.

        See Also
        --------
        :py:attr:`averaged_longitudinal_force`

        Notes
        -----
            The correction functions are computed depending on the sign of the
            axial force. The functions are also dependent on the
            :py:attr:`characteristic_number`, which is defined by
            the equation:

            .. math::
                \mu = \sqrt{\dfrac{\mid L \mid}{B_s}} \cdot \ell

            For a negative axial force, the following equations apply:

            .. math::
                f_1(\mu) = - \dfrac{B_s}{12EI} \cdot \dfrac{\mu^3 \cdot \sin
                ( \mu)}{2 [\frac{B_s}{GA_s \ell^2} \mu^2 + 1] (\cos (\mu) - 1)
                + \mu \cdot \sin ( \mu)}

            .. math::
                f_2(\mu) = \dfrac{B_s}{6EI} \cdot \dfrac{(\cos (\mu) - 1) \cdot
                \mu^2}{2 [\frac{B_s}{GA_s \ell^2} \mu^2 + 1] (\cos (\mu) - 1)
                + \mu \cdot \sin ( \mu)}

            .. math::
                f_3(\mu) = -\dfrac{B_s}{4EI} \cdot \dfrac{\big([\frac{B_s}{GA_s
                 \ell^2} \mu^2 + 1] \sin ( \mu) - \mu \cdot \cos( \mu ) \big)
                 \cdot \mu}{2 [\frac{B_s}{GA_s \ell^2} \mu^2 + 1] (\cos (\mu)
                 - 1) + \mu \cdot \sin ( \mu)}

            .. math::
                f_4(\mu) = \dfrac{B_s}{2EI} \cdot \dfrac{\big([\frac{B_s}{GA_s
                 \ell^2} \mu^2 + 1] \sin ( \mu) - \mu \big) \cdot \mu}{2 [\frac
                 {B_s}{GA_s \ell^2} \mu^2 + 1] (\cos (\mu) - 1) + \mu \cdot
                 \sin ( \mu)}

            For a positive axial force, the equations are as follows:

            .. math::
                f_1(\mu) = \dfrac{B_s}{12EI} \cdot \dfrac{\mu^3 \cdot \sinh (
                \mu)}{2 [\frac{B_s}{GA_s \ell^2} \mu^2 + 1] (\cosh (\mu) - 1)
                + \mu \cdot \sinh ( \mu)}

            .. math::
                f_2(\mu) = \dfrac{B_s}{6EI} \cdot \dfrac{(\cosh (\mu) - 1)
                \cdot \mu^2}{2 [\frac{B_s}{GA_s \ell^2} \mu^2 + 1] (\cosh (\mu)
                 - 1) + \mu \cdot \sinh ( \mu)}

            .. math::
                f_3(\mu) = \dfrac{B_s}{4EI} \cdot \dfrac{\big([\frac{B_s}{GA_s
                 \ell^2} \mu^2 + 1] \sinh ( \mu) - \mu \cdot \cosh( \mu ) \big)
                  \cdot \mu}{2 [\frac{B_s}{GA_s \ell^2} \mu^2 + 1] (\cosh (\mu)
                   - 1) + \mu \cdot \sinh ( \mu)}

            .. math::
                f_4(\mu) = \dfrac{B_s}{2EI} \cdot \dfrac{\big([\frac{B_s}{GA_s
                 \ell^2} \mu^2 + 1] \sinh ( \mu) - \mu \big) \cdot \mu}{2 [
                 \frac{B_s}{GA_s \ell^2} \mu^2 + 1] (\cosh (\mu) - 1) + \mu
                 \cdot \sinh ( \mu)}

            Which leads to the element stiffness matrix:

            .. math::
                k^{'} =
                \left[\begin{array}{rrr|rrr}
                \dfrac{EA}{\ell} & 0 & 0 & -\dfrac{EA}{\ell} & 0 & 0 \\
                0 & \dfrac{12EI}{\ell^3} \cdot f_1 & -\dfrac{6EI}{\ell^2}
                \cdot f_2 & 0 & -\dfrac{12EI}{\ell^3}\cdot f_1 & -\dfrac{6EI}
                {\ell^2}\cdot f_2 \\
                0 & -\dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{4EI}{\ell}\cdot f_3
                & 0 & \dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{2EI}{\ell}\cdot f_4
                 \\  \hline
                -\dfrac{EA}{\ell} & 0 & 0 & \dfrac{EA}{\ell} & 0 & 0 \\
                0 & -\dfrac{12EI}{\ell^3}\cdot f_1 & \dfrac{6EI}{\ell^2}\cdot
                f_2 & 0 & \dfrac{12EI}{\ell^3}\cdot f_1 & \dfrac{6EI}{\ell^2}
                \cdot f_2 \\
                0 & -\dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{2EI}{\ell}\cdot f_4
                & 0 & \dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{4EI}{\ell}\cdot f_3
                \end{array}\right]

        Examples
        --------
       >>> from sstatics.core.preprocessing.bar import BarSecond
        >>> from sstatics.core.preprocessing.cross_section import CrossSection
        >>> from sstatics.core.preprocessing.material import Material
        >>> from sstatics.core.preprocessing.node import Node
        >>> from sstatics.core.preprocessing.loads import NodePointLoad
        >>> n_load = NodePointLoad(0, 182, 0, rotation=0)
        >>> n1 = Node(0, 0, u='fixed', w='fixed', phi='fixed')
        >>> n2 = Node(0, -4, loads=n_load)
        >>> cross = CrossSection(0.00002769, 0.007684, 0.2, 0.2, 0.6275377)
        >>> material = Material(210000000, 0.1, 81000000, 0.1)
        >>> line_load = BarLineLoad(1, 1.5, 'z', 'bar', 'exact')
        >>> force = -181.99971053936605
        >>> b = BarSecond(n1, n2, cross, material, line_loads=line_load)
        >>> b.stiffness_matrix_analytic()
        array([[1, 0, 0, 1, 0, 0],
             [0, 0.9491546296, 0.9908554233, 0, 0.9491546296, 0.9908554233],
             [0, 0.9908554233, 0.9826126598,  0, 0.9908554233, 1.0073409503],
             [1, 0, 0, 1, 0, 0],
             [0, 0.9491546296, 0.9908554233, 0, 0.9491546296, 0.9908554233],
             [0, 0.9908554233, 1.0073409503,  0, 0.9908554233, 0.9826126598]])
        """
        mu = self.characteristic_number
        B_s = self.B_s
        factor = B_s / (self.GA_s * self.length ** 2)

        if self.f_axial < 0:
            sin_mu = np.sin(mu)
            cos_mu = np.cos(mu)
            denominator = 2 * (factor * mu ** 2 + 1) * (
                    cos_mu - 1) + mu * sin_mu
            f_1 = -(B_s / (12 * self.EI)) * (mu ** 3 * sin_mu) / denominator
            f_2 = (B_s / (6 * self.EI)) * (
                    (cos_mu - 1) * mu ** 2) / denominator
            f_3 = (-(B_s / (4 * self.EI)) * (
                    (factor * mu ** 2 + 1) * sin_mu - mu * cos_mu) * mu
                   / denominator)
            f_4 = (B_s / (2 * self.EI)) * (
                    (factor * mu ** 2 + 1) * sin_mu - mu) * mu / denominator
        else:
            sinh_mu = np.sinh(mu)
            cosh_mu = np.cosh(mu)
            denominator = 2 * (factor * mu ** 2 - 1) * (
                    cosh_mu - 1) + mu * sinh_mu
            f_1 = (B_s / (12 * self.EI)) * (mu ** 3 * sinh_mu) / denominator
            f_2 = (B_s / (6 * self.EI)) * (
                    (cosh_mu - 1) * mu ** 2) / denominator
            f_3 = ((B_s / (4 * self.EI)) * ((
                    factor * mu ** 2 - 1) * sinh_mu + mu * cosh_mu) * mu /
                   denominator)
            f_4 = -(B_s / (2 * self.EI)) * ((
                    factor * mu ** 2 - 1) * sinh_mu + mu) * mu / denominator

        return np.array([[1, 0, 0, 1, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_3, 0, f_2, f_4],
                         [1, 0, 0, 1, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_4, 0, f_2, f_3]])

    @cached_property
    def stiffness_matrix_taylor(self):
        r"""Creates the element stiffness matrix according to second-order
        theory for a Taylor series.

        To establish the stiffness matrix based on second-order theory, four
        correction functions must be calculated, which account for the
        influence of the axial force (:math:`L`).

        Returns
        -------
        :any:`numpy.array`
            A 6x6 matrix representing the theoretically exact element stiffness
            matrix according to second-order theory for a shear-flexible and
            bending-flexible beam under constant negative or positive axial
            force.

        See Also
        --------
        :py:attr:`averaged_longitudinal_force`

        Notes
        -----
            The correction function are calculated by the following equations:

            .. math::
                f_1= \dfrac{B_s}{12EI ( \frac{B_s}{GA_s \ell^2}+ \frac{1}{12})}
                 + \dfrac{L \ell^2}{144 EI} \cdot \Bigg(\dfrac{\frac{B_s}{GA_s
                  \ell^2} + \frac{1}{10}}{(\frac{B_s}{GA_s \ell^2} + \frac{1}
                  {12})^2}\Bigg)

            .. math::
                f_2= \dfrac{B_s}{12EI ( \frac{B_s}{GA_s \ell^2}+ \frac{1}{12})}
                 + \dfrac{L \ell^2}{8640 EI} \cdot \Bigg(\dfrac{1}{(\frac{B_s}
                 {GA_s \ell^2} + \frac{1}{12})^2}\Bigg)

            .. math::
                f_3= \dfrac{B_s ( \frac{B_s}{GA_s \ell^2}+ \frac{1}{3})}{4EI (
                 \frac{B_s}{GA_s \ell^2}+ \frac{1}{12})} + \dfrac{L \ell^2}
                 {48 EI} \cdot \Bigg(\dfrac{1}{240(\frac{B_s}{GA_s \ell^2} +
                 \frac{1}{12})^2} + 1\Bigg)

            .. math::
                f_4= -\dfrac{B_s ( \frac{B_s}{GA_s \ell^2}- \frac{1}{6})}{2EI
                ( \frac{B_s}{GA_s \ell^2}+ \frac{1}{12})} + \dfrac{L \ell^2}
                {24 EI} \cdot \Bigg(\dfrac{1}{240(\frac{B_s}{GA_s \ell^2} +
                \frac{1}{12})^2} - 1\Bigg)

            Which leads to the element stiffness matrix:

            .. math::
                k^{'} =
                \left[\begin{array}{rrr|rrr}
                \dfrac{EA}{\ell} & 0 & 0 & -\dfrac{EA}{\ell} & 0 & 0 \\
                0 & \dfrac{12EI}{\ell^3} \cdot f_1 & -\dfrac{6EI}{\ell^2}
                \cdot f_2 & 0 & -\dfrac{12EI}{\ell^3}\cdot f_1 & -\dfrac{6EI}
                {\ell^2}\cdot f_2 \\
                0 & -\dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{4EI}{\ell}\cdot f_3
                & 0 & \dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{2EI}{\ell}\cdot f_4
                 \\  \hline
                -\dfrac{EA}{\ell} & 0 & 0 & \dfrac{EA}{\ell} & 0 & 0 \\
                0 & -\dfrac{12EI}{\ell^3}\cdot f_1 & \dfrac{6EI}{\ell^2}\cdot
                f_2 & 0 & \dfrac{12EI}{\ell^3}\cdot f_1 & \dfrac{6EI}{\ell^2}
                \cdot f_2 \\
                0 & -\dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{2EI}{\ell}\cdot f_4
                & 0 & \dfrac{6EI}{\ell^2}\cdot f_2 & \dfrac{4EI}{\ell}\cdot f_3
                \end{array}\right]
        """
        B_s = self.B_s
        factor = B_s / (self.GA_s * self.length ** 2)
        denominator_common = factor + 1 / 12
        denominator_squared = denominator_common ** 2
        inv_denominator_common = 1 / denominator_common

        f_1 = (B_s / (12 * self.EI * denominator_common) +
               self.f_axial * self.length ** 2 / (144 * self.EI) *
               (factor + 1 / 10) * inv_denominator_common ** 2)

        f_2 = (B_s / (12 * self.EI * denominator_common) +
               self.f_axial * self.length ** 2 / (8640 * self.EI) *
               inv_denominator_common ** 2)

        f_3 = (B_s * (factor + 1 / 3) / (
                4 * self.EI * denominator_common) +
               self.f_axial * self.length ** 2 / (48 * self.EI) *
               (1 / (240 * denominator_squared) + 1))

        f_4 = (-B_s * (factor - 1 / 6) / (
                2 * self.EI * denominator_common) +
               self.f_axial * self.length ** 2 / (24 * self.EI) *
               (1 / (240 * denominator_squared) - 1))

        return np.array([[1, 0, 0, 1, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_3, 0, f_2, f_4],
                         [1, 0, 0, 1, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_4, 0, f_2, f_3]])

    @cached_property
    def stiffness_matrix_p_delta(self):
        r"""Creates the geometric stiffness matrix :math:`k_{G}^{'}`

        The geometric stiffness matrix describes the relationship between
        beam forces and displacements. It represents the transverse forces
        resulting from the moment offset of axial forces (:math:`L`)
        according to second-order theory.

        Returns
        -------
        :any:`numpy.array`
            A 6x6 matrix representing the geometric stiffness
            matrix according to second-order theory for the P- :math:`\Delta`
            -effect.

        Notes
        -----
            To obtain the approximate solution considering the P-Delta effect,
            a geometric stiffness matrix is added to the element stiffness
            matrix from first-order theory.

            .. math::
                k^{'} = k_{0}^{'} + k_{G}^{'}

            The geometric stiffness matrix is given by:

            .. math::
                k_{G}^{'} =
                \left[\begin{array}{ccc|ccc}
                0 & 0 & 0 & 0 & 0 & 0 \\
                0 & \dfrac{L}{\ell} & 0 & 0 & -\dfrac{L}{\ell} & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 \\  \hline
                0 & 0 & 0 & 0 & 0 & 0 \\
                0 & -\dfrac{L}{\ell} & 0 & 0 & \dfrac{L}{\ell} & 0 \\
                0 & 0 & 0 & 0 & 0 & 0 \\
                \end{array}\right]
        """
        c = self.f_axial / self.length
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, c, 0, 0, -c, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -c, 0, 0, c, 0],
            [0, 0, 0, 0, 0, 0],
        ])

    def f0(
            self,
            hinge_modification: bool = True, to_node_coord: bool = True
    ):
        return super().f0(hinge_modification, to_node_coord)

    def stiffness_matrix(
            self,
            hinge_modification: bool = True, to_node_coord: bool = True,
    ):
        EI_l = self.EI / self.length
        EI_l2 = EI_l / self.length
        k = np.array([
            [self.EA, 0, 0, -self.EA, 0, 0],
            [0, 12 * EI_l2, -6 * EI_l, 0, -12 * EI_l2, -6 * EI_l],
            [0, -6 * EI_l, 4 * self.EI, 0, 6 * EI_l, 2 * self.EI],
            [-self.EA, 0, 0, self.EA, 0, 0],
            [0, -12 * EI_l2, 6 * EI_l, 0, 12 * EI_l2, 6 * EI_l],
            [0, -6 * EI_l, 2 * self.EI, 0, 6 * EI_l, 4 * self.EI],
        ]) / self.length

        if self.approach == "analytic":
            k = k * self.stiffness_matrix_analytic
        elif self.approach == "taylor":
            k = k * self.stiffness_matrix_taylor
        elif self.approach == "p_delta":
            if "shear" in self.deformations:
                k = (k * self.stiffness_shear_force +
                     self.stiffness_matrix_p_delta)
            else:
                k = k + self.stiffness_matrix_p_delta

        if hinge_modification:
            for i, value in enumerate(self.hinge):
                if value:
                    k = k - 1 / k[i, i] * k[:, i:i + 1] @ k[:, i:i + 1].T

        if to_node_coord:
            trans_m = self.transformation_matrix()
            k = trans_m @ k @ trans_m.T

        return k
