
from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import cache, cached_property
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
import sympy


def transformation_matrix(alpha: float):
    r"""Create a 3x3 rotation matrix.

    Parameters
    ----------
    alpha : :any:`float`
        Rotation angle.

    Returns
    -------
    :any:`numpy.array`
        A 3x3 matrix for rotating 3x1 vectors.

    Notes
    -----
    The resulting matrix has the form

    .. math::
        \left(\begin{array}{c}
        \cos(\alpha) & \sin(\alpha) & 0 \\
        -\sin(\alpha) & \cos(\alpha) & 0 \\
        0 & 0 & 1
        \end{array}\right).

    Examples
    --------
    >>> import numpy
    >>> import sstatics
    >>> m = sstatics.transformation_matrix(numpy.pi)
    >>> m
    array([[-1, 0, 0],
           [0, -1, 0],
           [0, 0, 1]])
    >>> m @ numpy.array([[1], [2], [3]])
    array([[-1], [-2], [3]])
    """
    return np.array([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])


@dataclass(eq=False)
class DegreesOfFreedom:
    """ TODO """

    x: float
    z: float
    phi: float

    @cached_property
    def vector(self):
        """ TODO """
        return np.array([[self.x], [self.z], [self.phi]])


NodeDisplacement = DegreesOfFreedom
""" Alias of :py:class:`DegreesOfFreedom` to make clear that this class has
the purpose to shift nodes. """


@dataclass(eq=False)
class PointLoad(DegreesOfFreedom):
    """ TODO """

    rotation: float = 0.0

    def rotate(self, rotation: float):
        """ TODO """
        return transformation_matrix(self.rotation - rotation) @ self.vector


NodePointLoad = PointLoad
""" Alias of :py:class:`PointLoad` to make the use case of this class more
clear. """


@dataclass(eq=False)
class Node:
    """Create a node for a statical system.

    Parameters
    ----------
    x, z : :any:`float`
        description
    rotation : :any:`float`, default=0.0
        description
    u, w, phi : {'free', 'fixed'} or :any:`float`, default='free'
        Specify the fixtures of a node. Real numbers refer to nib widths.
    displacements : :any:`tuple`, default=()
        description
    loads : :any:`tuple`, default=()
        description

    Raises
    ------
    ValueError
        :py:attr:`u`, :py:attr:`w` and :py:attr:`phi` have to be either
        :python:`'free'`, :python:`'fixed'` or a real number.
    ValueError
        :py:attr:`u`, :py:attr:`w` or :py:attr:`phi` are set to zero. A spring
        with a nib width of zero behaves like a free fixture and therefore the
        value need to be set to :python:`'free'`.
    """

    x: float
    z: float
    rotation: float = 0.0
    u: Literal['free', 'fixed'] | float = 'free'
    w: Literal['free', 'fixed'] | float = 'free'
    phi: Literal['free', 'fixed'] | float = 'free'
    displacements: (
        tuple[NodeDisplacement, ...] | list[NodeDisplacement] |
        NodeDisplacement
    ) = ()
    loads: (
        tuple[NodePointLoad, ...] | list[NodePointLoad] | NodePointLoad
    ) = ()

    def __post_init__(self):
        for param in (self.u, self.w, self.phi):
            if isinstance(param, str) and param not in ('fixed', 'free'):
                raise ValueError(
                    f'"{param}" is an invalid argument. Has to be either '
                    f'"fixed" or "free" or a real number.'
                )
            elif param == 0:
                raise ValueError(
                    'Please set u, w or phi to "free" instead of zero.'
                )
        if isinstance(self.displacements, NodeDisplacement):
            self.displacements = self.displacements,
        self.displacements = tuple(self.displacements)
        if isinstance(self.loads, NodePointLoad):
            self.loads = self.loads,
        self.loads = tuple(self.loads)

    @cached_property
    def displacement(self):
        """The overall node displacement as a 3x1 vector.

        Returns
        -------
        :any:`numpy.array`
            Sum of all displacements specified in :py:attr:`displacements`.

        See Also
        --------
        :py:class:`NodeDisplacement`

        Notes
        -----
            If no displacements were specified, then a 3x1 zero vector is
            returned.

        Examples
        --------
            >>> from sstatics import Node
            >>> Node(1, 2).displacement
            array([[0], [0], [0]])

            >>> from sstatics import NodeDisplacement
            >>> displacements = (NodeDisplacement(1.5, 2, 0.5),
            >>>                  NodeDisplacement(-2, 3, -0.3))
            >>> Node(-1, 3, displacements=displacements).displacement
            array([[-0.5], [5], [0.2]])
        """
        if len(self.displacements) == 0:
            return np.array([[0], [0], [0]])
        return np.sum([d.vector for d in self.displacements], axis=0)

    # TODO: docu
    @cached_property
    def load(self):
        r"""Rotate the node loads.

        Every load from :py:attr:`loads` is rotated by the node's rotation by
        calling :any:`PointLoad.rotate`. So the resulting rotation angle is
        calculated by :python:`alpha = load.rotation - node.rotation`. Thus
        each rotated load is computed by

        .. math::
            \left(\begin{array}{c}
            \cos(\alpha) & \sin(\alpha) & 0 \\
            -\sin(\alpha) & \cos(\alpha) & 0 \\
            0 & 0 & 1
            \end{array}\right) \cdot
            \left(\begin{array}{c} x \\ z \\ \varphi \end{array}\right)

        and summed up afterward.

        Returns
        -------
        :any:`numpy.array`
            Sum of all loads specified in :py:attr:`loads`.

        See Also
        --------
        :py:class:`NodePointLoad`

        Notes
        -----
            If no loads were specified, then a 3x1 zero vector is returned.

        Examples
        --------
            >>> from sstatics import Node, NodePointLoad
            >>> import numpy
            >>> load = NodePointLoad(1, 2, 0.5, rotation=2 * numpy.pi)
            >>> Node(6, 5, rotation=numpy.pi, loads=(load,)).rotate_load()
            array([[-1], [-2], [0.5]])
        """
        if len(self.loads) == 0:
            return np.array([[0], [0], [0]])
        return np.sum(
            [load.rotate(self.rotation) for load in self.loads], axis=0
        )

    @cached_property
    def elastic_support(self):
        """ TODO """
        u = 0 if isinstance(self.u, str) else self.u
        w = 0 if isinstance(self.w, str) else self.w
        phi = 0 if isinstance(self.phi, str) else self.phi
        return np.diag([u, w, phi])

    def same_location(self, other):
        """Determine if two nodes have exactly the same :py:attr:`x`- and
        :py:attr:`z`-coordinates.

        Parameters
        ----------
        other : :any:`Node`
            Second node to compare coordinates to.

        Returns
        -------
        :any:`bool`
            :python:`True` if the nodes have exactly the same coordinates,
            :python:`False` otherwise.

        Examples
        --------
            >>> from sstatics import Node
            >>> node = Node(1, 2)
            >>> node.same_location(Node(1, 2))
            True
            >>> node.same_location(Node(1, -2))
            False
        """
        return self.x == other.x and self.z == other.z


@dataclass(eq=False)
class CrossSection:
    r"""Create a cross-section for a statical system.

    Parameters
    ----------
    mom_of_int : :any:`float`
        Moment of inertia (:math:`I`), a measure of an object's resistance to
        rotational acceleration.
    area : :any:`float`
        Area (:math:`A`), the cross-sectional area of the system.
    height : :any:`float`
        Height (:math:`h`) of the cross-section.
    width : :any:`float`
        Width of the cross-section.
    shear_cor : :any:`float`
        Shear correction factor (:math:`\kappa`), a dimensionless parameter.

    Raises
    ------
    ValueError
        :py:attr:`mom_of_int`, :py:attr:`area`, :py:attr:`height`,
        :py:attr:`width` and :py:attr:`shear_cor` have to be greater than
        zero.
    ValueError
        :py:attr:`area` has to be less than :py:attr:`width` times
        :py:attr:`height` are set to zero.
    """

    mom_of_int: float
    area: float
    height: float
    width: float
    shear_cor: float

    def __post_init__(self):
        if self.mom_of_int <= 0:
            raise ValueError('mom_of_int has to be greater than zero.')
        if self.height <= 0:
            raise ValueError('height has to be greater than zero.')
        if self.width <= 0:
            raise ValueError('width has to be greater than zero.')
        if not 0 <= self.area <= self.height * self.width:
            raise ValueError(
                f'area has to be greater than or equal to zero or less than '
                f'or equal to {self.width * self.height}.')
        if self.shear_cor <= 0:
            raise ValueError('shear_cor has to be greater than zero.')


@dataclass(eq=False)
class Material:
    r"""Create a material for a statical system.

    Parameters
    ----------
    young_mod : :any:`float`
        Young's modulus (:math:`E`), a measure of the material's stiffness.
    poisson : :any:`float`
        Poisson's ratio (:math:`\nu`), the negative ratio of transverse to
        axial strain.
    shear_mod : :any:`float`
        Shear modulus (:math:`G`), a measure of the material's response
        to shear stress.
    therm_exp_coeff : :any:`float`
        Thermal expansion coefficient (:math:`\alpha_T`), describing how the
        material's dimensions change with temperature (in 1/K).

    Raises
    ------
    ValueError
        :py:attr:`young_mod`, :py:attr:`poisson`, :py:attr:`shear_mod`,
        and :py:attr:`therm_exp_coeff` have to be greater than
        zero.
    """

    young_mod: float
    poisson: float
    shear_mod: float
    therm_exp_coeff: float

    def __post_init__(self):
        if self.young_mod <= 0:
            raise ValueError('young_mod has to be greater than zero.')
        if self.poisson <= 0:
            raise ValueError('poisson has to be greater than zero.')
        if self.shear_mod <= 0:
            raise ValueError('shear_mod has to be greater than zero.')
        if self.therm_exp_coeff <= 0:
            raise ValueError('therm_exp_coeff has to be greater than zero.')


@dataclass(eq=False)
class BarLineLoad:
    """ TODO """

    pi: float
    pj: float
    direction: Literal['x', 'z'] = 'z'
    coord: Literal['bar', 'system'] = 'bar'
    length: Literal['exact', 'proj'] = 'exact'

    def __post_init__(self):
        if self.direction not in ('x', 'z'):
            raise ValueError('direction has to be either "x" or "z".')
        if self.coord not in ('bar', 'system'):
            raise ValueError('coord has to be either "bar" or "system".')
        if self.length not in ('exact', 'proj'):
            raise ValueError('length has to be either "exact" or "proj".')

    @cached_property
    def vector(self):
        """ TODO """
        vec = np.zeros((6, 1))
        vec[0 if self.direction == 'x' else 1] = self.pi
        vec[3 if self.direction == 'x' else 4] = self.pj
        return vec

    def rotate(self, rotation: float):
        """ TODO """
        if self.coord == 'bar':
            return self.vector
        sin, cos = np.sin(rotation), np.cos(rotation)
        m = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 0]])
        if self.length == 'proj':
            m = m @ np.diag([sin, cos, 0])
        m = np.vstack((
            np.hstack((m, np.zeros((3, 3)))), np.hstack((np.zeros((3, 3)), m))
        ))
        return m @ self.vector


@dataclass(eq=False)
class BarTemp:
    """Create a temperatur load case for a statical system.

    Parameters
    ----------
    temp_o : :any:`float`
        Temperature change above the neutral axis [K].
    temp_u : :any:`float`
        Temperature change below the neutral axis [K].

    Raises
    ------
    ValueError
        :py:attr:`temp_o`, :py:attr:`temp_u` have to be greater than or
        equal to zero since its unit is Kelvin.
    """

    temp_o: float
    temp_u: float

    def __post_init__(self):
        if self.temp_o < 0:
            raise ValueError(
                'temp_o has to be greater than or equal to zero since its '
                'unit is Kelvin.'
            )
        if self.temp_u < 0:
            raise ValueError(
                'temp_u has to be greater than or equal to zero since its '
                'unit is Kelvin.'
            )

    @cached_property
    def temp_s(self):
        """Calculates the uniform temperature change in the unit Kelvin.

        Returns
        -------
        :any:`float`
            Averaged value of temperature changes above and below the neutral
            axis in Kelvin.

        Example
        --------
        >>> from sstatics import BarTemp
        >>> temp = BarTemp(15, 30).temp_s
        22.5
        """
        return (self.temp_o + self.temp_u) / 2

    @cached_property
    def temp_delta(self):
        """Calculates the non-uniform temperature change in the unit Kelvin.

        Returns
        -------
        :any:`float`
            The temperature difference between the upper and lower side of the
            neutral axis, indicating the non-uniform temperature change in
            Kelvin.

        Example
        --------
        >>> from sstatics import BarTemp
        >>> temp_diff = BarTemp(10, 20).temp_delta
        10.0
        """
        return self.temp_u - self.temp_o


@dataclass(eq=False)
class BarPointLoad(PointLoad):
    """Creates a point load applied at a specific position along a bar.

    Parameters
    ----------
    x : :any:`float`
        The force component in the x-direction.
    z : :any:`float`
        The force component in the z-direction.
    phi : :any:`float`
        The moment applied along the beam.
    rotation : :any:`float`, default=0.0
        The rotation of the load components.
    position : :any:`float`, default=0.0
        Describes the relative position of the load along the bar. A value of
        `0` indicates the start of the bar, and `1` indicates the end of the
        bar.

    Notes
    -----
    This class models a point load applied to a bar (or beam) at a specific
    position. The load is applied in the x and z directions and includes a
    moment (phi) along the beam. The position is a normalized value between 0
    and 1, where 0 corresponds to the start of the bar and 1 corresponds to the
    end of the bar.

    Raises
    ------
    ValueError
        :py:attr:`position` has to be a value between 0 and 1.

    See Also
    -----
    :py:class:`NodePointLoad` and :py:class:`DegreesOfFreedom`

    """

    position: float = 0.0

    def __post_init__(self):
        if not (0 <= self.position <= 1):
            raise ValueError("position must be between 0 and 1")

    # TODO: test, docu
    def rotate(self):
        """ TODO """
        vec = super().rotate(rotation=0.0)
        if self.position == 0:
            return np.vstack((vec, np.zeros((3, 1))))
        elif self.position == 1:
            return np.vstack((np.zeros((3, 1)), vec))
        else:
            return np.zeros((6, 1))


# muss dringend zusammengefasst werden :$
# TODO: find solution for factor in EI, EA, GA_s, B_s
@dataclass(eq=False)
class Bar:
    """ TODO """

    node_i: Node
    node_j: Node
    cross_section: CrossSection
    material: Material
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
        """ TODO """
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
        """ TODO """
        return np.arctan2(
            -self.node_j.z + self.node_i.z, self.node_j.x - self.node_i.x
        )

    @cached_property
    def length(self):
        """ TODO """
        return np.sqrt(
            (self.node_j.x - self.node_i.x) ** 2 +
            (self.node_j.z - self.node_i.z) ** 2
        )

    @cached_property
    def hinge(self):
        """ TODO """
        return (
            self.hinge_u_i, self.hinge_w_i, self.hinge_phi_i,
            self.hinge_u_j, self.hinge_w_j, self.hinge_phi_j,
        )

    @cached_property
    def flexural_stiffness(self):
        """ TODO """
        EI = self.material.young_mod * self.cross_section.mom_of_int
        return EI if 'moment' in self.deformations else 1_000 * EI

    EI = property(lambda self: self.flexural_stiffness)
    """ Alias of :py:attr:`flexural_stiffness`. """

    @cached_property
    def modified_flexural_stiffness(self):
        """ TODO """
        if 'shear' in self.deformations:
            return self.EI * (1 + self.f0_first_order[0][0] / self.GA_s)
        else:
            return self.EI

    B_s = property(lambda self: self.modified_flexural_stiffness)
    """ Alias of :py:attr:`modified_flexural_stiffness`. """

    @cached_property
    def extensional_stiffness(self):
        """ TODO """
        EA = self.material.young_mod * self.cross_section.area
        return EA if 'normal' in self.deformations else 1_000 * EA

    EA = property(lambda self: self.extensional_stiffness)
    """ Alias of :py:attr:`extensional_stiffness`. """

    @cached_property
    def shear_stiffness(self):
        """ TODO """
        GA_s = self.material.shear_mod * self.cross_section.area
        GA_s *= self.cross_section.shear_cor
        return GA_s if 'shear' in self.deformations else 1_000 * GA_s

    GA_s = property(lambda self: self.shear_stiffness)
    """ Alias of :py:attr:`shear_stiffness`. """

    @cached_property
    def phi(self):
        """ TODO """
        return 12 * self.EI / (self.GA_s * self.length ** 2)

    @cached_property
    def characteristic_number(self):
        """ TODO """
        f0_x_i = self.f0_first_order[0][0]
        return np.sqrt(abs(f0_x_i) / self.B_s) * self.length

    @cached_property
    def line_load(self):
        """ TODO """
        if len(self.line_loads) == 0:
            return np.array([[0], [0], [0], [0], [0], [0]])
        return np.sum(
            [load.rotate(self.inclination) for load in self.line_loads], axis=0
        )

    @cached_property
    def point_load(self):
        """ TODO """
        if len(self.point_loads) == 0:
            return np.array([[0], [0], [0], [0], [0], [0]])
        return np.sum(
            [load.rotate() for load in self.point_loads], axis=0
        )

    @cached_property
    def f0_point_load(self):
        """ TODO """
        m = np.transpose(self.transformation_matrix(to_node_coord=True))
        return m @ self.point_load

    @cached_property
    def f0_temp(self):
        """ TODO """
        factor = self.material.therm_exp_coeff * self.material.young_mod
        f0_x = factor * self.temp.temp_s * self.cross_section.area
        f0_m = factor * self.temp.temp_delta * self.cross_section.mom_of_int
        f0_m /= self.cross_section.height
        return np.array([[f0_x], [0], [f0_m], [-f0_x], [0], [-f0_m]])

    @cached_property
    def f0_displacement(self):
        """ TODO """
        f0_displacement = np.vstack(
            (self.node_i.displacement, self.node_j.displacement)
        )
        trans_m = self.transformation_matrix(to_node_coord=False)
        k = self.stiffness_matrix(
            hinge_modification=False, to_node_coord=False
        )
        return k @ trans_m @ f0_displacement

    @cached_property
    def f0_first_order(self):
        """ TODO """
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
        return f0 * np.array([[-1], [-1], [1], [-1], [1], [-1]])

    @cached_property
    def f0_second_order_analytic(self):
        """ TODO """
        p_vec = self.line_load
        f0_x_i = self.f0_first_order[0][0]
        mu = self.characteristic_number
        B_s = self.B_s
        p_i, p_j = p_vec[1][0], p_vec[4][0]
        p_sum, p_diff = p_vec[1][0] + p_vec[4][0], p_vec[1][0] - p_vec[4][0]

        if f0_x_i < 0:
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
                    self.EI * self.length ** 2 * p_diff /
                    (B_s * self.GA_s * mu ** 3))

            f0_z_i = - (B_s * mu ** 2 / self.length ** 2) * c_2 - (
                    (self.length / mu ** 2)
                    + (self.EI / (self.GA_s * self.length))) * p_diff
            f0_z_j = - (
                B_s * mu ** 2 / self.length ** 2) * c_2 - p_i * self.length - (
                  (self.length / mu ** 2) + (
                    self.EI / (self.GA_s * self.length)) + (self.length / 2)
            ) * p_diff

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
                   ) * c_2 + (self.EI * self.length ** 2 * p_diff) / (
                          B_s * self.GA_s * mu ** 3)

            f0_z_i = (B_s * mu ** 2 / self.length ** 2) * c_2 + (
                    (self.length / mu ** 2) - (
                        self.EI / (self.GA_s * self.length))) * p_diff

            f0_z_j = (B_s * mu ** 2 / self.length ** 2) * c_2 - (
                    (p_sum + p_diff) * self.length / 2) + (
                    (self.length / mu ** 2) - (
                        self.EI / (self.GA_s * self.length))) * p_diff

            f0_m_i = -(B_s * mu ** 2 / self.length ** 2) * c_3 + (
                    (self.length ** 2 / mu ** 2) - (
                        self.EI / self.GA_s)) * p_sum

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
    def f0_second_order_taylor(self):
        """ TODO """
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
                12 * B_s / self.length) * (self.EI - B_s) * (p_j - p_i) / (
                12 * B_s + self.GA_s * self.length ** 2)

        f0_m_i = (4320 * B_s ** 3 * (p_j + p_i) + (
                6 * self.EI * self.GA_s ** 2 * self.length ** 4) * (
                p_j - p_i) + 60 * B_s * self.GA_s * self.length ** 2 * (
                12 * B_s * p_i - self.GA_s * self.length ** 2 * p_j) - (
                self.GA_s ** 3 * self.length ** 6) * (2 * p_j + 3 * p_i)) / (
                60 * self.GA_s * (
                    12 * B_s + self.GA_s * self.length ** 2) ** 2) - (
                self.EI * p_i) / self.GA_s + (6 * B_s / self.length) * (
                self.EI - B_s) * (p_j - p_i) / (
                12 * B_s + self.GA_s * self.length ** 2)

        f0_z_j = (self.length / 20) * (720 * B_s ** 2 * (p_j + p_i) - (
                4 * self.EI * self.GA_s * self.length ** 2) * (p_j - p_i) + (
                20 * B_s * self.GA_s * self.length ** 2) * (
                5 * p_j + 7 * p_i) + self.GA_s ** 2 * self.length ** 4 * (
                3 * p_j + 7 * p_i)) / (
                12 * B_s + self.GA_s * self.length ** 2) ** 2 - (
                self.EI * (p_j - p_i)) / (self.GA_s * self.length) - (
                self.length * (p_j + p_i)) / 2 - (12 * B_s / self.length) * (
                self.EI - B_s) * (p_j - p_i) / (
                12 * B_s + self.GA_s * self.length ** 2)

        f0_m_j = (4320 * B_s ** 3 * (p_j + p_i) - (
                6 * self.EI * self.GA_s ** 2 * self.length ** 4) * (
                p_j - p_i) + 60 * B_s * self.GA_s * self.length ** 2 * (
                12 * B_s * p_j - self.GA_s * self.length ** 2 * p_i) - (
                self.GA_s ** 3 * self.length ** 6) * (
                3 * p_j + 2 * p_i)) / (60 * self.GA_s * (
                    12 * B_s + self.GA_s * self.length ** 2) ** 2) - (
                self.EI * p_j) / self.GA_s - (6 * B_s / self.length) * (
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
    def stiffness_shear_force(self):
        """ TODO """
        return np.array([
            [1 + self.phi, 0, 0, 1 + self.phi, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 4 + self.phi, 0, 1, 2 - self.phi],
            [1 + self.phi, 0, 0, 1 + self.phi, 0, 0],
            [0, 1, 1, 0, 1, 1],
            [0, 1, 2 - self.phi, 0, 1, 4 + self.phi],
        ]) / (1 + self.phi)

    @cached_property
    def stiffness_second_order_analytic(self):
        """ TODO """
        f0_x_i = self.f0_first_order[0][0]
        mu = self.characteristic_number
        B_s = self.B_s
        factor = B_s / (self.GA_s * self.length ** 2)

        if f0_x_i < 0:
            sin_mu = np.sin(mu)
            cos_mu = np.cos(mu)
            denominator = 2 * (factor * mu ** 2 + 1) * (
                    cos_mu - 1) + mu * sin_mu
            f_1 = -(B_s / (12 * self.EI)) * (mu ** 3 * sin_mu) / denominator
            f_2 = (B_s / (6 * self.EI)) * (
                    (cos_mu - 1) * mu ** 2) / denominator
            f_3 = -(B_s / (4 * self.EI)) * (
                    factor * mu ** 2 + 1) * sin_mu * mu / denominator
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
            f_3 = (B_s / (4 * self.EI)) * (
                    factor * mu ** 2 - 1) * sinh_mu * mu / denominator
            f_4 = -(B_s / (2 * self.EI)) * (
                    factor * mu ** 2 - 1) * sinh_mu * mu / denominator

        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_3, 0, f_2, f_4],
                         [0, 0, 0, 1, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_4, 0, f_2, f_3]])

    @cached_property
    def stiffness_second_order_taylor(self):
        """ TODO """
        f0_x_i = self.f0_first_order[0][0]
        B_s = self.B_s
        factor = B_s / (self.GA_s * self.length ** 2)
        denominator_common = factor + 1 / 12
        denominator_squared = denominator_common ** 2
        inv_denominator_common = 1 / denominator_common

        f_1 = (B_s / (12 * self.EI * denominator_common) +
               f0_x_i * self.length ** 2 / (144 * self.EI) *
               (factor + 1 / 10) * inv_denominator_common ** 2)

        f_2 = (B_s / (12 * self.EI * denominator_common) +
               f0_x_i * self.length ** 2 / (8640 * self.EI) *
               inv_denominator_common ** 2)

        f_3 = (B_s * (factor + 1 / 3) / (
                4 * self.EI * denominator_common) +
               f0_x_i * self.length ** 2 / (48 * self.EI) *
               (1 / (240 * denominator_squared) + 1))

        f_4 = (-B_s * (factor - 1 / 6) / (
                2 * self.EI * denominator_common) +
               f0_x_i * self.length ** 2 / (24 * self.EI) *
               (1 / (240 * denominator_squared) - 1))

        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_3, 0, f_2, f_4],
                         [0, 0, 0, 1, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_4, 0, f_2, f_3]])

    @cached_property
    def stiffness_second_order_p_delta(self):
        """ TODO """
        c = self.f0_first_order[0][0] / self.length
        return np.array([
            [0, 0, 0, 0, 0, 0],
            [0, c, 0, 0, -c, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, -c, 0, 0, c, 0],
            [0, 0, 0, 0, 0, 0],
        ])

    @staticmethod
    def _validate_order_approach(order, approach):
        if order not in ('first', 'second'):
            raise ValueError('order has to be either "first" or "second".')
        if approach not in ('analytic', 'taylor', 'p_delta', None):
            raise ValueError(
                'approach has to be either "analytic", "taylor", "p_delta" or '
                'None.'
            )
        if approach == 'first' and approach is not None:
            raise ValueError('In first order the approach has to be None.')

    # TODO: Ludwigs Kriterium verwenden, wann man die analytische Lösung
    # TODO: verwenden kann?
    def f0(
        self, order: Literal['first', 'second'] = 'first',
        approach: Literal['analytic', 'taylor', 'p_delta'] | None = None,
        hinge_modification: bool = True, to_node_coord: bool = True
    ):
        """ TODO """
        self._validate_order_approach(order, approach)

        if order == 'first':
            f0 = self.f0_first_order
        else:
            if approach == 'analytic':
                f0 = self.f0_second_order_analytic
            elif approach == 'taylor':
                f0 = self.f0_second_order_taylor
            else:
                f0 = self.f0_first_order
        f0 += self.f0_temp + self.f0_displacement + self.f0_point_load

        if hinge_modification:
            k = self.stiffness_matrix(
                order, approach, hinge_modification=False, to_node_coord=False
            )
            for i, value in enumerate(self.hinge):
                if value:
                    f0 = f0 - 1 / k[i, i] * k[:, i:i + 1] * f0[i, :]
                    k = k - 1 / k[i, i] * k[:, i:i + 1] @ np.transpose(
                        k[:, i:i + 1]
                    )

        if to_node_coord:
            f0 = self.transformation_matrix() @ f0

        return f0

    # TODO: analytic + taylor und shear not in deform => Error?
    def stiffness_matrix(
        self, order: Literal['first', 'second'] = 'first',
        approach: Literal['analytic', 'taylor', 'p_delta'] | None = None,
        hinge_modification: bool = True, to_node_coord: bool = True
    ):
        """ TODO """
        self._validate_order_approach(order, approach)

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

        if order == 'first':
            if 'shear' in self.deformations:
                k @= self.stiffness_shear_force
        else:
            if approach == 'analytic':
                k @= self.stiffness_second_order_analytic
            elif approach == 'taylor':
                k @= self.stiffness_second_order_taylor
            else:
                if 'shear' in self.deformations:
                    k @= self.stiffness_shear_force
                    k += self.stiffness_second_order_p_delta
                else:
                    k += self.stiffness_second_order_p_delta

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

    def segment(self, positions: list[float] | None = None):
        """ TODO """
        if positions is None:
            positions = []
        for position in positions:
            if not 0 <= position <= 1.0:
                raise ValueError(
                    'All positions have to be in the interval [0, 1].'
                )

        pos, segmentation = defaultdict(list), False
        for load in self.point_loads:
            pos[load.position].append(load)
            if load.position not in (0.0, 1.0):
                segmentation = True
        if not segmentation:
            return [self]

        bars = []
        positions = list(set(positions) | set(pos.keys()))
        for position in sorted(positions):

            if position == 0.0:
                continue

            # calculate nodes
            node_i = bars[-1].node_j if bars else self.node_i
            c, s = np.cos(self.inclination), np.sin(self.inclination)
            x = self.node_i.x + c * position * self.length
            z = self.node_i.z - s * position * self.length
            node_loads = [
                NodePointLoad(load.x, load.z, load.phi, load.rotation)
                for load in pos[position]
            ]
            node_j = self.node_j if position == 1.0 else Node(
                x, z, loads=node_loads
            )

            # calculate bar line loads
            line_loads = []
            for i, line_load in enumerate(self.line_loads):
                pi = (
                    bars[-1].line_loads[i].pj if bars
                    else self.line_loads[i].pi
                )
                pj = line_load.pi + (line_load.pj - line_load.pi) * position
                line_loads.append(replace(line_load, pi=pi, pj=pj))

            # set point loads
            point_loads = []
            if not bars:
                point_loads = pos[0.0]
            elif position == 1.0:
                point_loads = pos[1.0]

            bars.append(replace(
                self, node_i=node_i, node_j=node_j, line_loads=line_loads,
                point_loads=point_loads
            ))

        return bars

    # TODO: refactor
    def deform_line(
        self, deform: ArrayLike, force: ArrayLike, scale: float = 1.0,
        lambdify: bool = True, n_points: int | None = None
    ):
        """ TODO """
        try:
            deform = np.reshape(deform, shape=(6, 1))
            force = np.reshape(force, shape=(6, 1))
        except ValueError:
            raise ValueError(
                'deform and force have to be array-like objects that can be '
                'reshaped to a flat numpy array with a length of 6.'
            )
        if n_points is not None and n_points <= 0:
            raise ValueError(
                'n_points has to be greater than zero or None.'
            )

        length = self.length + (deform[3][0] - deform[0][0]) * scale
        q_r, q_l = self.line_load[4][0], self.line_load[1][0]
        temp = self.material.therm_exp_coeff * self.temp.temp_delta
        a_5 = (q_r - q_l) / (120 * self.EI * length)
        a_4 = q_l / (24 * self.EI)
        a_3 = (-(q_r - q_l) / (self.GA_s * length) + force[1][0] / self.EI) / 6
        a_2 = force[2][0] / self.EI - q_l / self.GA_s
        a_2 = (a_2 - temp / self.cross_section.height) / 2
        a_1 = -deform[2][0] - force[1][0] / self.GA_s
        a_0 = deform[1][0]
        correction = sum([
            c * length ** i
            for i, c in enumerate((a_0, a_1, a_2, a_3, a_4, a_5))
        ])
        a_1 += (deform[4][0] - correction) / length
        a_0 += self.node_i.z

        x = sympy.Symbol('x')
        w = sum([
            scale * coeff * x ** i
            for i, coeff in enumerate((a_0, a_1, a_2, a_3, a_4, a_5))
        ])
        w_lambda = sympy.lambdify(x, w, modules='numpy')

        if n_points is None:
            return w_lambda if lambdify else w

        samples = np.linspace(start=0, stop=length, num=n_points)
        x = samples + self.node_i.x + deform[0][0] * scale
        z = w_lambda(samples)
        s, c = np.sin(self.inclination), np.cos(self.inclination)
        x, z = (
            self.node_i.x + c * (x - self.node_i.x) + s * (z - self.node_i.z),
            self.node_i.z - s * (x - self.node_i.x) + c * (z - self.node_i.z),
        )
        return x, z

    def max_deform(
        self, deform: np.array, force: np.array, n_points: int = 50
    ):
        """ TODO """
        x, z = self.deform_line(deform, force, n_points=n_points)

        x_ = np.linspace(0, self.length, n_points)
        x__ = np.linspace(deform[0][0], self.length + deform[3][0], n_points)

        dif = np.sqrt(np.square(x_ - x__) + np.square(z))
        idx = np.argmax(dif)
        return dif[idx], idx


@dataclass(eq=False)
class System:

    bars: tuple[Bar, ...] | list[Bar]

    # weitere Validierungen? sich schneidende Stäbe?
    def __post_init__(self):
        self.bars = tuple(self.bars)
        if len(self.bars) == 0:
            raise ValueError('There need to be at least one bar.')
        for i, bar in enumerate(self.bars[0:-1]):
            if any([
                bar.same_location(other_bar) for other_bar in self.bars[i + 1:]
            ]):
                raise ValueError(
                    'Cannot instantiate a system with bars that share the '
                    'same location.'
                )
        nodes = self.nodes(segmented=False)
        for i, node in enumerate(nodes[0:-1]):
            for other_node in nodes[i + 1:]:
                if node.same_location(other_node) and node != other_node:
                    raise ValueError(
                        'Inconsistent system. Nodes with the same location '
                        'need to be the same instance.'
                    )
        to_visit, visited = [nodes[0]], []
        while to_visit:
            current_node = to_visit.pop(0)
            if current_node not in visited:
                visited.append(current_node)
                to_visit += self.connected_nodes(segmented=False)[current_node]
        if set(visited) != set(nodes):
            raise ValueError("The system's graph needs to be connected.")

        self.segmented_bars = []
        for bar in self.bars:
            self.segmented_bars += bar.segment()
        self.segmented_bars = tuple(self.segmented_bars)

        self.dof = 3

    @cache
    def connected_nodes(self, segmented: bool = True):
        bars = self.segmented_bars if segmented else self.bars
        connections = {}
        for bar in bars:
            for node in (bar.node_i, bar.node_j):
                if node not in connections:
                    connections[node] = set()
            connections[bar.node_i].add(bar.node_j)
            connections[bar.node_j].add(bar.node_i)
        return {
            node: list(connected_nodes)
            for node, connected_nodes in connections.items()
        }

    @cache
    def nodes(self, segmented: bool = True):
        return list(self.connected_nodes(segmented=segmented).keys())

    def _get_zero_matrix(self):
        x = len(self.nodes()) * self.dof
        return np.zeros((x, x))

    def _get_zero_vec(self):
        x = len(self.nodes()) * self.dof
        return np.zeros((x, 1))

    def stiffness_matrix(self, order: str = 'first',
                         approach: str | None = None):
        k_system = self._get_zero_matrix()
        nodes = self.nodes()
        for bar in self.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            k = bar.stiffness_matrix()

            k_system[i:i + self.dof, i:i + self.dof] += k[:self.dof, :self.dof]
            k_system[i:i + self.dof, j:j + self.dof] += (
                k[:self.dof, self.dof:2 * self.dof])
            k_system[j:j + self.dof, i:i + self.dof] += (
                k[self.dof:2 * self.dof, :self.dof])
            k_system[j:j + self.dof, j:j + self.dof] += (
                k[self.dof:2 * self.dof, self.dof:2 * self.dof])
        return k_system

    def elastic_matrix(self):
        elastic = self._get_zero_matrix()
        nodes = self.nodes()
        for bar in self.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            el_bar = np.vstack((
                np.hstack((bar.node_i.elastic_support, np.zeros((3, 3)))),
                np.hstack((np.zeros((3, 3)), bar.node_j.elastic_support))
            ))

            elastic[i:i + self.dof, i:i + self.dof] = (
                el_bar)[:self.dof, :self.dof]
            elastic[i:i + self.dof, j:j + self.dof] = (
                el_bar[:self.dof, self.dof:2 * self.dof])
            elastic[j:j + self.dof, i:i + self.dof] = (
                el_bar[self.dof:2 * self.dof, :self.dof])
            elastic[j:j + self.dof, j:j + self.dof] = (
                el_bar[self.dof:2 * self.dof, self.dof:2 * self.dof])
        return elastic

    def system_matrix(self, order: str = 'first',
                      approach: str | None = None):
        return self.stiffness_matrix(order, approach) + self.elastic_matrix()

    def f0(self, order: str = 'first', approach: str | None = None):
        f0_system = self._get_zero_vec()
        nodes = self.nodes()
        for bar in self.segmented_bars:
            i = nodes.index(bar.node_i) * self.dof
            j = nodes.index(bar.node_j) * self.dof

            f0 = bar.f0()

            f0_system[i:i + self.dof, :] += f0[:self.dof, :]
            f0_system[j:j + self.dof, :] += f0[self.dof:2 * self.dof, :]
        return f0_system

    def p0(self):
        p0 = self._get_zero_vec()
        for i, node in enumerate(self.nodes()):
            p0[i * self.dof:i * self.dof + self.dof, :] = (
                node.load)
        return p0

    def p(self):
        return self.p0() - self.f0()

    def apply_boundary_conditions(self, order: str = 'first',
                                  approach: str | None = None):
        k = self.system_matrix(order, approach)
        p = self.p()
        for idx, node in enumerate(self.nodes()):
            node_offset = idx * self.dof
            for dof_nr, attribute in enumerate(['u', 'w', 'phi']):
                condition = getattr(node, attribute, 'free')
                if condition == 'fixed':
                    k[node_offset + dof_nr, :] = 0
                    k[:, node_offset + dof_nr] = 0
                    k[node_offset + dof_nr, node_offset + dof_nr] = 1
                    p[node_offset + dof_nr] = 0
        return k, p

    def node_deformation(self, order: str = 'first',
                         approach: str | None = None):
        modified_stiffness_matrix, modified_p = (
            self.apply_boundary_conditions(order, approach))
        return np.linalg.solve(modified_stiffness_matrix, modified_p)

    def bar_deform(self, order: str = 'first', approach: str | None = None):
        node_deform = self.node_deformation(order, approach)
        nodes = self.nodes()

        return [
            np.transpose(bar.transformation_matrix())
            @ np.vstack([
                node_deform[nodes.index(bar.node_i) *
                            self.dof: nodes.index(bar.node_i) * self.dof +
                            self.dof, :],
                node_deform[nodes.index(bar.node_j) *
                            self.dof: nodes.index(bar.node_j) * self.dof +
                            self.dof, :]
            ])
            for bar in self.segmented_bars
        ]

    def create_list_of_bar_forces(self, order: str = 'first',
                                  approach: str | None = None):
        bar_deform = self.bar_deform(order, approach)
        f_node = [
            bar.stiffness_matrix() @ deform +
            bar.f0()
            for bar, deform in zip(self.segmented_bars, bar_deform)
        ]
        return [
            np.transpose(
                bar.transformation_matrix()) @ forces - bar.f0_point_load
            for bar, forces in zip(self.segmented_bars, f_node)
        ]

    def _apply_hinge_modification(self, order: str = 'first',
                                  approach: str | None = None):
        deform_list = []
        bar_deform_list = self.bar_deform(order, approach)
        for i, bar in enumerate(self.segmented_bars):
            delta_rel = np.zeros((6, 1))
            if True in bar.hinge:
                k = bar.stiffness_matrix(order, approach,
                                         hinge_modification=False)
                bar_deform = bar_deform_list[i]
                f0 = bar.f0(order, approach, hinge_modification=False)

                idx = [i for i, value in enumerate(bar.hinge) if value]
                if idx:
                    rhs = (-np.dot(k[np.ix_(idx, range(6))], bar_deform)
                           - f0[idx])
                    delta_rel_reduced = (
                        np.linalg.solve(k[np.ix_(idx, idx)], rhs)
                    )
                    delta_rel[idx] = delta_rel_reduced
            deform_list.append(delta_rel)
        return deform_list

    def bar_deform_node_displacement(self):
        return [
            np.transpose(bar.transformation_matrix())
            @ np.vstack(
                (bar.node_i.displacement, bar.node_j.displacement))
            for bar in self.segmented_bars
        ]

    def create_bar_deform_list(self, order: str = 'first',
                               approach: str | None = None):
        hinge_modifications = self._apply_hinge_modification(order, approach)
        bar_deforms = self.bar_deform(order, approach)
        bar_deform_node_displacement = self.bar_deform_node_displacement()
        combined_results = []
        for i in range(len(hinge_modifications)):
            result = (hinge_modifications[i] + bar_deforms[i] +
                      bar_deform_node_displacement[i])
            combined_results.append(result)
        return combined_results

    def solvable(self, order: str = 'first', approach: str | None = None):
        k, p = self.apply_boundary_conditions(order, approach)
        if np.linalg.matrix_rank(k) < k.shape[0]:
            print("Stiffness matrix is singular.")
            return False
        return True


@dataclass(eq=False)
class Model:

    system: System
    order: Literal['first', 'second'] = 'first'
    approach: Literal['analytic', 'taylor', 'p_delta'] = None

    def calc(self):
        if self.order == 'first':
            if self.system.solvable(self.order, self.approach):
                deform = self.system.create_bar_deform_list(
                    self.order, self.approach)
                force = self.system.create_list_of_bar_forces(
                    self.order, self.approach)
                return deform, force
        elif self.order == 'second':
            # TODO: integrate MA Ludwig
            return None
