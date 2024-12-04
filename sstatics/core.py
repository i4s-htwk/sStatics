
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal

import numpy as np


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
    displacements: tuple[NodeDisplacement, ...] = ()
    loads: tuple[NodePointLoad, ...] = ()

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

    @cached_property
    def load(self):
        """The overall node load as a 3x1 vector.

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
            >>> from sstatics import Node
            >>> Node(1, 2).load
            array([[0], [0], [0]])

            >>> from sstatics import NodePointLoad
            >>> loads = (NodePointLoad(2.5, 3, 1.5), NodePointLoad(3, -5, 0.3))
            >>> Node(-1, 3, loads=loads).load
            array([[5.5], [-2], [1.8]])
        """
        if len(self.loads) == 0:
            return np.array([[0], [0], [0]])
        return np.sum([load.vector for load in self.loads], axis=0)

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

    def rotate_load(self):
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

        Examples
        --------
            >>> from sstatics import Node, NodePointLoad
            >>> import numpy
            >>> load = NodePointLoad(1, 2, 0.5, rotation=2 * numpy.pi)
            >>> Node(6, 5, rotation=numpy.pi, loads=(load,)).rotate_load()
            array([[-1], [-2], [0.5]])
        """
        return np.sum(
            [load.rotate(self.rotation) for load in self.loads], axis=0
        )


@dataclass(eq=False)
class CrossSection:
    """ TODO """

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
    """ TODO """

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
    """ TODO """

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
        """ TODO """
        return (self.temp_o + self.temp_u) / 2

    @cached_property
    def temp_delta(self):
        """ TODO """
        return self.temp_u - self.temp_o


@dataclass(eq=False)
class BarPointLoad(PointLoad):
    """ TODO """

    # TODO: Documentation for variable position
    position: float = 0.0

    def __post_init__(self):
        if not (0 <= self.position <= 1):
            raise ValueError("position must be between 0 and 1")

    # TODO: test
    # TODO: this should overwrite rotate method from PointLoad
    def rotate_load(self):
        """ TODO """
        vec = transformation_matrix(self.rotation) @ self.vector
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
    deformations: tuple[Literal['moment', 'normal', 'shear'], ...] = (
        'moment', 'normal'
    )
    line_loads: tuple[BarLineLoad, ...] = ()
    temp: BarTemp = field(default_factory=lambda: BarTemp(0, 0))
    point_loads: tuple[BarPointLoad, ...] = ()

    # TODO: other validations? validate hinges
    def __post_init__(self):
        if self.node_i.same_location(self.node_j):
            raise ValueError(
                'node_i and node_j need to have different locations.'
            )
        for d in self.deformations:
            if d not in ('moment', 'normal', 'shear'):
                raise ValueError(
                    'Valid deformation key words are "moment", "normal" and '
                    '"shear".'
                )
        # TODO: find a solution for this edge case
        if len(self.deformations) == 0:
            raise ValueError('There has to be at least one deformation.')

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
            [load.rotate_load() for load in self.point_loads], axis=0
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

    def w(self, x, deform: np.array, force: np.array, length: float):
        """ TODO """
        # x: [float, numpy.ndarray]
        line_load = self.line_load
        temp = (- self.material.therm_exp_coeff * self.temp.temp_delta /
                self.cross_section.height)
        c_1 = force[1][0] / self.EI
        c_2 = force[2][0] / self.EI - line_load[1][0] / self.GA_s
        c_3 = - deform[2][0] - force[1][0] / self.GA_s
        c_4 = deform[1][0]
        return (
                ((line_load[4][0] - line_load[1][0]) * (x ** 5) /
                 (120 * length * self.EI)) +
                (line_load[1][0] * (x ** 4) / (24 * self.EI)) -
                ((line_load[4][0] - line_load[1][0]) * (x ** 3) /
                 (6 * self.GA_s * length)) +
                (c_1 * (x ** 3)) / 6 +
                (c_2 * (x ** 2)) / 2 +
                c_3 * x +
                c_4 +
                1 / 2 * temp * x ** 2
        )

    def _get_points_deform_line(self, deform: np.array, force: np.array,
                                scale: float, num_points: int):
        deform = np.array(deform)
        force = np.array(force)

        if deform.shape != (6, 1):
            raise ValueError("deform must have the shape (6, 1), but has " +
                             str(deform.shape))
        if force.shape != (6, 1):
            raise ValueError("force must have the shape (6, 1), but has " +
                             str(deform.shape))

        new_length = self.length + (deform[3][0] - deform[0][0]) * scale
        sample_points = np.linspace(start=0, stop=new_length, num=num_points)

        correctur = ((deform[4][0] -
                     self.w(new_length, deform, force, new_length)) /
                     new_length)

        return [
            sample_points + self.node_i.x + deform[0][0] * scale,
            ((self.w(sample_points, deform, force, new_length) + correctur *
              sample_points) * scale + self.node_i.z)
        ]

    def _rotate_points(self, x: np.array, z: np.array):
        return [
            self.node_i.x + np.cos(self.inclination) * (
                    x - self.node_i.x) + np.sin(self.inclination) * (
                    z - self.node_i.z),
            self.node_i.z - np.sin(self.inclination) * (
                    x - self.node_i.x) + np.cos(self.inclination) * (
                    z - self.node_i.z)
        ]

    def deform_line(self, deform: np.array, force: np.array,
                    scale: float, num_points: int):
        """ TODO """
        x, z = self._get_points_deform_line(deform, force, scale, num_points)

        if self.inclination != 0:
            return self._rotate_points(x, z)
        return [x, z]

    def max_deform(self, deform: np.array, force: np.array, num_points: int):
        """ TODO """
        x, z = self._get_points_deform_line(deform, force, 1, num_points)

        x_ = np.linspace(0, self.length, 100)
        x__ = np.linspace(deform[0][0], self.length + deform[3][0], 100)

        dif = np.sqrt(np.square(x_ - x__) + np.square(z))
        idx = np.argmax(dif)
        return dif[idx], idx


@dataclass(eq=False)
class System:

    _bars: list[Bar]
    bars: list[Bar] = field(default_factory=lambda: [])

    def __post_init__(self):
        self.bars = self.bar_segmentation()
        self.nodes = self._get_nodes()
        self.dof = 3

    def _get_zero_matrix(self):
        x = len(self.nodes) * self.dof
        return np.zeros((x, x))

    def _get_zero_vec(self):
        x = len(self.nodes) * self.dof
        return np.zeros((x, 1))

    def _get_nodes(self):
        all_nodes = [n for bar in self.bars for n in (bar.node_i, bar.node_j)]
        unique_nodes = []
        for node in all_nodes:
            if any([node.same_location(n) for n in unique_nodes]) is False:
                unique_nodes.append(node)
        return unique_nodes

    def bar_segmentation(self):
        bars = []
        for bar in self._bars:
            positions = []
            bar_point_load = [[BarPointLoad(0, 0, 0, 0, 0)],
                              [BarPointLoad(0, 0, 0, 0, 0)]]

            for i in bar.point_loads:
                if i.position == 0:
                    bar_point_load[0].append(i)
                elif i.position == 1:
                    bar_point_load[1].append(i)
                else:
                    positions.append((i.position, i))

            for i in range(bar.segments):
                position = (i + 1) / (bar.segments + 1)
                if position not in [pos[0] for pos in positions]:
                    positions.append((position, None))

            positions.sort(key=lambda x: x[0])

            if positions:
                for i, (position, point_load) in enumerate(positions):
                    new_x = (bar.node_i.x + np.cos(bar.inclination) * position
                             * bar.length)
                    new_z = (bar.node_i.z - np.sin(bar.inclination) * position
                             * bar.length)

                    if point_load:
                        new_node_load = NodePointLoad(
                            point_load.x, point_load.z, point_load.phi,
                            point_load.rotation
                        )
                        new_node_j = Node(new_x, new_z, loads=[new_node_load])
                    else:
                        new_node_j = Node(new_x, new_z)

                    new_bar_line_load = []
                    for j, line_load in enumerate(bar.line_loads):
                        line_load_i = bars[-1].line_loads[j].pj if i \
                            else bar.line_loads[j].pi
                        new_line_load_j = (
                                (line_load.pj - line_load.pi) * position +
                                line_load.pi
                        )
                        new_bar_line_load.append(
                            BarLineLoad(line_load_i, new_line_load_j,
                                        line_load.direction, line_load.coord,
                                        line_load.length))

                    if i:
                        bars.append(Bar(bars[-1].node_j, new_node_j,
                                        bar.cross_section, bar.material,
                                        line_loads=new_bar_line_load,
                                        temp=bar.temp))
                    else:
                        bars.append(Bar(bar.node_i, new_node_j,
                                        bar.cross_section, bar.material,
                                        line_loads=new_bar_line_load,
                                        temp=bar.temp,
                                        point_loads=bar_point_load[0]))

                new_bar_line_load = []
                for i, line_load in enumerate(bar.line_loads):
                    new_bar_line_load.append(
                        BarLineLoad(bars[-1].line_loads[i].pj, line_load.pj,
                                    line_load.direction, line_load.coord,
                                    line_load.length))
                bars.append(Bar(bars[-1].node_j, bar.node_j, bar.cross_section,
                                bar.material, line_loads=new_bar_line_load,
                                temp=bar.temp, point_loads=bar_point_load[1]))

            else:
                bars.append(bar)

        return bars

    def stiffness_matrix(self, order: str = 'first',
                         approach: str | None = None):
        k_system = self._get_zero_matrix()
        for bar in self.bars:
            i = self.nodes.index(bar.node_i) * self.dof
            j = self.nodes.index(bar.node_j) * self.dof

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
        for bar in self.bars:
            i = self.nodes.index(bar.node_i) * self.dof
            j = self.nodes.index(bar.node_j) * self.dof

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
        for bar in self.bars:
            i = self.nodes.index(bar.node_i) * self.dof
            j = self.nodes.index(bar.node_j) * self.dof

            f0 = bar.f0()

            f0_system[i:i + self.dof, :] += f0[:self.dof, :]
            f0_system[j:j + self.dof, :] += f0[self.dof:2 * self.dof, :]
        return f0_system

    def p0(self):
        p0 = self._get_zero_vec()
        for i, node in enumerate(self.nodes):
            p0[i * self.dof:i * self.dof + self.dof, :] = (
                node.rotate_load())
        return p0

    def p(self):
        return self.p0() - self.f0()

    def apply_boundary_conditions(self, order: str = 'first',
                                  approach: str | None = None):
        k = self.system_matrix(order, approach)
        p = self.p()
        for idx, node in enumerate(self.nodes):
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

        return [
            np.transpose(bar.transformation_matrix())
            @ np.vstack([
                node_deform[self.nodes.index(bar.node_i) *
                            self.dof: self.nodes.index(bar.node_i) * self.dof +
                            self.dof, :],
                node_deform[self.nodes.index(bar.node_j) *
                            self.dof: self.nodes.index(bar.node_j) * self.dof +
                            self.dof, :]
            ])
            for bar in self.bars
        ]

    def create_list_of_bar_forces(self, order: str = 'first',
                                  approach: str | None = None):
        bar_deform = self.bar_deform(order, approach)
        f_node = [
            bar.stiffness_matrix() @ deform +
            bar.f0()
            for bar, deform in zip(self.bars, bar_deform)
        ]
        return [
            np.transpose(
                bar.transformation_matrix()) @ forces - bar.f0_point_load
            for bar, forces in zip(self.bars, f_node)
        ]

    def _apply_hinge_modification(self, order: str = 'first',
                                  approach: str | None = None):
        deform_list = []
        bar_deform_list = self.bar_deform(order, approach)
        for i, bar in enumerate(self.bars):
            delta_rel = np.zeros((6, 1))
            if True in bar.hinge:
                k = bar.stiffness_matrix(order, approach)
                bar_deform = bar_deform_list[i]
                f0 = bar.f0(order, approach)

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
            for bar in self.bars
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

    def solvable(self, order: str = 'first', approach: str | None = None,
                 tolerance: float = 1e-10):
        k, p = self.apply_boundary_conditions(order, approach)
        u, s, vt = np.linalg.svd(k)

        if np.any(s < tolerance):
            print("Stiffness matrix is singular.")
            if np.allclose(
                    k @ np.dot(np.linalg.pinv(k), p), p, atol=tolerance):
                print("The system has infinitely many solutions.")
            else:
                print("The system is inconsistent and has no solution.")
            return False
        else:
            return True


@dataclass(eq=False)
class Model:

    system: System
    order: Literal['first', 'second'] = 'first'
    approach: Literal['analytic', 'taylor', 'p_delta'] = None
    tolerance: float = 1e-12

    def calc(self):
        if self.order == 'first':
            if self.system.solvable(self.order, self.approach, self.tolerance):
                deform = self.system.create_bar_deform_list(
                    self.order, self.approach)
                force = self.system.create_list_of_bar_forces(
                    self.order, self.approach)
                return deform, force
        elif self.order == 'second':
            # TODO: integrate MA Ludwig
            return None
