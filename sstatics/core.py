
from dataclasses import dataclass, field
from functools import cached_property
from typing import Literal, List, Optional

import numpy as np


def transformation_matrix(alpha: float):
    return np.array([
        [np.cos(alpha), np.sin(alpha), 0],
        [-np.sin(alpha), np.cos(alpha), 0],
        [0, 0, 1]
    ])


@dataclass(eq=False)
class DegreesOfFreedom:
    x: float
    z: float
    phi: float

    @cached_property
    def vector(self):
        return np.array([[self.x], [self.z], [self.phi]])


NodeDisplacement = DegreesOfFreedom


# TODO: name PointLoad, ...?
@dataclass(eq=False)
class NodeLoad(DegreesOfFreedom):

    rotation: float = 0.0

    def rotate(self, rotation: float):
        return transformation_matrix(self.rotation - rotation) @ self.vector


# TODO: load -> loads
# TODO: validation for load and displacements?
@dataclass(eq=False)
class Node:

    x: float
    z: float
    rotation: float = 0.0
    u: Literal['free', 'fixed'] | float = 'free'
    w: Literal['free', 'fixed'] | float = 'free'
    phi: Literal['free', 'fixed'] | float = 'free'
    load: NodeLoad = field(default_factory=lambda: NodeLoad(0, 0, 0))
    displacements: Optional[List[NodeDisplacement]] = field(
        default_factory=lambda: []
    )

    def __post_init__(self):
        for param in (self.u, self.w, self.phi):
            if isinstance(param, str) and param not in ('fixed', 'free'):
                raise ValueError(
                    f'"{param}" is an invalid argument. Has to be either '
                    f'"fixed" or "free" or a real number.'
                )
            # TODO: is this correct?
            elif param == 0:
                raise ValueError(
                    'Please set u, w or phi to "free" instead of zero.'
                )

    @cached_property
    def displacement(self):
        if len(self.displacements) == 0:
            return np.array([[0], [0], [0]])
        return np.sum([d.vector for d in self.displacements], axis=0)

    # TODO: name
    # TODO: test
    @cached_property
    def el_node(self):
        u = 0 if isinstance(self.u, str) else self.u
        w = 0 if isinstance(self.w, str) else self.w
        phi = 0 if isinstance(self.phi, str) else self.phi
        return np.diag([u, w, phi])

    def same_location(self, other):
        return self.x == other.x and self.z == other.z

    def rotate_load(self):
        return self.load.rotate(self.rotation)


# Warum area validierung nicht notwendig?
@dataclass(eq=False)
class CrossSection:

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
        if self.shear_cor <= 0:
            raise ValueError('shear_cor has to be greater than zero.')


@dataclass(eq=False)
class Material:

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
        vec = np.zeros((6, 1))
        vec[0 if self.direction == 'x' else 1] = self.pi
        vec[3 if self.direction == 'x' else 4] = self.pj
        return vec

    def rotate(self, rotation: float):
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
        return (self.temp_o + self.temp_u) / 2

    @cached_property
    def temp_delta(self):
        return self.temp_u - self.temp_o


@dataclass(eq=False)
class BarPointLoad(NodeLoad):
    # TODO: Documentation for variable position
    position: float = 0.0

    def __post_init__(self):
        if not (0 <= self.position <= 1):
            raise ValueError("position must be between 0 and 1")

    # warum nur 0 und 1 wichtig? weil stab geteilt wird?
    def rotate_load(self):
        vec = transformation_matrix(self.rotation) @ self.vector
        if self.position == 0:
            return np.vstack((vec, np.zeros((3, 1))))
        elif self.position == 1:
            return np.vstack((np.zeros((3, 1)), vec))
        else:
            return np.zeros((6, 1))


# validierung?
# muss dringend zusammengefasst werden :$
# TODO: find solution for factor in EI, EA, GA_s
@dataclass(eq=False)
class Bar:

    node_i: Node
    node_j: Node
    cross_section: CrossSection
    material: Material
    hinge_u_i: Optional[bool] = False
    hinge_w_i: Optional[bool] = False
    hinge_phi_i: Optional[bool] = False
    hinge_u_j: Optional[bool] = False
    hinge_w_j: Optional[bool] = False
    hinge_phi_j: Optional[bool] = False
    # was gibt es für optionen?
    deform: List[str] = field(default_factory=lambda: ['moment', 'normal'])
    line_loads: Optional[List[BarLineLoad]] = field(
        default_factory=lambda: [BarLineLoad(0, 0, 'z', 'bar', 'exact')])
    temp: Optional[BarTemp] = field(default_factory=lambda: BarTemp(0, 0))
    point_loads: Optional[List[BarPointLoad]] = field(
        default_factory=lambda: [BarPointLoad(0, 0, 0, 0, 0)])
    # segments?
    segments: Optional[int] = 0

    # TODO: Parameter sinnvoll? Wird es noch andere Rotationsmatrizen geben?
    def transformation_matrix(self, to_node_coord=True):
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
        return np.arctan2(
            -self.node_j.z + self.node_i.z, self.node_j.x - self.node_i.x
        )

    @cached_property
    def length(self):
        return np.sqrt(
            (self.node_j.x - self.node_i.x) ** 2 +
            (self.node_j.z - self.node_i.z) ** 2
        )

    @cached_property
    def hinge(self):
        return (
            self.hinge_u_i, self.hinge_w_i, self.hinge_phi_i,
            self.hinge_u_j, self.hinge_w_j, self.hinge_phi_j,
        )

    @cached_property
    def flexural_stiffness(self):
        EI = self.material.young_mod * self.cross_section.mom_of_int
        return EI if 'moment' in self.deform else 1_000 * EI

    EI = property(lambda self: self.flexural_stiffness)

    @cached_property
    def extensional_stiffness(self):
        EA = self.material.young_mod * self.cross_section.area
        return EA if 'normal' in self.deform else 1_000 * EA

    EA = property(lambda self: self.extensional_stiffness)

    @cached_property
    def shear_stiffness(self):
        GA_s = self.material.shear_mod * self.cross_section.area
        GA_s *= self.cross_section.shear_cor
        return GA_s if 'shear' in self.deform else 1_000 * GA_s

    GA_s = property(lambda self: self.shear_stiffness)

    @cached_property
    def phi(self):
        return 12 * self.EI / (self.GA_s * self.length ** 2)

    @cached_property
    def line_load(self):
        return np.sum(
            [load.rotate(self.inclination) for load in self.line_loads], axis=0
        )

    @cached_property
    def point_load(self):
        return np.sum(
            [load.rotate_load() for load in self.point_loads], axis=0
        )

    @cached_property
    def f0_load_first_order(self):
        p_vec = self.line_load

        f0_m_i = -(
                self.length ** 2 * (
                    30 * self.EI * p_vec[4][0] + 30 * self.EI * p_vec[1][0] +
                    2 * self.GA_s * self.length ** 2 * p_vec[4][0] +
                    3 * self.GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                720 * self.EI + 60 * self.GA_s * self.length ** 2
        )

        f0_m_j = -(
                self.length ** 2 * (
                    30 * self.EI * p_vec[4][0] + 30 * self.EI * p_vec[1][0] +
                    3 * self.GA_s * self.length ** 2 * p_vec[4][0] +
                    2 * self.GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                720 * self.EI + 60 * self.GA_s * self.length ** 2
        )

        f0_z_i = (
                self.length * (
                      40 * self.EI * p_vec[4][0] + 80 * self.EI * p_vec[1][0] +
                      3 * self.GA_s * self.length ** 2 * p_vec[4][0] +
                      7 * self.GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                240 * self.EI + 20 * self.GA_s * self.length ** 2
        )
        f0_z_j = -(
                self.length * (
                    80 * self.EI * p_vec[4][0] + 40 * self.EI * p_vec[1][0] +
                    7 * self.GA_s * self.length ** 2 * p_vec[4][0] +
                    3 * self.GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                240 * self.EI + 20 * self.GA_s * self.length ** 2
        )

        return np.array([
                [-(7 * p_vec[0][0] + 3 * p_vec[3][0]) * self.length / 20],
                [-f0_z_i],
                [-f0_m_i],
                [-(3 * p_vec[0][0] + 7 * p_vec[3][0]) * self.length / 20],
                [f0_z_j],
                [f0_m_j]
        ])

    @cached_property
    def f0_point_load(self):
        m = np.transpose(self.transformation_matrix(to_node_coord=True))
        return m @ self.point_load

    @cached_property
    def f0_temp(self):
        factor = self.material.therm_exp_coeff * self.material.young_mod
        f0_x = factor * self.temp.temp_s * self.cross_section.area
        f0_m = factor * self.temp.temp_delta * self.cross_section.mom_of_int
        f0_m /= self.cross_section.height
        return np.array([[f0_x], [0], [f0_m], [-f0_x], [0], [-f0_m]])

    @cached_property
    def f0_displacement(self):
        f0_displacement = np.vstack(
            (self.node_i.displacement, self.node_j.displacement)
        )
        trans_m = self.transformation_matrix(to_node_coord=False)
        return self.stiffness_matrix() @ trans_m @ f0_displacement

    @cached_property
    def _get_matrix_to_apply_shear_force(self):
        f_1 = 1 / (1 + self.phi)
        f_2 = (f_1 + self.phi / (4 * (1 + self.phi)))
        f_3 = (f_1 + self.phi / (2 * (1 + self.phi)))

        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, f_1, f_1, 0, f_1, f_1],
                         [0, f_1, f_2, 0, f_1, f_3],
                         [0, 0, 0, 1, 0, 0],
                         [0, f_1, f_1, 0, f_1, f_1],
                         [0, f_1, f_3, 0, f_1, f_2]])

    # separate properties?
    @cached_property
    def _prepare_factors_sec_order(self):
        p_vec = self.line_load
        f0_x_i = (-(7 * p_vec[0][0] + 3 * p_vec[3][0]) * self.length / 20)
        B_s = self.EI * (1 + f0_x_i / self.GA_s)
        return p_vec, f0_x_i, B_s

    @cached_property
    def _prepare_factors_sec_order_f0(self):
        p_vec, f0_x_i, B_s = self._prepare_factors_sec_order
        mu = np.sqrt(abs(f0_x_i) / B_s) * self.length

        return (f0_x_i, B_s, mu, (p_vec[1][0] + p_vec[4][0]),
                (p_vec[1][0] - p_vec[4][0]), p_vec[1][0], p_vec[4][0], p_vec)

    @cached_property
    def _f0_load_second_order_analytic(self):
        f0_x_i, B_s, mu, p_sum, p_diff, p_i, p_j, p_vec = (
            self._prepare_factors_sec_order_f0
        )

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
    def _f0_load_second_order_taylor(self):
        f0_x_i, B_s, mu, p_sum, p_diff, p_i, p_j, p_vec = (
            self._prepare_factors_sec_order_f0
        )

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
    def _prepare_factors_sec_order_stiffness_matrix(self):
        p_vec, f0_x_i, B_s = self._prepare_factors_sec_order
        factor = B_s / (self.GA_s * self.length ** 2)
        return f0_x_i, B_s, factor

    @cached_property
    def _apply_second_order_analytic_solution(self):
        f0_x_i, B_s, factor = (
            self._prepare_factors_sec_order_stiffness_matrix)
        mu = np.sqrt(abs(f0_x_i) / B_s) * self.length

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
    def _apply_second_order_approximate_by_taylor(self):
        f0_x_i, B_s, factor = (
            self._prepare_factors_sec_order_stiffness_matrix)
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
    def _apply_second_order_approximate_by_p_delta(self):
        factor = (self._prepare_factors_sec_order_stiffness_matrix[0] /
                  self.length)
        return np.array([[0, 0, 0, 0, 0, 0],
                         [0, factor, 0, 0, -factor, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, -factor, 0, 0, factor, 0],
                         [0, 0, 0, 0, 0, 0]])

    def _apply_hinge_modification(self, f0, stiffness_matrix):
        k = stiffness_matrix
        for i, value in enumerate(self.hinge):
            if value:
                idx = i
                f0 = f0 - 1 / k[i, i] * k[:, i:i + 1] * f0[i, :]
                k = k - 1 / k[idx, i] * k[:, i:i + 1] @ np.transpose(
                    k[:, i:i + 1])
        return f0, k

    def _transform_from_bar_in_node_coord(self, f0, stiffness_matrix):
        trans_m = self.transformation_matrix()
        return (
            trans_m @ f0,
            trans_m @ stiffness_matrix @ np.transpose(trans_m)
        )

    def _get_element_relation(self, f0, stiffness_matrix):
        # modification hinge
        if True in self.hinge:
            f0, stiffness_matrix = (
                self._apply_hinge_modification(f0, stiffness_matrix))

        # transformation
        if (self.node_i.rotation or self.node_j.rotation or
                self.inclination != 0):
            f0, stiffness_matrix = (
                self._transform_from_bar_in_node_coord(f0, stiffness_matrix))

        return f0, stiffness_matrix

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

    def f0(self, order: str = 'first', approach: Optional[str] = None):
        self._validate_order_approach(order, approach)
        if order == 'first':
            f0 = self.f0_load_first_order
        else:
            if approach == 'analytic':
                f0 = self._f0_load_second_order_analytic
            elif approach == 'taylor':
                f0 = self._f0_load_second_order_taylor
            else:
                f0 = self.f0_load_first_order
        return f0 + self.f0_temp + self.f0_displacement - self.f0_point_load

    # analytic + taylor und shear not in deform => Error?
    def stiffness_matrix(
        self, order: str = 'first', approach: Optional[str] = None
    ):
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
            if 'shear' in self.deform:
                return k @ self._get_matrix_to_apply_shear_force
            else:
                return k
        else:
            if approach == 'analytic':
                return k @ self._apply_second_order_analytic_solution
            elif approach == 'taylor':
                return k @ self._apply_second_order_approximate_by_taylor
            else:
                if 'shear' in self.deform:
                    return (
                        k @ self._get_matrix_to_apply_shear_force +
                        self._apply_second_order_approximate_by_p_delta
                    )
                else:
                    return k + self._apply_second_order_approximate_by_p_delta

    def element_relation(
        self, order: str = 'first', approach: Optional[str] = None
    ):
        return self._get_element_relation(
            self.f0(order, approach), self.stiffness_matrix(order, approach)
        )


@dataclass(eq=False)
class System:

    _bars: List[Bar]
    bars: List[Bar] = field(default_factory=lambda: [])

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
                        new_node_load = NodeLoad(point_load.x, point_load.z,
                                                 point_load.phi,
                                                 point_load.rotation)
                        new_node_j = Node(new_x, new_z, load=new_node_load)
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
                         approach: Optional[str] = None):
        k_system = self._get_zero_matrix()
        for bar in self.bars:
            i = self.nodes.index(bar.node_i) * self.dof
            j = self.nodes.index(bar.node_j) * self.dof

            k = bar.element_relation(order=order, approach=approach)[1]

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
                np.hstack((bar.node_i.el_node, np.zeros((3, 3)))),
                np.hstack((np.zeros((3, 3)), bar.node_j.el_node))
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
                      approach: Optional[str] = None):
        return self.stiffness_matrix(order, approach) + self.elastic_matrix()

    def f0(self, order: str = 'first', approach: Optional[str] = None):
        f0_system = self._get_zero_vec()
        for bar in self.bars:
            i = self.nodes.index(bar.node_i) * self.dof
            j = self.nodes.index(bar.node_j) * self.dof

            f0 = bar.element_relation(order=order, approach=approach)[0]

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
                                  approach: Optional[str] = None):
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
                         approach: Optional[str] = None):
        modified_stiffness_matrix, modified_p = (
            self.apply_boundary_conditions(order, approach))
        return np.linalg.solve(modified_stiffness_matrix, modified_p)

    def bar_deform(self, order: str = 'first', approach: Optional[str] = None):
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
                                  approach: Optional[str] = None):
        bar_deform = self.bar_deform(order, approach)
        f_node = [
            bar.element_relation(order, approach)[1] @ deform +
            bar.element_relation(order, approach)[0]
            for bar, deform in zip(self.bars, bar_deform)
        ]
        return [
            np.transpose(
                bar.transformation_matrix()) @ forces - bar.f0_point_load
            for bar, forces in zip(self.bars, f_node)
        ]

    def _apply_hinge_modification(self, order: str = 'first',
                                  approach: Optional[str] = None):
        deform_list = []

        for bar in self.bars:
            delta_rel = np.zeros((6, 1))
            k = bar.stiffness_matrix(order, approach)
            bar_deform = self.bar_deform(order, approach)
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

    def create_bar_deform_list(self, order: str = 'first',
                               approach: Optional[str] = None):
        hinge_modifications = self._apply_hinge_modification(order, approach)
        bar_deforms = self.bar_deform(order, approach)
        combined_results = []
        for i in range(len(hinge_modifications)):
            result = np.add(hinge_modifications[i], bar_deforms[i])
            combined_results.append(result)
        return combined_results

    def solvable(self, order: str = 'first', approach: Optional[str] = None,
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
    approach: Optional[Literal['analytic', 'taylor', 'p_delta']] = None
    tolerance = 1e-10

    def calc_deform(self):
        if self.order == 'first':
            if self.system.solvable(self.order, self.approach, self.tolerance):
                return self.system.create_bar_deform_list(
                    self.order, self.approach)
        elif self.order == 'second':
            # TODO: integrate MA Ludwig
            return None
