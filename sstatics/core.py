
from dataclasses import asdict, dataclass, field, replace
from functools import cached_property

from typing import Literal, List, Optional
import numpy as np


def get_transformation_matrix(alpha: float, beta: float = 0):
    gamma = alpha - beta
    return np.array([[np.cos(gamma), np.sin(gamma), 0],
                     [- np.sin(gamma), np.cos(gamma), 0],
                     [0, 0, 1]])


def get_trans_mat_bar(
        rot_bar: float, rot_node_i: float = 0, rot_node_j: float = 0
):
    mat = np.zeros((6, 6))
    mat[:3, :3] = get_transformation_matrix(rot_bar, rot_node_i)
    mat[3:, 3:] = get_transformation_matrix(rot_bar, rot_node_j)
    return mat


@dataclass(eq=False)
class DegreesOfFreedom:
    x: float
    z: float
    phi: float

    @property
    def vector(self):
        return np.array([[self.x], [self.z], [self.phi]])


NodeDisplace = DegreesOfFreedom


@dataclass(eq=False)
class NodeLoad(DegreesOfFreedom):
    rotation: float = 0

    def __post_init__(self):
        self.rotation = self.rotation % (2 * np.pi)

    def __eq__(self, other):
        return bool(np.isclose(
            list(asdict(self).values()), list(asdict(other).values())
        ).all())

    # parameter neu setzen, kein replace
    def rotate(self, node_rotation):
        x, z, phi = np.dot(
            get_transformation_matrix(self.rotation, node_rotation),
            self.vector
        ).flatten().tolist()
        return replace(self, x=x, z=z, phi=phi, rotation=0)


@dataclass(eq=False)
class Node:

    x: float
    z: float
    rotation: float = 0
    u: Literal['free', 'fixed'] = 'free'
    w: Literal['free', 'fixed'] = 'free'
    phi: Literal['free', 'fixed'] = 'free'
    u_spring: Optional[float] = 0
    w_spring: Optional[float] = 0
    phi_spring: Optional[float] = 0
    load: NodeLoad = field(default_factory=lambda: NodeLoad(0, 0, 0))
    displacements: Optional[List[NodeDisplace]] = field(
        default_factory=lambda: [NodeDisplace(0, 0, 0)])

    # lieber dort wo verglichen wird
    def __eq__(self, other):
        return self.x == other.x and self.z == other.z

    # displacement replacen, validation der parameter
    def __post_init__(self):
        self.rotation = self.rotation % (2 * np.pi)
        self.load = replace(self.load)
        if self.u not in {'free', 'fixed'} and self.u_spring == 0:
            raise ValueError('u must be either "fixed" or "free".')
        if self.w not in {'free', 'fixed'} and self.w_spring == 0:
            raise ValueError('w must be either "fixed" or "free".')
        if self.phi not in {'free', 'fixed'} and self.phi_spring == 0:
            raise ValueError('phi must be either "fixed" or "free".')

    # kein replace -> gleich rotieren
    def rotate_load(self):
        rotated_load = self.load.rotate(self.rotation)
        return replace(self, load=rotated_load)

    @cached_property
    def displacement(self):
        return np.sum([load.vector for load in self.displacements], axis=0)


# validierung
@dataclass(eq=False)
class CrossSection:

    mom_of_int: float
    area: float
    height: float
    width: float
    cor_far: float


# validierung
@dataclass(eq=False)
class Material:

    young_mod: float
    poisson: float
    shear_mod: float
    therm_exp_coeff: float


# pi, pj validierung?
@dataclass(eq=False)
class BarLineLoad:

    pi: float
    pj: float
    direction: Literal['x', 'z']
    coord: Literal['bar', 'system']
    length: Literal['exact', 'proj']

    def __post_init__(self):
        if self.direction not in {'x', 'z'}:
            raise ValueError('direction has to be either "x" or "z".')
        if self.coord not in {'bar', 'system'}:
            raise ValueError('coord has to be either "bar" or "system".')
        if self.length not in {'exact', 'proj'}:
            raise ValueError('length has to be either "exact" or "proj".')

    @property
    def vector(self):
        vec = np.zeros((6, 1))
        vec[0 if self.direction == 'x' else 1] = self.pi
        vec[3 if self.direction == 'x' else 4] = self.pj
        return vec

    # kein rückgabewert, parameter werden überschrieben
    def rotate(self, bar_rotation):
        p_vec = self.vector
        if self.coord == 'system':
            trans_mat = np.zeros((3, 3))
            c = np.cos(bar_rotation)
            s = np.sin(bar_rotation)
            if self.length == 'exact':
                matrix_exact = np.array([[c, -s, 0], [s, c, 0], [0, 0, 0]])
                trans_mat = np.zeros((6, 6))
                trans_mat[:3, :3] = trans_mat[3:, 3:] = matrix_exact
            elif self.length == 'proj':
                matrix_proj = np.array(
                    [[s * c, -c * s, 0], [s ** 2, c ** 2, 0],
                     [0, 0, 0]])
                trans_mat = np.zeros((6, 6))
                trans_mat[:3, :3] = trans_mat[3:, 3:] = matrix_proj
            return trans_mat @ p_vec
        elif self.coord == 'bar':
            return p_vec


# validierung
@dataclass(eq=False)
class BarTemp:

    temp_o: float
    temp_u: float

    @cached_property
    def temp_s(self):
        return (self.temp_o + self.temp_u) / 2

    @cached_property
    def temp_delta(self):
        return self.temp_u - self.temp_o


@dataclass(eq=False)
class BarPointLoad(NodeLoad):
    # TODO: Documentation for variable position
    position: float = field(default=0)

    def __post_init__(self):
        self.rotation = self.rotation % (2 * np.pi)
        if not (0 <= self.position <= 1):
            raise ValueError("position must be between 0 and 1")

    def rotate_load(self):
        vec = get_transformation_matrix(self.rotation) @ self.vector
        if self.position == 0:
            return np.vstack((vec, np.zeros((3, 1))))
        elif self.position == 1:
            return np.vstack((np.zeros((3, 1)), vec))
        else:
            return np.zeros((6, 1))


# validierung?
# muss dringend zusammengefasst werden :$
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
    deform: List[str] = field(default_factory=lambda: ['moment', 'normal'])
    line_loads: Optional[List[BarLineLoad]] = field(
        default_factory=lambda: [BarLineLoad(0, 0, 'z', 'bar', 'exact')])
    temp: Optional[BarTemp] = field(default_factory=lambda: BarTemp(0, 0))
    point_loads: Optional[List[BarPointLoad]] = field(
        default_factory=lambda: [BarPointLoad(0, 0, 0, 0, 0)])
    # segments?
    segments: Optional[int] = 0

    # property?
    # line_load, temp, point_load replacen
    def __post_init__(self):
        self.node_i = replace(self.node_i)
        self.node_j = replace(self.node_j)
        self.material = replace(self.material)
        self.cross_section = replace(self.cross_section)

    @cached_property
    def rotation(self):
        """ bar inclination angle """
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
        return [
            self.hinge_u_i, self.hinge_w_i, self.hinge_phi_i,
            self.hinge_u_j, self.hinge_w_j, self.hinge_phi_j
        ]

    def _EI(self):
        return self.material.young_mod * self.cross_section.mom_of_int

    @cached_property
    def EI(self):
        return self._EI() * 1000 if 'moment' not in self.deform else self._EI()

    def _EA(self):
        return self.material.young_mod * self.cross_section.area

    @cached_property
    def EA(self):
        return self._EA() * 1000 if 'normal' not in self.deform else self._EA()

    def _GA_s(self):
        return (self.material.shear_mod * self.cross_section.area *
                self.cross_section.cor_far)

    @cached_property
    def GA_s(self):
        return (
            self._EI() * 1000 if 'shear' not in self.deform else self._GA_s()
        )

    @cached_property
    def phi(self):
        return 12 * self.EI / (self.GA_s * self.length ** 2)

    @cached_property
    def lineload(self):
        return np.sum(
            [load.rotate(self.rotation) for load in self.line_loads], axis=0)

    @cached_property
    def pointload(self):
        return np.sum(
            [load.rotate_load() for load in self.point_loads], axis=0)

    @cached_property
    def f0_load_first_order(self):
        p_vec = self.lineload

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
        print(self.pointload)
        return get_trans_mat_bar(self.rotation) @ self.pointload

    @cached_property
    def f0_displace(self):
        f0_displace = np.vstack(
            (self.node_i.displacement, self.node_j.displacement))
        return (self.stiffness_matrix() @ get_trans_mat_bar(self.rotation)
                @ f0_displace)

    # Fallunterscheidung nicht nötig? Was wenn cross_section.height = 0?
    # property
    @cached_property
    def f0_temp(self):
        if self.temp.temp_delta == 0 and self.temp.temp_s == 0:
            return np.zeros((2 * 3, 1))
        else:
            f0_x = (
                    self.material.therm_exp_coeff * self.temp.temp_s *
                    self.material.young_mod * self.cross_section.area
            )
            f0_m = (
                    (self.material.therm_exp_coeff * self.temp.temp_delta *
                     self.material.young_mod * self.cross_section.mom_of_int) /
                    self.cross_section.height
            )
            return np.array([[f0_x],
                             [0],
                             [f0_m],
                             [-f0_x],
                             [0],
                             [-f0_m]])

    # Name? -> Knotenklasse
    @cached_property
    def el_bar(self):
        return np.diag([
            self.node_i.u_spring, self.node_i.w_spring, self.node_i.phi_spring,
            self.node_j.u_spring, self.node_j.w_spring, self.node_j.phi_spring
        ])

    @cached_property
    def _stiffness_matrix_without_shear_force(self):
        EA_l = self.EA / self.length
        EI_l3 = self.EI / self.length ** 3
        EI_l2 = self.EI / self.length ** 2
        EI_l = self.EI / self.length

        return np.array([[EA_l, 0, 0, -EA_l, 0, 0],
                         [0, 12 * EI_l3, -6 * EI_l2,
                          0, -12 * EI_l3, -6 * EI_l2],
                         [0, -6 * EI_l2, 4 * EI_l,
                          0, 6 * EI_l2, 2 * EI_l],
                         [-EA_l, 0, 0, EA_l, 0, 0],
                         [0, -12 * EI_l3, 6 * EI_l2,
                          0, 12 * EI_l3, 6 * EI_l2],
                         [0, -6 * EI_l2, 2 * EI_l,
                          0, 6 * EI_l2, 4 * EI_l]])

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

    @cached_property
    def _prepare_factors_sec_order(self):
        p_vec = self.lineload
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
        trans_mat = get_trans_mat_bar(
            self.rotation, self.node_i.rotation, self.node_j.rotation
        )
        return (trans_mat @ f0,
                (trans_mat @ stiffness_matrix @ np.transpose(trans_mat)))

    def _get_element_relation(self, f0, stiffness_matrix):
        # modification hinge
        if True in self.hinge:
            f0, stiffness_matrix = (
                self._apply_hinge_modification(f0, stiffness_matrix))

        # transformation
        if self.node_i.rotation or self.node_j.rotation or self.rotation != 0:
            f0, stiffness_matrix = (
                self._transform_from_bar_in_node_coord(f0, stiffness_matrix))

        return f0, stiffness_matrix

    def f0(self, order: str = 'first', approach: Optional[str] = None):
        if order == 'first':
            if approach:
                return ValueError('in first order approach has to be "None"')
            f0 = self.f0_load_first_order
        elif order == 'second':
            if approach == 'analytic':
                f0 = self._f0_load_second_order_analytic
            elif approach == 'taylor':
                f0 = self._f0_load_second_order_taylor
            elif approach == 'p_delta':
                f0 = self.f0_load_first_order
            else:
                return ValueError(
                    'approach has to be either "analytic", '
                    '"taylor" or "p_delta".')
        else:
            return ValueError('order has to be either "first" or "second".')
        return (
                f0 + self.f0_temp + self.f0_displace - self.f0_point_load
        )

    def stiffness_matrix(self, order: str = 'first',
                         approach: Optional[str] = None):
        if order == 'first':
            if approach:
                return ValueError('in first order approach has to be "None"')
            if 'shear' in self.deform:
                return (
                    (self._stiffness_matrix_without_shear_force @
                     self._get_matrix_to_apply_shear_force))
            else:
                return self._stiffness_matrix_without_shear_force
        elif order == 'second':
            if approach == 'analytic':
                return (
                        self._stiffness_matrix_without_shear_force @
                        self._apply_second_order_analytic_solution)
            elif approach == 'taylor':
                return (
                        self._stiffness_matrix_without_shear_force @
                        self._apply_second_order_approximate_by_taylor)
            elif approach == 'p_delta':
                if 'shear' in self.deform:
                    return (
                            self._stiffness_matrix_without_shear_force @
                            self._get_matrix_to_apply_shear_force +
                            self._apply_second_order_approximate_by_p_delta)
                else:
                    return (self._stiffness_matrix_without_shear_force +
                            self._apply_second_order_approximate_by_p_delta)
            else:
                return ValueError(
                    'approach has to be either "analytic", '
                    '"taylor" or "p_delta".')
        else:
            return ValueError('order has to be either "first" or "second".')

    def element_relation(self, order: str = 'first',
                         approach: Optional[str] = None):
        if order == 'first':
            if approach:
                return ValueError('in first order approach has to be "None"')
        elif order == 'second':
            if approach not in {'analytic', 'taylor', 'p_delta'}:
                return ValueError(
                    'approach has to be either "analytic", '
                    '"taylor" or "p_delta".')
        else:
            return ValueError('order has to be either "first" or "second".')
        return (
            self._get_element_relation(
                self.f0(order, approach),
                self.stiffness_matrix(order, approach)))


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
        nodes = []
        for bar in self.bars:
            if bar.node_i not in nodes:
                nodes.append(bar.node_i)
            if bar.node_j not in nodes:
                nodes.append(bar.node_j)

        return nodes

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
                    new_x = (bar.node_i.x + np.cos(bar.rotation) * position
                             * bar.length)
                    new_z = (bar.node_i.z - np.sin(bar.rotation) * position
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

            el_bar = bar.el_bar

            elastic[i:i + self.dof, i:i + self.dof] += (
                el_bar)[:self.dof, :self.dof]
            elastic[i:i + self.dof, j:j + self.dof] += (
                el_bar[:self.dof, self.dof:2 * self.dof])
            elastic[j:j + self.dof, i:i + self.dof] += (
                el_bar[self.dof:2 * self.dof, :self.dof])
            elastic[j:j + self.dof, j:j + self.dof] += (
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
                node.rotate_load().load.vector)
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

    def bar_deform(self, order: str = 'first',
                   approach: Optional[str] = None):
        node_deform = self.node_deformation(order, approach)
        deform_list = []
        for idx, bar in enumerate(self.bars):
            i = self.nodes.index(bar.node_i) * self.dof
            j = self.nodes.index(bar.node_j) * self.dof
            # extract the deformation for node_i and node_j for every bar and
            # save them in a 6x1-vector
            deform_node = np.zeros((2 * self.dof, 1))
            deform_node[:self.dof, :] = node_deform[i:i + self.dof, :]
            deform_node[self.dof:2 * self.dof, :] = (
                node_deform[j:j + self.dof, :])
            # transform the deformation into the node coordination
            deform_bar = (np.transpose(get_trans_mat_bar(
                bar.rotation, bar.node_i.rotation, bar.node_j.rotation))
                          @ deform_node)
            deform_list.append(deform_bar)

        return deform_list

    def create_list_of_bar_forces(self, order: str = 'first',
                                  approach: Optional[str] = None):
        bar_deform = self.bar_deform(order, approach)[0]

        f_node = [
            k @ bar_deform + f0
            for bar in self.bars
            for f0, k in [bar.element_relation(order, approach)]
        ]

        f_bar = [
            np.transpose(get_trans_mat_bar(
                bar.rotation)) @ forces_node - bar.f0_point_load
            for bar, forces_node in zip(self.bars, f_node)
        ]

        return f_node, f_bar


# -> System
@dataclass(eq=False)
class Model:

    system: System
    order: Literal['first', 'second'] = 'first'
    approach: Optional[Literal['analytic', 'taylor', 'p_delta']] = None

    def calc(self):
        if self.order == 'first':
            # e.g. get list of bar deformation
            return (
                self.system.bar_deform(
                    self.order, self.approach))
        elif self.order == 'second':
            # TODO: integrate MA Ludwig
            return None
