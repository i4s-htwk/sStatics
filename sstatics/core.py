
from dataclasses import asdict, dataclass, field, replace
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


@dataclass
class NodeLoad:

    x: float
    z: float
    phi: float
    rotation: float = 0

    def __post_init__(self):
        self.rotation = self.rotation % (2 * np.pi)

    def __eq__(self, other):
        return bool(np.isclose(
            list(asdict(self).values()), list(asdict(other).values())
        ).all())

    @property
    def vector(self):
        return np.array([[self.x], [self.z], [self.phi]])

    def rotate(self, node_rotation):
        x, z, phi = np.dot(
            get_transformation_matrix(self.rotation, node_rotation),
            self.vector
        ).flatten().tolist()
        return replace(self, x=x, z=z, phi=phi, rotation=0)


@dataclass
class Node:

    x: float
    z: float
    rotation: float = 0
    load: NodeLoad = field(default_factory=lambda: NodeLoad(0, 0, 0))

    def __eq__(self, other):
        return self.x == other.x and self.z == other.z

    def __post_init__(self):
        self.rotation = self.rotation % (2 * np.pi)
        self.load = replace(self.load)

    def rotate_load(self):
        rotated_load = self.load.rotate(self.rotation)
        return replace(self, load=rotated_load)


@dataclass
class CrossSection:

    mom_of_int: float
    area: float
    height: float
    width: float
    cor_far: float

    def __eq__(self, other):
        return bool(np.isclose(
            list(asdict(self).values()), list(asdict(other).values())
        ).all())


@dataclass
class Material:

    young_mod: float
    poisson: float
    shear_mod: float
    therm_exp_coeff: float

    def __eq__(self, other):
        return bool(np.isclose(
            list(asdict(self).values()), list(asdict(other).values())
        ).all())


@dataclass
class BarLoad:

    pi: float
    pj: float
    direction: Literal['x', 'z']
    coord: Literal['bar', 'system']
    length: Literal['exact', 'proj']

    @property
    def vector(self):
        vec = np.zeros((6, 1))
        vec[0 if self.direction == 'x' else 1] = self.pi
        vec[3 if self.direction == 'x' else 4] = self.pj
        return vec

    def rotate(self, bar_rotation):
        p_vec = self.vector
        if self.coord == 'system':
            if self.length == 'exact':
                if self.direction == 'x':
                    perm_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
                else:  # passt
                    perm_mat = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, 1]])
                perm_trans = (
                        perm_mat @ get_transformation_matrix(bar_rotation)
                )

                trans_mat = np.zeros((6, 6))
                trans_mat[:3, :3] = trans_mat[3:, 3:] = perm_trans
                return trans_mat @ p_vec
            else:
                if self.direction == 'x':
                    perm_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
                else:  # passt
                    perm_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
                perm_trans = (
                        perm_mat @ get_transformation_matrix(bar_rotation)
                )
                trans_mat = np.zeros((6, 6))
                trans_mat[:3, :3] = trans_mat[3:, 3:] = perm_trans
                mat_a = np.diag([1, 0, 0, 1, 0, 0])
                return (
                        (trans_mat @ mat_a @ np.transpose(trans_mat)) @ p_vec
                )
        else:
            return p_vec


@dataclass
class BarTemp:

    temp_o: float
    temp_u: float

    def __post_init__(self):
        self.temp_s = (self.temp_o + self.temp_u) / 2
        self.temp_delta = self.temp_u - self.temp_o


@dataclass
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
    load: Optional[List[BarLoad]] = field(
        default_factory=lambda: [BarLoad(0, 0, 'z', 'bar', 'exact')])
    temp: Optional[BarTemp] = field(default_factory=lambda: BarTemp(0, 0))

    def __post_init__(self):
        self.hinge = [
            self.hinge_u_i, self.hinge_w_i, self.hinge_phi_i,
            self.hinge_u_j, self.hinge_w_j, self.hinge_phi_j
        ]

    @property
    def rotation(self):
        """ bar inclination angle """
        return np.arctan2(
            -self.node_j.z + self.node_i.z, self.node_j.x - self.node_i.x
        )

    @property
    def length(self):
        return np.sqrt(
            (self.node_j.x - self.node_i.x) ** 2 +
            (self.node_j.z - self.node_i.z) ** 2
        )

    @property
    def EI(self):
        return self.material.young_mod * self.cross_section.mom_of_int

    @property
    def EA(self):
        return self.material.young_mod * self.cross_section.area

    @property
    def GA_s(self):
        return (
                self.material.shear_mod * self.cross_section.area *
                self.cross_section.cor_far
        )

    @property
    def phi(self):
        return 12 * self.EI / (self.GA_s * self.length ** 2)

    def get_p(self):
        p = np.zeros((6, 1))
        for load in self.load:
            p = p + load.rotate(self.rotation)
        return p

    def f0_load_first_order(self):
        p_vec = self.get_p()

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
        return (
            np.array(
                [[-(7 * p_vec[0][0] + 3 * p_vec[3][0]) * self.length / 20],
                 [-f0_z_i],
                 [-f0_m_i],
                 [-(3 * p_vec[0][0] + 7 * p_vec[3][0]) * self.length / 20],
                 [f0_z_j],
                 [f0_m_j]])
        )

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

    # First Order
    def f0_first_order(self):
        return self.f0_load_first_order() + self.f0_temp()

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

        # return np.array([[EA_l, 0, 0, -EA_l, 0, 0],
        #                  [0, 12 * EI_l3 * factor, -6 * EI_l2 * factor,
        #                   0, -12 * EI_l3 * factor, -6 * EI_l2 * factor],
        #                  [0, -6 * EI_l2 * factor,
        #                   EI_l * (4 + self.phi) * factor,
        #                   0, 6 * EI_l2 * factor,
        #                   EI_l * (2 - self.phi) * factor],
        #                  [-EA_l, 0, 0, EA_l, 0, 0],
        #                  [0, -12 * EI_l3 * factor, 6 * EI_l2 * factor,
        #                   0, 12 * EI_l3 * factor, 6 * EI_l2 * factor],
        #                  [0, -6 * EI_l2 * factor,
        #                   EI_l * (2 - self.phi) * factor,
        #                   0, 6 * EI_l2 * factor,
        #                   EI_l * (4 + self.phi) * factor]])

    def _get_matrix_to_apply_shear_force_to_stiffness_matrix(self):
        f_1 = 1 / (1 + self.phi)
        f_2 = (f_1 + self.phi / (4 * (1 + self.phi)))
        f_3 = (f_1 + self.phi / (2 * (1 + self.phi)))

        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, f_1, f_1, 0, f_1, f_1],
                         [0, f_1, f_2, 0, f_1, f_3],
                         [0, 0, 0, 1, 0, 0],
                         [0, f_1, f_1, 0, f_1, f_1],
                         [0, f_1, f_3, 0, f_1, f_2]])

    def stiffness_matrix_first_order(self):
        if self.phi != 0:
            return (
                (self._stiffness_matrix_without_shear_force() @
                 self._get_matrix_to_apply_shear_force_to_stiffness_matrix()))
        else:
            return self._stiffness_matrix_without_shear_force()

    # Second Order
    def __prepare_factors_sec_order(self):
        p_vec = self.get_p()
        f0_x_i = (-(7 * p_vec[0][0] + 3 * p_vec[3][0]) * self.length / 20)
        B_s = self.EI * (1 + f0_x_i / self.GA_s)
        return p_vec, f0_x_i, B_s

    def __prepare_factors_sec_order_f0(self):
        p_vec, f0_x_i, B_s = self.__prepare_factors_sec_order()
        mu = np.sqrt(abs(f0_x_i) / B_s) * self.length

        return (f0_x_i, B_s, mu, (p_vec[1][0] + p_vec[4][0]),
                (p_vec[1][0] - p_vec[4][0]), p_vec[1][0], p_vec[4][0], p_vec)

    def _f0_load_second_order_analytic(self):
        f0_x_i, B_s, mu, p_sum, p_diff, p_i, p_j, p_vec = (
            self.__prepare_factors_sec_order_f0()
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
            print(mu)

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

    def _f0_load_second_order_taylor(self):
        f0_x_i, B_s, mu, p_sum, p_diff, p_i, p_j, p_vec = (
            self.__prepare_factors_sec_order_f0()
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

    # f0
    def f0_second_order(self, approach: str = 'analytic'):
        if approach == 'analytic':
            return self._f0_load_second_order_analytic() + self.f0_temp()
        elif approach == 'taylor':
            return self._f0_load_second_order_taylor() + self.f0_temp()
        elif approach == 'p_delta':
            return self.f0_load_first_order() + self.f0_temp()

    def __prepare_factors_sec_order_stiffness_matrix(self):
        p_vec, f0_x_i, B_s = self.__prepare_factors_sec_order()
        factor = B_s / (self.GA_s * self.length ** 2)
        return f0_x_i, B_s, factor

    def _get_matrix_to_apply_second_order_analytic_solution(self):
        f0_x_i, B_s, factor = (
            self.__prepare_factors_sec_order_stiffness_matrix())
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

    def _get_matrix_to_apply_second_order_approximate_by_taylor(self):
        f0_x_i, B_s, factor = (
            self.__prepare_factors_sec_order_stiffness_matrix())
        denominator_common = factor + 1/12
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

    def _get_matrix_to_apply_second_order_approximate_by_p_delta(self):
        factor = (self.__prepare_factors_sec_order_stiffness_matrix()[0] /
                  self.length)
        return np.array([[0, 0, 0, 0, 0, 0],
                         [0, factor, 0, 0, -factor, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0],
                         [0, -factor, 0, 0, factor, 0],
                         [0, 0, 0, 0, 0, 0]])

    # stiffness matrix
    def stiffness_matrix_second_order(self, approach: str = 'analytic'):
        if approach == 'analytic':
            return (
                    self._stiffness_matrix_without_shear_force() @
                    self._get_matrix_to_apply_second_order_analytic_solution())
        elif approach == 'taylor':
            return (
                self._stiffness_matrix_without_shear_force() @
                self._get_matrix_to_apply_second_order_approximate_by_taylor())
        elif approach == 'p_delta':
            return (
                self.stiffness_matrix_first_order() +
                self._get_matrix_to_apply_second_order_approximate_by_p_delta()
                )
        else:
            return ValueError('calc_method has to either,....')

    # element relation
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

    def element_relation_first_order(self):
        return self._get_element_relation(self.f0_first_order(),
                                          self.stiffness_matrix_first_order())

    def element_relation_second_order(self, approach: str = 'analytic'):
        return (
            self._get_element_relation(
                self.f0_second_order(approach),
                self.stiffness_matrix_second_order(approach)))
