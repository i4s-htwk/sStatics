
from dataclasses import asdict, dataclass, field, replace

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
    direction: str
    coord: str
    length: str

    @property
    def vector(self):
        if self.direction == 'x':
            return np.array([[self.pi], [0], [0], [self.pj], [0], [0]])
        elif self.direction == 'z':
            return np.array([[0], [self.pi], [0], [0], [self.pj], [0]])
        else:
            raise ValueError("'direction' has to be either 'x' or 'z'.")

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
                print(trans_mat)
                return trans_mat @ p_vec
            elif self.length == 'proj':
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
                print((trans_mat @ mat_a @ np.transpose(trans_mat)))
                return (
                        (trans_mat @ mat_a @ np.transpose(trans_mat)) @ p_vec
                )
            else:
                raise ValueError(
                    "'length' has to be either 'exact' or 'proj'."
                )
        elif self.coord == 'bar':
            return p_vec
        else:
            raise ValueError(
                "'coord' has to be either 'bar' or 'system'."
            )


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
    hinge_u_i: bool
    hinge_w_i: bool
    hinge_phi_i: bool
    hinge_u_j: bool
    hinge_w_j: bool
    hinge_phi_j: bool
    load: BarLoad
    temp: BarTemp

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

    def get_p(self):
        return self.load.rotate(self.rotation)

    def f0_load(self):
        p_vec = self.get_p()

        EI = self.material.young_mod * self.cross_section.mom_of_int
        GA_s = (
                self.material.shear_mod * self.cross_section.area *
                self.cross_section.cor_far
        )

        M_a = -(
                self.length ** 2 * (
                    30 * EI * p_vec[4][0] + 30 * EI * p_vec[1][0] +
                    2 * GA_s * self.length ** 2 * p_vec[4][0] +
                    3 * GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                720 * EI + 60 * GA_s * self.length ** 2
        )

        M_e = -(
                self.length ** 2 * (
                    30 * EI * p_vec[4][0] + 30 * EI * p_vec[1][0] +
                    3 * GA_s * self.length ** 2 * p_vec[4][0] +
                    2 * GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                720 * EI + 60 * GA_s * self.length ** 2
        )

        T_a = (
                self.length * (
                      40 * EI * p_vec[4][0] + 80 * EI * p_vec[1][0] +
                      3 * GA_s * self.length ** 2 * p_vec[4][0] +
                      7 * GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                240 * EI + 20 * GA_s * self.length ** 2
        )

        T_e = -(
                self.length * (
                    80 * EI * p_vec[4][0] + 40 * EI * p_vec[1][0] +
                    7 * GA_s * self.length ** 2 * p_vec[4][0] +
                    3 * GA_s * self.length ** 2 * p_vec[1][0]
                )
        ) / (
                240 * EI + 20 * GA_s * self.length ** 2
        )
        return (
            np.array(
                [[-(7 * p_vec[0][0] + 3 * p_vec[3][0]) * self.length / 20],
                 [-T_a],
                 [-M_a],
                 [-(3 * p_vec[0][0] + 7 * p_vec[3][0]) * self.length / 20],
                 [T_e],
                 [M_e]])
        )

    def f0_temp(self):
        if self.temp.temp_delta == 0 and self.temp.temp_s == 0:
            return np.zeros((2 * 3, 1))
        else:
            f0_i_x = (
                    self.material.therm_exp_coeff * self.temp.temp_s *
                    self.material.young_mod * self.cross_section.area
            )
            f0_i_m = (
                    (self.material.therm_exp_coeff * self.temp.temp_delta *
                     self.material.young_mod * self.cross_section.mom_of_int) /
                    self.cross_section.height
            )
            return np.array([[f0_i_x],
                             [0],
                             [f0_i_m],
                             [-f0_i_x],
                             [0],
                             [-f0_i_m]])

    @property
    def f0(self):
        return self.f0_load() + self.f0_temp()
