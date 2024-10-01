
import numpy as np
from typing import Optional

class CrossSection:

    def __init__(self, id: int, mom_of_int: float = 0, area: float = 0, height: float = 0, width: float = 0,
                 cor_far: float = 0, m_plast: float = 0):
        self.id = id
        self.mom_of_int = mom_of_int
        self.area = area
        self.height = height
        self.width = width
        self.cor_far = cor_far
        self.m_plast = m_plast


class Material:
    def __init__(self, id: int, young_mod: float = 0, poisson: float = 0, shear_mod: float = 0,
                 therm_exp_coeff: float = 0,
                 weight: float = 0):
        self.id = id
        self.young_mod = young_mod
        self.poisson = poisson
        self.shear_mod = shear_mod
        self.therm_exp_coeff = therm_exp_coeff
        self.weight = weight

class Angle:

    def __init__(self, rotation, kind='degree'):
        if kind not in ('degree', 'radiant'):
            raise ValueError("'kind' has to be either 'degree' or 'radiant'.")
        self.kind = kind
        self.rotation = rotation

    @property
    def degree(self):
        if self.kind == 'degree':
            return self.rotation % 360
        else:
            return np.rad2deg(self.rotation)

    @property
    def radiant(self):
        if self.kind == 'degree':
            return np.deg2rad(self.rotation)
        else:
            return self.rotation % (2 * np.pi)


class Transformation:

    def __init__(self, alpha: float = 0, beta: Optional[float] = 0):
        gamma = alpha - beta
        self.matrix = np.array([[np.cos(gamma), np.sin(gamma), 0],
                                [- np.sin(gamma), np.cos(gamma), 0],
                                [0, 0, 1]])


class NodeLoadCalc:

    def __init__(self, vec, rotation):
        self._rotation = rotation
        self._vec = vec

    def rotate(self, node_rot: Angle):
        """ rotates NodeLoad-vector in the node coordination-system """
        t_matrix = Transformation(self._rotation.radiant, node_rot.radiant).matrix
        load_vector = np.dot(t_matrix, self._vec)
        return load_vector


class NodeLoad:

    def __init__(self, x: float = 0, z: float = 0, phi: float = 0, rotation: float = 0):
        self.x = x
        self.z = z
        self.phi = phi
        self._rotation = Angle(rotation)

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = Angle(value)

    # @property
    # def vector(self):
    #     return NodeLoadCalc(np.array([[self.x], [self.z], [self.phi]]), self._rotation)

    def get_vector(self, n):
        return NodeLoadCalc(np.array([[self.x], [self.z], [self.phi]]), self._rotation).rotate(n)

    # @property
    # def plot(self):
    #     return NodeLoadView()


class Node:

    def __init__(self, id: int, x: float = 0, z: float = 0, rotation: float = 0, load: NodeLoad = None):

        self.id = id
        self.x = x
        self.z = z
        self._rotation = Angle(rotation)
        self.load = load

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = Angle(value)





class BarLoad:

    def __init__(self, id: int, pi: float = 0, pj: float = 0, local_x: any = None, local_z: any = None, global_x: any = None,
                 global_z: any = None, pro_length: any = None, true_length: any = None):
        self.id = id
        self.pi = pi
        self.pj = pj
        self.local_x = local_x
        self.local_z = local_z
        self.global_x = global_x
        self.global_z = global_z
        self.pro_length = pro_length
        self.true_length = true_length

    # @property
    # def vector_distributed_loads(self):
    #     print(BarLoadCalc.set_p())
    #     return BarLoadCalc.set_p

class Bar:

    def __init__(self, id: int, node_i: Node, node_j: Node, cross_section: CrossSection, material: Material,
                 hinge_u_i: str = 'x', hinge_w_i: str = 'x', hinge_phi_i: str = 'x', hinge_u_j: str = 'x',
                 hinge_w_j: str = 'x',
                 hinge_phi_j: str = 'x', dis_loads: BarLoad = None):
        self.id = id
        self.node_i = node_i
        self.node_j = node_j
        self.material = material
        self.cross_section = cross_section
        self.hinge_u_i = hinge_u_i
        self.hinge_w_i = hinge_w_i
        self.hinge_phi_i = hinge_phi_i
        self.hinge_u_j = hinge_u_j
        self.hinge_w_j = hinge_w_j
        self.hinge_phi_j = hinge_phi_j
        self.dis_loads = dis_loads

    @property
    def alpha(self):
        """ bar inclination angle """
        return Angle(np.arctan2((-self.node_i.z + self.node_j.z), (self.node_j.x - self.node_i.x))).radiant

    @property
    def length(self):
        return np.sqrt((self.node_j.x - self.node_i.x) ** 2 + (self.node_j.z - self.node_i.z) ** 2)


class BarLoadCalc:

    def __init__(self, loads: BarLoad, bar: Bar):
        self.loads = loads
        self.bar = bar

    def set_p(self):
        p_vec = np.zeros((6,1))
        alpha = self.bar.alpha
        print(alpha)
        pi = self.loads.pi
        pj = self.loads.pj
        if self.loads.local_x == 'x':
            p_vec += np.array([[pi], [0], [0], [pj], [0], [0]])
        elif self.loads.local_z == 'x':
            p_vec += np.array([[0], [pi], [0], [0], [pj], [0]])
        elif self.loads.global_x == 'x':
            if self.loads.pro_length == 'x':
                sign = 1 if self.bar.alpha > 0 else -1
                p_vec += np.array([[sign * pi * np.cos(alpha) * np.sin(alpha)],
                                   [sign * pi * (np.sin(alpha)) ** 2],
                                   [0],
                                  [sign * pj * np.cos(alpha) * np.sin(alpha)],
                                  [sign * pj * (np.sin(alpha)) ** 2],
                                 [0]])
            elif self.loads.true_length == 'x':
                p_vec += np.array([[pi * np.cos(alpha)], [pi * np.sin(alpha)],[0],
                                   [pj * np.cos(alpha)], [pj * np.sin(alpha)], [0]])
        elif self.loads.global_z == 'x':
            if self.loads.pro_length == 'x':
                sign_1 = -1 if abs(alpha) <= 1/2 * np.pi else 1
                sign_2 = 1 if abs(alpha) <= 1/2 * np.pi else -1
                p_vec += np.array([[sign_1 * pi * np.cos(alpha) * np.sin(alpha)],
                                   [sign_2 * pi * (np.cos(alpha)) ** 2],
                                   [0],
                                   [sign_1 * pj * np.cos(alpha) * np.sin(alpha)],
                                   [sign_2 * pj * (np.cos(alpha)) ** 2],
                                   [0]])
            elif self.loads.true_length == 'x':
                p_vec += np.array([[- pi * np.sin(alpha)], [pi * np.cos(alpha)], [0],
                                   [- pj * np.sin(alpha)], [pj * np.cos(alpha)], [0]])
        return p_vec