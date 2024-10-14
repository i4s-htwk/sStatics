
from dataclasses import asdict, dataclass, field, KW_ONLY, replace

import numpy as np


@dataclass
class NodeLoad:

    x: float
    z: float
    phi: float
    _: KW_ONLY
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
        diff = self.rotation - node_rotation
        transformation = np.array([
            [np.cos(diff), np.sin(diff), 0],
            [-np.sin(diff), np.cos(diff), 0],
            [0, 0, 1]
        ])
        x, z, phi = np.dot(transformation, self.vector).flatten().tolist()
        return replace(self, x=x, z=z, phi=phi, rotation=0)


@dataclass
class Node:

    x: float
    z: float
    _: KW_ONLY
    load: NodeLoad = field(default_factory=lambda: NodeLoad(0, 0, 0))
    rotation: float = 0

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
    weight: float

    def __eq__(self, other):
        return bool(np.isclose(
            list(asdict(self).values()), list(asdict(other).values())
        ).all())


class BarLoad:

    def __init__(
        self, pi: float = 0, pj: float = 0, local_x: any = None,
        local_z: any = None, global_x: any = None, global_z: any = None,
        pro_length: any = None, true_length: any = None
    ):
        self.pi = pi
        self.pj = pj
        self.local_x = local_x
        self.local_z = local_z
        self.global_x = global_x
        self.global_z = global_z
        self.pro_length = pro_length
        self.true_length = true_length


class Bar:

    def __init__(
        self, node_i: Node, node_j: Node,
        cross_section: CrossSection, material: Material,
        hinge_u_i: str = 'x', hinge_w_i: str = 'x', hinge_phi_i: str = 'x',
        hinge_u_j: str = 'x', hinge_w_j: str = 'x', hinge_phi_j: str = 'x',
        dis_loads: BarLoad = None
    ):
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
        return np.arctan2(
            -self.node_j.z + self.node_i.z, self.node_j.x - self.node_i.x
        )

    @property
    def length(self):
        return np.sqrt(
            (self.node_j.x - self.node_i.x) ** 2 +
            (self.node_j.z - self.node_i.z) ** 2
        )


class BarLoadCalc:

    def __init__(self, loads: BarLoad, bar: Bar):
        self.loads = loads
        self.bar = bar

    def set_p(self):
        p_vec = np.zeros((6, 1))
        alpha = self.bar.alpha
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
                p_vec += np.array([[pi * np.cos(alpha)],
                                   [pi * np.sin(alpha)],
                                   [0],
                                   [pj * np.cos(alpha)],
                                   [pj * np.sin(alpha)],
                                   [0]])
        elif self.loads.global_z == 'x':
            if self.loads.pro_length == 'x':
                sign_1 = -1 if abs(alpha) <= 1/2 * np.pi else 1
                sign_2 = 1 if abs(alpha) <= 1/2 * np.pi else -1
                p_vec += np.array(
                    [[sign_1 * pi * np.cos(alpha) * np.sin(alpha)],
                     [sign_2 * pi * (np.cos(alpha)) ** 2],
                     [0],
                     [sign_1 * pj * np.cos(alpha) * np.sin(alpha)],
                     [sign_2 * pj * (np.cos(alpha)) ** 2],
                     [0]]
                )
            elif self.loads.true_length == 'x':
                p_vec += np.array([[- pi * np.sin(alpha)],
                                   [pi * np.cos(alpha)],
                                   [0],
                                   [- pj * np.sin(alpha)],
                                   [pj * np.cos(alpha)],
                                   [0]]
                                  )
        return p_vec
