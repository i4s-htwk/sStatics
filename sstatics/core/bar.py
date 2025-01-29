
from collections import defaultdict
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
import sympy

from sstatics.core import (
    BarLineLoad, BarPointLoad, BarTemp, CrossSection, Node, NodePointLoad,
    Material, transformation_matrix
)


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

    def modified_flexural_stiffness(self, f_axial):
        """ TODO """
        if 'shear' in self.deformations:
            return self.EI * (1 + f_axial / self.GA_s)
        else:
            return self.EI

    def B_s(self, f_axial):
        return self.modified_flexural_stiffness(f_axial)
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
        EI = self.material.young_mod * self.cross_section.mom_of_int
        return GA_s if 'shear' in self.deformations else 1_000 * EI

    GA_s = property(lambda self: self.shear_stiffness)
    """ Alias of :py:attr:`shear_stiffness`. """

    @cached_property
    def phi(self):
        """ TODO """
        return 12 * self.EI / (self.GA_s * self.length ** 2)

    def characteristic_number(self, f_axial):
        """ TODO """
        return np.sqrt(abs(f_axial) / self.B_s(f_axial)) * self.length

    @cached_property
    def line_load(self):
        """ TODO """
        if len(self.line_loads) == 0:
            return np.array([[0], [0], [0], [0], [0], [0]])
        return np.sum(
            [load.rotate(self.inclination) for load in self.line_loads], axis=0
        )

    @property
    def point_load(self):
        """ TODO """
        if len(self.point_loads) == 0:
            return np.array([[0], [0], [0], [0], [0], [0]])
        return np.sum(
            [load.rotate() for load in self.point_loads], axis=0
        )

    @property
    def f0_point(self):
        """ TODO """
        m = np.transpose(self.transformation_matrix())
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
        trans_m = self.transformation_matrix()
        k = self.stiffness_matrix(
            hinge_modification=False
        )
        return k @ trans_m @ f0_displacement

    @cached_property
    def f0_line(self):
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
        return f0 * np.array([[-1], [-1], [1], [-1], [-1], [-1]])

    def f0_line_analytic(self, f_axial):
        """ TODO """
        p_vec = self.line_load
        mu = self.characteristic_number(f_axial)
        B_s = self.B_s(f_axial)
        p_i, p_j = p_vec[1][0], p_vec[4][0]
        p_sum, p_diff = p_vec[1][0] + p_vec[4][0], p_vec[1][0] - p_vec[4][0]

        if f_axial < 0:
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

    def f0_line_taylor(self, f_axial):
        """ TODO """
        p_vec = self.line_load
        B_s = self.B_s(f_axial)
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

        f0_z_j = f0_z_i - (p_j + p_i) * self.length / 2

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

    def stiffness_second_order_analytic(self, f_axial):
        """ TODO """
        mu = self.characteristic_number(f_axial)
        B_s = self.B_s(f_axial)
        factor = B_s / (self.GA_s * self.length ** 2)

        if f_axial < 0:
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

    def stiffness_second_order_taylor(self, f_axial):
        """ TODO """
        B_s = self.B_s(f_axial)
        factor = B_s / (self.GA_s * self.length ** 2)
        denominator_common = factor + 1 / 12
        denominator_squared = denominator_common ** 2
        inv_denominator_common = 1 / denominator_common

        f_1 = (B_s / (12 * self.EI * denominator_common) +
               f_axial * self.length ** 2 / (144 * self.EI) *
               (factor + 1 / 10) * inv_denominator_common ** 2)

        f_2 = (B_s / (12 * self.EI * denominator_common) +
               f_axial * self.length ** 2 / (8640 * self.EI) *
               inv_denominator_common ** 2)

        f_3 = (B_s * (factor + 1 / 3) / (
                4 * self.EI * denominator_common) +
               f_axial * self.length ** 2 / (48 * self.EI) *
               (1 / (240 * denominator_squared) + 1))

        f_4 = (-B_s * (factor - 1 / 6) / (
                2 * self.EI * denominator_common) +
               f_axial * self.length ** 2 / (24 * self.EI) *
               (1 / (240 * denominator_squared) - 1))

        return np.array([[1, 0, 0, 0, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_3, 0, f_2, f_4],
                         [0, 0, 0, 1, 0, 0],
                         [0, f_1, f_2, 0, f_1, f_2],
                         [0, f_2, f_4, 0, f_2, f_3]])

    def stiffness_second_order_p_delta(self, f_axial):
        """ TODO """
        c = f_axial / self.length
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
        if approach not in ('analytic', 'taylor', 'p_delta', 'iterativ', None):
            raise ValueError(
                'approach has to be either "analytic", "taylor", "p_delta", '
                '"iterativ" or ''None.'
            )
        if approach == 'first' and approach is not None:
            raise ValueError('In first order the approach has to be None.')

    # TODO: Ludwigs Kriterium verwenden, wann man die analytische Lösung
    # TODO: verwenden kann?
    def f0(

            self, order: Literal['first', 'second'] = 'first',
            approach:
            Literal['analytic', 'taylor', 'p_delta', 'iterativ'] | None = None,
            hinge_modification: bool = True, to_node_coord: bool = True,
            f_axial: float = 0
    ):
        """ TODO """
        self._validate_order_approach(order, approach)

        if order == 'first':
            f0_line = self.f0_line
        else:
            if approach == 'analytic':
                f0_line = self.f0_line_analytic(f_axial)
            elif approach == 'taylor':
                f0_line = self.f0_line_taylor(f_axial)
            else:
                f0_line = self.f0_line
        f0 = f0_line + self.f0_temp + self.f0_displacement - self.f0_point

        if hinge_modification:
            k = self.stiffness_matrix(
                order, approach, hinge_modification=False, to_node_coord=False,
                f_axial=f_axial
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
            approach:
            Literal['analytic', 'taylor', 'p_delta', 'iterativ'] | None = None,
            hinge_modification: bool = True, to_node_coord: bool = True,
            f_axial: float = 0
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
                k = k @ self.stiffness_shear_force
        else:
            if approach == 'analytic':
                k @= self.stiffness_second_order_analytic(f_axial)
            elif approach == 'taylor':
                k @= self.stiffness_second_order_taylor(f_axial)
            elif approach == 'p_delta':
                if 'shear' in self.deformations:
                    k = (k @ self.stiffness_shear_force +
                         self.stiffness_second_order_p_delta(f_axial))
                else:
                    k = k + self.stiffness_second_order_p_delta(f_axial)
            else:
                if 'shear' in self.deformations:
                    k @= self.stiffness_shear_force

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
