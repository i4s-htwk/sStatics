
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core import System, Bar


@dataclass
class SystemResult:

    system: System
    bar_deformations: list[np.ndarray]
    bar_forces: list[np.ndarray]

    def __post_init__(self):
        if len(self.system.segmented_bars) != len(self.bar_deformations):
            raise ValueError(
                'The number of bars in "system.segmented_bars" does not match '
                'the number of entries in "bar_deformations".'
            )
        if len(self.system.segmented_bars) != len(self.bar_forces):
            raise ValueError(
                'The number of bars in "system.segmented_bars" does not match '
                'the number of entries in "bar_forces".'
            )
        self.bar_results_discrete = [
            BarResult(bar, self.bar_deformations[i], self.bar_forces[i])
            for i, bar in enumerate(self.system.segmented_bars)
        ]

    @cached_property
    def length_discrete(self):
        return [result.length_discrete for result in self.bar_results_discrete]

    @cached_property
    def bar_deformations_discrete(self):
        return [
            result.bar_deformations_discrete
            for result in self.bar_results_discrete
        ]

    @cached_property
    def bar_forces_discrete(self):
        return [
            result.bar_forces_discrete for result in self.bar_results_discrete
        ]

    @cached_property
    def system_results_discrete(self):
        return self.bar_deformations_discrete, self.bar_forces_discrete


@dataclass
class BarResult:

    bar: Bar
    bar_deformations: np.ndarray
    bar_forces: np.ndarray
    discrete: int = 10

    def __post_init__(self):
        if self.bar_deformations.shape != (6, 1):
            raise ValueError('"bar_deformations" must have shape (6, 1).')
        if self.bar_forces.shape != (6, 1):
            raise ValueError('"bar_forces" must have shape (6, 1).')

    @cached_property
    def length_discrete(self):
        return (
            self.bar.length * np.linspace(0, 1, self.discrete + 1)
        )

    @cached_property
    def x_coef(self):
        l, EA = self.bar.length, self.bar.EA
        p0_ix, p0_jx = self.bar.line_load[0][0], self.bar.line_load[3][0]
        n, u = self.bar_forces[0][0], self.bar_deformations[0][0]
        dp0_x = p0_jx - p0_ix
        return np.array([
            [p0_ix, -n, u],
            [dp0_x / l, -p0_ix, -n / EA],
            [0, -dp0_x / (2 * l), p0_ix / (2 * EA)],
            [0, 0, dp0_x / (6 * l * EA)]
        ])

    @cached_property
    def z_coef(self):
        l, EI = self.bar.length, self.bar.EI
        p0_iz, p0_jz = self.bar.line_load[1][0], self.bar.line_load[4][0]
        v, m = self.bar_forces[1][0], self.bar_forces[2][0]
        w, phi = self.bar_deformations[1][0], self.bar_deformations[2][0]
        dp0_z = p0_jz - p0_iz
        return np.array([
            [p0_iz, -v, -m, phi, w],
            [dp0_z / l, -p0_iz, -v, -m / EI, -phi],
            [0, -dp0_z / (2 * l), -p0_iz / 2, -v / (2 * EI), m / (2 * EI)],
            [0, 0, -dp0_z / (6 * l), -p0_iz / (6 * EI), v / (6 * EI)],
            [0, 0, 0, -dp0_z / (24 * l * EI), p0_iz / (24 * EI)],
            [0, 0, 0, 0,  dp0_z / (120 * l * EI)]
        ])

    def get_ax(self, i: int):
        return self.x_coef[:, i]

    def get_az(self, i: int):
        return self.z_coef[:, i]

    def compute_discrete_polynomial(self, coef: np.ndarray):
        powers = np.vander(self.length_discrete, N=len(coef), increasing=True)
        return powers @ coef

    @cached_property
    def bar_deformations_discrete(self):
        coef = [self.get_ax(2), self.get_az(4), self.get_az(3)]
        return np.vstack([self.compute_discrete_polynomial(c) for c in coef]).T

    @cached_property
    def bar_forces_discrete(self):
        coef = [self.get_ax(1), self.get_az(1), self.get_az(2)]
        return np.vstack([self.compute_discrete_polynomial(c) for c in coef]).T
