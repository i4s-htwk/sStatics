
from dataclasses import dataclass
from functools import cached_property

import numpy as np

from sstatics.core.preprocessing.system import System
from sstatics.core.preprocessing.bar import Bar


@dataclass
class SystemResult:

    system: System
    deforms: list[np.ndarray]
    forces: list[np.ndarray]
    n_disc: int = 10

    def __post_init__(self):
        if len(self.system.segmented_bars) != len(self.deforms):
            raise ValueError(
                'The number of bars in "system.segmented_bars" does not match '
                'the number of entries in "deforms".'
            )
        if len(self.system.segmented_bars) != len(self.forces):
            raise ValueError(
                'The number of bars in "system.segmented_bars" does not match '
                'the number of entries in "forces".'
            )
        self.results_disc = [
            BarResult(bar, self.deforms[i], self.forces[i], self.n_disc)
            for i, bar in enumerate(self.system.segmented_bars)
        ]

    @cached_property
    def length_disc(self):
        return [result.length_disc for result in self.results_disc]

    @cached_property
    def deforms_disc(self):
        return [result.bar_deforms_disc for result in self.results_disc]

    @cached_property
    def forces_disc(self):
        return [result.bar_forces_disc for result in self.results_disc]

    @cached_property
    def system_results_disc(self):
        return self.deforms_disc, self.forces_disc


@dataclass
class BarResult:

    bar: Bar
    bar_deforms: np.ndarray
    bar_forces: np.ndarray
    n_disc: int = 10

    def __post_init__(self):
        if self.bar_deforms.shape != (6, 1):
            raise ValueError('"bar_deforms" must have shape (6, 1).')
        if self.bar_forces.shape != (6, 1):
            raise ValueError('"bar_forces" must have shape (6, 1).')
        if self.n_disc < 1:
            raise ValueError('"n_disc" has to be greater than 0')

    @cached_property
    def length_disc(self):
        return np.linspace(0, self.bar.length, self.n_disc + 1)

    @cached_property
    def _x_coef(self):
        l, EA = self.bar.length, self.bar.EA
        p0_ix, p0_jx = self.bar.line_load[0][0], self.bar.line_load[3][0]
        n, u = self.bar_forces[0][0], self.bar_deforms[0][0]
        dp0_x = p0_jx - p0_ix
        return np.array([
            [p0_ix, -n, u],
            [dp0_x / l, -p0_ix, -n / EA],
            [0, -dp0_x / (2 * l), p0_ix / (2 * EA)],
            [0, 0, dp0_x / (6 * l * EA)]
        ])

    @cached_property
    def _z_coef(self):
        l, EI = self.bar.length, self.bar.EI
        p0_iz, p0_jz = self.bar.line_load[1][0], self.bar.line_load[4][0]
        v, m = self.bar_forces[1][0], self.bar_forces[2][0]
        w, phi = self.bar_deforms[1][0], self.bar_deforms[2][0]
        dp0_z = p0_jz - p0_iz
        return np.array([
            [p0_iz, -v, -m, phi, w],
            [dp0_z / l, -p0_iz, -v, -m / EI, -phi],
            [0, -dp0_z / (2 * l), -p0_iz / 2, -v / (2 * EI), m / (2 * EI)],
            [0, 0, -dp0_z / (6 * l), -p0_iz / (6 * EI), v / (6 * EI)],
            [0, 0, 0, -dp0_z / (24 * l * EI), p0_iz / (24 * EI)],
            [0, 0, 0, 0,  dp0_z / (120 * l * EI)]
        ])

    def _eval_poly(self, coef: np.ndarray):
        powers = np.vander(self.length_disc, N=len(coef), increasing=True)
        return powers @ coef

    @cached_property
    def bar_deforms_disc(self):
        coef = [self._x_coef[:, 2], self._z_coef[:, 4], self._z_coef[:, 3]]
        return np.vstack([self._eval_poly(c) for c in coef]).T

    @cached_property
    def bar_forces_disc(self):
        coef = [self._x_coef[:, 1], self._z_coef[:, 1], self._z_coef[:, 2]]
        return np.vstack([self._eval_poly(c) for c in coef]).T
