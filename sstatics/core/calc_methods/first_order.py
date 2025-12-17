
from dataclasses import dataclass
from typing import Literal

from sstatics.core.preprocessing.system import Bar
from sstatics.core.solution.solver import Solver
from sstatics.core.postprocessing.stress import BarStressDistribution
from sstatics.core.utils import (get_differential_equation, plot_results,
                                 plot_stress_results)

from sstatics.core.postprocessing.graphic_objects import ObjectRenderer


@dataclass(eq=False)
class FirstOrder(Solver):
    """
    First-order static solver derived from Solver.

    Inherits all functionality from Solver. Additionally, it validates that
    all bars in the system are instances of the base `Bar` class. If any
    `BarSecond` instances are found, a ValueError is raised.
    """

    def __post_init__(self):
        if hasattr(super(), '__post_init__'):
            super().__post_init__()

        for bar in self.system.bars:
            if not isinstance(bar, Bar):
                raise ValueError(
                    f"All bars must be instances of `Bar`. "
                    f"Found {type(bar).__name__}."
                )

    def differential_equation(
            self, bar_index: int | None = None, n_disc: int = 10
    ):
        return get_differential_equation(
            self.system, self.bar_deform_total, self.internal_forces,
            bar_index, n_disc
        )

    def stress_distribution(self, n_disc: int = 10):
        return [
            BarStressDistribution(
                bar=bar,
                deform=self.bar_deform_total[i],
                force=self.internal_forces[i],
                n_disc=n_disc,
            )
            for i, bar in enumerate(self.system.mesh)
        ]

    def plot(
            self,
            kind: Literal[
                'normal', 'shear', 'moment', 'u', 'w', 'phi',
                'bending_line'] = 'normal',
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            decimals: int = 2,
            sig_digits: int | None = None,
            n_disc: int = 10,
            mode: str = 'mpl',
            color: 'str' = 'red',
            show_load: bool = False
    ):
        valid_kinds = ['normal', 'shear', 'moment', 'u', 'w', 'phi',
                       'bending_line']
        if kind not in valid_kinds:
            raise ValueError(
                f"Invalid kind '{kind}'. "
                f"Expected one of {valid_kinds}."
            )

        diff = self.differential_equation(n_disc=n_disc)

        sys_geo, result_geo = plot_results(self.system, diff, kind,
                                           bar_mesh_type, decimals,
                                           sig_digits, color, show_load)

        ObjectRenderer([sys_geo, result_geo], mode).show()

    def plot_stress(
            self,
            kind: Literal[
                'normal', 'shear', 'bending', 'bending_top',
                'bending_bottom'] = 'normal',
            z: float | None = None,
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            decimals: int = 2,
            sig_digits: int | None = None,
            n_disc: int = 10,
            mode: str = 'mpl',
            color: 'str' = 'red',
            show_load: bool = False
    ):
        valid_kinds = ['normal', 'shear', 'bending',
                       'bending_top', 'bending_bottom']
        if kind not in valid_kinds:
            raise ValueError(
                f"Invalid kind '{kind}'. "
                f"Expected one of {valid_kinds}."
            )

        diff = self.stress_distribution(n_disc)

        sys_geo, result_geo = plot_stress_results(
            self.system, diff, kind, z, bar_mesh_type, decimals,
            sig_digits, color, show_load)

        ObjectRenderer([sys_geo, result_geo], mode).show()
