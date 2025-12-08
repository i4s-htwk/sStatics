
from dataclasses import dataclass
from typing import Literal

from sstatics.core.preprocessing.system import Bar
from sstatics.core.solution.solver import Solver
from sstatics.core.postprocessing import BarStressDistribution
from sstatics.core.utils import get_differential_equation


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
            self.system, self.bar_deform_list, self.internal_forces,
            bar_index, n_disc
        )

    def stress_distribution(self, n_disc: int = 10):
        return [
            BarStressDistribution(
                bar=bar,
                deform=self.bar_deform_list[i],
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
            result_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: int | None = None, n_disc: int = 10
             ):
        from sstatics.graphic_objects import ResultGraphic
        from sstatics.core.postprocessing import SystemResult
        result = SystemResult(
            self.system, self.bar_deform_list, self.internal_forces,
            self.node_deform, self.node_support_forces,
            self.system_support_forces, n_disc=n_disc)
        ResultGraphic(
            result, kind, bar_mesh_type, result_mesh_type, decimals).show()

    def plot_stress(
            self,
            kind: Literal['normal', 'shear', 'bending_top', 'bending_bottom'],
            z: float | None = None,
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            result_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: int | None = None,
            n_disc: int = 10
    ):
        # TODO: not implemented yet
        stress = [
            sd.stress_at_z(z) if z is not None else sd.stress_disc
            for sd in self.stress_distribution(n_disc)
        ]
        print(stress)
        print(kind)
        print(bar_mesh_type)
        print(result_mesh_type)
        print(decimals)
