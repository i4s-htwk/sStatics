
from dataclasses import dataclass
from typing import Literal

from sstatics.core.preprocessing.system import Bar
from sstatics.core.solution.solver import Solver
from sstatics.core.postprocessing.stress import BarStressDistribution
from sstatics.core.utils import get_differential_equation

from sstatics.core.postprocessing.graphic_objects.geo.system import (
    SystemGeo)
from sstatics.core.postprocessing.graphic_objects.geo.state_line import (
    StateLineGeo)

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
            # result_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: int | None = None,
            sig_digits: int | None = None,
            n_disc: int = 10,
            mode: str = 'mpl',
            text_color: 'str' = 'red',
    ):

        sys_geo = SystemGeo(self.system, mesh_type=bar_mesh_type)
        diff = self.differential_equation(n_disc=n_disc)
        kind_map = {
            'normal': ('forces_disc', 0),
            'shear': ('forces_disc', 1),
            'moment': ('forces_disc', 2),
            'u': ('deform_disc', 0),
            'w': ('deform_disc', 1),
            'phi': ('deform_disc', 2),
        }

        try:
            attr, idx = kind_map[kind]
        except KeyError:
            raise ValueError(f"Unbekannter kind: {kind}")

        # def result_i(x, z, translation, rotation):
        #     return dict(x=x, z=z, translation=translation, rotation=rotation)
        #
        #

        #
        # diff_idx = 0
        # for index, bar in enumerate(self.system.bars):
        #     print(index)
        #     mesh_seg = self.system.mesh_segments_of(bar)
        #     translation = (bar.node_i.x, bar.node_i.z)
        #     rotation = bar.inclination
        #
        #     x_plot = []
        #     z_plot = []
        #     offset = 0
        #     for i_seg, seg in enumerate(mesh_seg):
        #         print('i_seg: ',i_seg)
        #         diff_idx_seg = diff_idx + i_seg + index
        #         print('index in diff: ', diff_idx_seg)
        #         diff_seg = diff[diff_idx_seg]
        #
        #         x_diff = diff_seg.x
        #         data = getattr(diff_seg, attr)
        #         z = data[:, idx]
        #
        #         print(x_diff)
        #
        #         print(seg.length)
        #         offset = offset + seg.length
        #         if i_seg != 0:
        #             x = [x_value + offset for x_value in x_diff]
        #             x_plot.extend(x)
        #         else:
        #             x_plot.extend(x_diff)
        #         z_plot.extend(z)
        #         print(x_plot)
        #         print(z_plot)
        #
        #
        #
        #     diff_idx = diff_idx_seg

        result = []
        for i, bar in enumerate(self.system.mesh):
            data = getattr(diff[i], attr)
            x = diff[i].x
            z = data[:, idx]

            translation = (bar.node_i.x, bar.node_i.z)
            rotation = bar.inclination
            result.append(dict(
                x=x, z=z, translation=translation, rotation=rotation
            ))

        state_line = StateLineGeo(state_line_data=result,
                                  global_scale=sys_geo._base_scale,
                                  decimals=decimals,
                                  sig_digits=sig_digits,
                                  text_style={
                                      'textfont': dict(color=text_color), })

        ObjectRenderer([sys_geo, state_line], mode).show()

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
