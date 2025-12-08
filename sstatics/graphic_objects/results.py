
import numpy as np
import math
from functools import cached_property
from typing import Literal, Tuple, List, Any

from sstatics.core.postprocessing.results import (
    BarResult, SystemResult, RigidBodyDisplacement)

from sstatics.graphic_objects.utils import SingleGraphicObject, transform
from sstatics.graphic_objects.geometry import LineGraphic
from sstatics.graphic_objects.system import SystemGraphic


class ResultGraphic(SingleGraphicObject):

    def __init__(
            self, system_result: SystemResult,
            kind: (
                Literal['normal', 'shear', 'moment', 'u', 'w', 'phi',
                        'bending_line']) = 'normal',
            bar_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            result_mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: int | None = None, sig_digits: int | None = None,
            **kwargs
    ):
        if not isinstance(system_result, SystemResult):
            raise TypeError(
                '"system_result" has to be an instance of SystemResult'
            )
        super().__init__(
            system_result.bars[0].bar.node_i.x,
            system_result.bars[0].bar.node_i.z,
            **kwargs
        )
        self.system_result = system_result
        self.system = system_result.system
        self._result_graphics = [
            SystemGraphic(
                self.system, bar_mesh_type, self._base_scale,
                scatter_options=self.scatter_kwargs,
                annotation_options=self.annotation_kwargs
            ),
            SystemResultGraphic(
                self.system_result, kind, result_mesh_type, decimals,
                sig_digits, self._base_scale,
                scatter_options=self.scatter_kwargs,
                annotation_options=self.annotation_kwargs
            )
        ]

    @cached_property
    def _base_scale(self):
        dims = self.system.max_dimensions
        return 0.08 * math.sqrt(dims[0] * 1.75 * dims[1])

    @property
    def annotations(self):
        return tuple(
            ann for rg in self._result_graphics for ann in rg.annotations
        )

    @property
    def traces(self):
        return [
            trace
            for rg in self._result_graphics
            for trace in
            rg.transform_traces(self.x, self.z, self.rotation, self.scale)
        ]


class SystemResultGraphic(SingleGraphicObject):

    def __init__(
            self, system_result: SystemResult, kind='normal',
            mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'mesh',
            decimals: int | None = None, sig_digits: int | None = None,
            base_scale=None, **kwargs
    ):
        if not isinstance(system_result, SystemResult):
            raise TypeError(
                '"system_result" has to be an instance of SystemResult'
            )
        if kind not in ('normal', 'shear', 'moment', 'u', 'w', 'phi',
                        'bending_line'):
            raise ValueError(
                '"kind" must be one of: '
                '"normal", "shear", "moment", "u", "w", "phi", "bending_line"'
            )
        if mesh_type not in {'bars', 'user_mesh', 'mesh'}:
            raise ValueError(
                '"mesh_type" must be one of ["bars", "user_mesh", "mesh"]'
            )
        super().__init__(
            system_result.bars[0].bar.node_i.x,
            system_result.bars[0].bar.node_i.z,
            **kwargs
        )
        self.system_result = system_result
        self.kind = kind
        self.base_scale = base_scale
        if kind == 'bending_line':
            self._bar_result_graphic = []

            from sstatics.core.postprocessing.bending_line import (
                BendingLine)
            dgl_list = self.system_result.bars
            gb = BendingLine(dgl_list)

            for x, z in gb.deformed_lines():

                points = [(float(xi), float(zi)) for xi, zi in zip(x, z)]

                self._bar_result_graphic.append(
                    LineGraphic.from_points(
                        points,
                        scatter_options=self.scatter_kwargs | {
                            'line': dict(width=4),
                            'line_color': 'red'
                        }
                    )
                )
        else:
            self._bar_result_graphic = [
                BarResultGraphic(
                    bar_result, self.select_result[i], decimals, sig_digits,
                    self._base_scale, self.max_value, rotation=self.rotation,
                    scale=self.scale, scatter_options=self.scatter_kwargs,
                    annotation_options=self.annotation_kwargs
                ) for i, bar_result in enumerate(self.system_result.bars)
            ]

    @cached_property
    def _base_scale(self):
        if self.base_scale:
            return self.base_scale
        return 0.08 * max(self.system_result.system.max_dimensions)

    @cached_property
    def select_result(self):
        forces = self.system_result.forces_disc
        deforms = self.system_result.deforms_disc
        result_lists = {
            'normal': [f[:, 0] for f in forces],
            'shear': [f[:, 1] for f in forces],
            'moment': [f[:, 2] for f in forces],
            'u': [d[:, 0] for d in deforms],
            'w': [d[:, 1] for d in deforms],
            'phi': [d[:, 2] for d in deforms],
        }
        return result_lists[self.kind]

    @cached_property
    def max_value(self):
        max_val = max(np.abs(r).max() for r in self.select_result)
        return 1e-6 if np.isclose(max_val, 0) else max_val

    @property
    def annotations(self):
        annotations = []
        for brg in self._bar_result_graphic:
            annotations.extend(brg.annotations)
        return tuple(annotations)

    @property
    def traces(self):
        traces = []
        for brg in self._bar_result_graphic:
            traces.extend(
                brg.transform_traces(self.x, self.z, self.rotation, self.scale)
            )
        return traces


class BarResultGraphic(SingleGraphicObject):

    def __init__(
            self, bar_result: BarResult | RigidBodyDisplacement,
            results: np.ndarray, decimals: int | None = None, sig_digits:
            int | None = None, base_scale=None, max_value=None, **kwargs
    ):
        if not isinstance(results, np.ndarray) or results.ndim != 1:
            raise TypeError('"results" must be a one-dimensional NumPy array')
        if decimals is not None and sig_digits is not None:
            raise ValueError(
                'Specify only one of "decimals" or "sig_digits", not both.'
            )
        if sig_digits is not None and sig_digits <= 0:
            raise ValueError('"sig_digits" has to be greater than zero.')
        super().__init__(
            bar_result.bar.node_i.x,
            bar_result.bar.node_i.z,
            **kwargs
        )
        self.bar_result = bar_result
        self.origin = (bar_result.bar.node_i.x, bar_result.bar.node_i.z)
        self.inclination = bar_result.bar.inclination
        self.length = bar_result.x
        self.results = results
        self.decimals = decimals
        self.sig_digits = sig_digits
        self.base_scale = base_scale
        self.max_value = max_value

    @cached_property
    def _bound_results(self) -> Tuple[float, float]:
        return self.results[0], self.results[-1]

    def _round_value(self, value):
        if self.decimals is not None:
            return np.round(value, self.decimals)
        if self.sig_digits is not None:
            return f"{value:.{self.sig_digits}g}"
        return np.round(value, 2)

    @cached_property
    def _annotation_pos(self) -> List[Tuple[float, float]]:
        d = 0.15 * self._max_value
        return [
            (self.length[0], (self.results[0] - d) * self._base_scale),
            (self.length[-1], (self.results[-1] - d) * self._base_scale)
        ]

    @cached_property
    def _annotation_text(self) -> Tuple[Any | str, ...]:
        return tuple(
            self._round_value(r) if not np.isclose(r, 0, atol=1e-8) else ""
            for r in self._bound_results
        )

    @cached_property
    def _max_value(self):
        if self.max_value:
            return self.max_value
        max_val = max(abs(self.results))
        return 1e-6 if np.isclose(max_val, 0) else max_val

    @cached_property
    def _max_dim(self):
        bar = self.bar_result.bar
        return max(
            abs(bar.node_i.x - bar.node_j.x),
            abs(bar.node_i.z - bar.node_j.z)
        )

    @cached_property
    def _base_scale(self):
        return self.base_scale / self._max_value if self.base_scale \
            else 0.08 * self._max_dim / self._max_value  # TODO: + 0.02

    @property
    def _annotations(self):
        x, z = np.array(list(zip(*self._annotation_pos)))
        x, z = transform(
            0, 0, x, z, rotation=self.inclination,
            scale=self.scale, translation=self.origin
        )
        return (
            (x[0], z[0], self._annotation_text[0]),
            (x[1], z[1], self._annotation_text[1])
        )

    @property
    def traces(self):
        points = [
            (0, 0),
            *((x, z) for x, z in zip(
                self.length, self.results * self._base_scale
            )),
            (self.length[-1], 0)
        ]
        return (
            LineGraphic.from_points(
                points, scatter_options=self.scatter_kwargs
            ).transform_traces(0, 0, self.inclination, self.scale, self.origin)
        )
