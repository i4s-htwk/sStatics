
from functools import cached_property
from typing import Literal

from sstatics.core.preprocessing.system import System

from sstatics.graphic_objects.utils import SingleGraphicObject
from sstatics.graphic_objects.node import NodeGraphic
from sstatics.graphic_objects.bar import BarGraphic


class SystemGraphic(SingleGraphicObject):

    def __init__(
            self, system: System,
            mesh_type: Literal['bars', 'user_mesh', 'mesh'] = 'bars',
            base_scale=None, **kwargs
    ):
        if not isinstance(system, System):
            raise TypeError('"system" has to be an instance of System')
        super().__init__(
            system.bars[0].node_i.x, system.bars[0].node_i.z, **kwargs
        )
        if mesh_type not in {'bars', 'user_mesh', 'mesh'}:
            raise ValueError(
                '"mesh_type" must be one of ["bars", "user_mesh", "mesh"]'
            )
        self.bars = getattr(system, mesh_type)
        self.nodes = system.nodes(mesh_type=mesh_type)
        self._max_dim = max(system.max_dimensions)
        self.base_scale = base_scale
        self._node_graphic = [
            NodeGraphic(
                node, i+1, self._base_scale,
                scatter_options=self.scatter_kwargs,
                annotation_options=self.annotation_kwargs
            ) for i, node in enumerate(self.nodes)
        ]
        self._bar_graphic = [
            BarGraphic(
                bar, i+1, self._base_scale, self._max_dim,
                scatter_options=self.scatter_kwargs,
                annotation_options=self.annotation_kwargs
            ) for i, bar in enumerate(self.bars)
        ]

    @cached_property
    def _base_scale(self):
        return self.base_scale if self.base_scale \
            else 0.08 * self._max_dim  # TODO: + 0.02

    @property
    def annotations(self):
        annotations = []
        for ng in self._node_graphic:
            annotations.extend(ng.annotations)
        for bg in self._bar_graphic:
            annotations.extend(bg.annotations)
        return tuple(annotations)

    @property
    def traces(self):
        traces = []

        for bg in self._bar_graphic:
            traces.extend(
                bg.transform_traces(self.x, self.z, self.rotation, self.scale)
            )
        for ng in self._node_graphic:
            traces.extend(
                ng.transform_traces(self.x, self.z, self.rotation, self.scale)
            )

        return traces
