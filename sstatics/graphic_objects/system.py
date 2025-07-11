
from sstatics.core.preprocessing.system import System

from sstatics.graphic_objects.utils import SingleGraphicObject
from sstatics.graphic_objects.node import NodeGraphic
from sstatics.graphic_objects.bar import BarGraphic


class SystemGraphic(SingleGraphicObject):

    def __init__(self, system: System, **kwargs):
        if not isinstance(system, System):
            raise TypeError('"system" has to be an instance of System')
        super().__init__(
            system.bars[0].node_i.x, system.bars[0].node_i.z, **kwargs
        )
        self.bars = system.bars
        self.nodes = system.nodes()
        self.node_graphic = [
            NodeGraphic(
                node, i+1, scale=self.base_scale, **self.scatter_kwargs
            ) for i, node in enumerate(self.nodes)
        ]

    @property
    def max_length(self):
        return max(bar.length for bar in self.bars)

    @property
    def base_scale(self):
        return 0.08 * self.max_length  # TODO: + 0.02

    @property
    def annotations(self):
        annotations = []
        for ng in self.node_graphic:
            annotations.extend(ng.annotations)
        return tuple(annotations)

    @property
    def traces(self):
        traces = []

        for bar in self.bars:
            bar_traces = BarGraphic(
                bar, self.base_scale, **self.scatter_kwargs
            ).transform_traces(self.x, self.z, self.rotation, self.scale)
            traces.extend(bar_traces)

        for ng in self.node_graphic:
            traces.extend(
                ng.transform_traces(self.x, self.z, self.rotation, self.scale)
            )

        return traces
