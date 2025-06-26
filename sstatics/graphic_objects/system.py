
from sstatics.core.system import System

from sstatics.graphic_objects.utils import SingleGraphicObject
from sstatics.graphic_objects.node import GraphicNode
from sstatics.graphic_objects.bar import GraphicBar


class GraphicSystem(SingleGraphicObject):

    def __init__(self, system: System, **kwargs):
        if not isinstance(system, System):
            raise TypeError('"system" has to be an instance of System')
        super().__init__(
            system.bars[0].node_i.x, system.bars[0].node_i.z, **kwargs
        )
        self.bars = system.bars
        self.nodes = system.nodes()

    @property
    def max_length(self):
        return max(bar.length for bar in self.bars)

    @property
    def base_scale(self):
        return 0.08 * self.max_length  # TODO: + 0.02

    @property
    def traces(self):
        traces = []

        for bar in self.bars:
            bar_traces = GraphicBar(
                bar, self.base_scale, **self.scatter_kwargs
            ).transform_traces(self.x, self.z, self.rotation, self.scale)
            traces.extend(bar_traces)

        for node in self.nodes:
            node_traces = GraphicNode(
                node, scale=self.base_scale, **self.scatter_kwargs
            ).transform_traces(self.x, self.z, self.rotation, self.scale)
            traces.extend(node_traces)

        return traces
