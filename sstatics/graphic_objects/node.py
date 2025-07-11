
import numpy as np
import plotly.graph_objs as go

from sstatics.core.preprocessing.node import Node

from sstatics.graphic_objects.utils import SingleGraphicObject
from sstatics.graphic_objects.supports import (
    FreeNode, RollerSupport, PinnedSupport, FixedSupportUW, FixedSupportUPhi,
    FixedSupportWPhi, ChampedSupport
)


class NodeGraphic(SingleGraphicObject):
    def __init__(self, node: Node, node_number=None, **kwargs):
        if not isinstance(node, Node):
            raise TypeError('"node" has to be an instance of Node')
        super().__init__(node.x, node.z, **kwargs)
        self.node = node
        self.number = node_number

    @property
    def select_support(self):
        u, w, phi = self.node.u, self.node.w, self.node.phi
        x, z = self.x, self.z

        support_classes = {
            ('free', 'free', 'free'): FreeNode,
            ('fixed', 'free', 'free'): RollerSupport,
            ('free', 'fixed', 'free'): RollerSupport,
            ('free', 'free', 'fixed'): PinnedSupport,
            ('fixed', 'fixed', 'free'): FixedSupportUW,
            ('fixed', 'free', 'fixed'): FixedSupportUPhi,
            ('free', 'fixed', 'fixed'): FixedSupportWPhi,
            ('fixed', 'fixed', 'fixed'): ChampedSupport,
        }

        support = support_classes.get((u, w, phi), FreeNode)
        if (support is RollerSupport and
                u == 'fixed' and w == 'free' and phi == 'free'):
            return support(x, z, rotation=-np.pi/2, **self.scatter_kwargs)
        return support(x, z, **self.scatter_kwargs)

    @property
    def annotations(self):
        if self.number is not None:
            d = 0.25 * self.scale
            x, z = self.x, self.z - d
            return (go.layout.Annotation(
                x=x, y=z, text=self.number, showarrow=False,
                font=dict(size=20, family='Times New Roman'), textangle=None
            ),)
        return ()

    @property
    def traces(self):
        traces = []

        traces.extend(
            self.select_support.transform_traces(
                self.x, self.z, self.node.rotation, self.scale
            )
        )

        # TODO: displacment

        # TODO: load

        return traces
