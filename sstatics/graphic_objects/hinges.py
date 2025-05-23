
import numpy as np
from abc import ABC
import plotly.graph_objs as go

from sstatics.graphic_objects import (
    transform, SingleGraphicObject, Line, Ellipse
)


class Hinge(GraphicObject):
class Hinge(SingleGraphicObject, ABC):

    def __init__(self, x, z, width, **kwargs):
        if width <= 0:
            raise ValueError('"width" has to be a number greater than zero.')
        super().__init__(x, z, **kwargs)
        self.width = width * self.scale


class NormalForceHinge(Hinge):

    def __init__(self, x, z, width=11/30, **kwargs):
        super().__init__(x, z, width, **kwargs)

    @property
    def traces(self):
        x_off, z_off = 11 / 20, self.width / 2
        x = np.array(
            [self.x + 3 / 4 * x_off, -1 / 4 * x_off + self.x,
             -1 / 4 * x_off + self.x, self.x + 3 / 4 * x_off]
        )
        z = np.array([
            self.z - z_off, self.z - z_off,
            self.z + z_off, self.z + z_off
        ])
        x, z = transform(self.x, self.z, x, z, self.rotation, self.scale)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class ShearForceHinge(Hinge):

    def __init__(self, x, z, width=11/80, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        length, x_offset = 11 / 20, self.width / 2
        left_line = Line.from_center(
            self.x - x_offset, self.z, length,
            rotation=np.pi / 2, **self.scatter_kwargs
        )
        left_line_traces = left_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        right_line = Line.from_center(
            self.x + x_offset, self.z, length,
            rotation=np.pi / 2, **self.scatter_kwargs
        )
        right_line_traces = right_line.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *left_line_traces, *right_line_traces


class MomentHinge(Hinge, Ellipse):

    def __init__(self, x, z, width=11/40, **kwargs):
        super().__init__(x, z, a=width / 2, width=width, **kwargs)
