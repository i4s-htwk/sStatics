
import numpy as np
import plotly.graph_objs as go
from sstatics.graphic_objects import (
    rotate, GraphicObject, Line, Ellipse
)


class Hinge(GraphicObject):

    def __init__(self, x, z, width, **kwargs):
        if width <= 0:
            raise ValueError('"width" has to be a number greater than zero.')
        super().__init__(x, z, **kwargs)
        self.width = width * self.scale


class ShearForceHinge(Hinge):

    def __init__(self, x, z, width=1/6, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        left_line = Line(
            self.x - self.width / 2, self.z, 2/3, np.pi/2, scale=self.scale
        )
        left_line_traces = left_line.rotate_traces(
            self.x, self.z, self.rotation
        )
        right_line = Line(
            self.x + self.width / 2, self.z, 2/3, np.pi/2, scale=self.scale
        )
        right_line_traces = right_line.rotate_traces(
            self.x, self.z, self.rotation
        )
        return *left_line_traces, *right_line_traces


class NormalForceHinge(Hinge):

    def __init__(self, x, z, width=2/5, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        x = np.array([
            self.x + self.scale * 2/3, self.x,
            self.x, self.x + self.scale * 2/3
        ])
        z = np.array([
            self.z - self.width / 2, self.z - self.width / 2,
            self.z + self.width / 2, self.z + self.width / 2
        ])
        x, z = rotate(self.x, self.z, x, z, rotation=self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class MomentHinge(Hinge, Ellipse):

    def __init__(self, x, z, width=1/3, **kwargs):
        super().__init__(x, z, a=width / 2, width=width, **kwargs)
