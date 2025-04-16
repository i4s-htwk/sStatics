# arrow, coordinatesystem, hatching

import numpy as np
import plotly.graph_objs as go
from sstatics.graphic_objects import (
    rotate, GraphicObject, Line, IsoscelesTriangle, Rectangle
)


class Arrow(GraphicObject):

    scatter_options = IsoscelesTriangle.scatter_options | {
        'fill': 'toself',
        'fillcolor': 'black',
    }

    def __init__(self, x, z, tail_length=2, **kwargs):
        if tail_length < 0:
            raise ValueError(
                '"tail_length" has to be greater than or equal to zero.'
            )
        super().__init__(x, z, **kwargs)
        self.tail_length = tail_length

    @property
    def traces(self):
        arrow = IsoscelesTriangle(
            self.x, self.z, rotation=np.pi,
            scale=self.scale / (1 + self.tail_length), **self.scatter_kwargs
        )
        arrow_traces = arrow.rotate_traces(self.x, self.z, self.rotation)
        z_offset = np.cos(np.pi / 8) * self.scale / (1 + self.tail_length)
        tail = Line(
            self.x, self.z - (self.scale + z_offset) / 2,
            self.scale - z_offset, rotation=np.pi / 2, **self.scatter_kwargs
        )
        tail_traces = tail.rotate_traces(self.x, self.z, self.rotation)
        return *arrow_traces, *tail_traces


class CoordinateSystem(GraphicObject):

    def __init__(self, x, z, x_text=None, z_text=None, **kwargs):
        super().__init__(x, z, **kwargs)
        self.x_text = x_text
        self.z_text = z_text

    @property
    def annotations(self):
        annotations = []
        if self.x_text is not None:
            x, z = rotate(
                self.x, self.z,
                (2 * self.x + self.scale) / 2, self.z - self.scale / 15,
                self.rotation
            )
            annotations.append(go.layout.Annotation(
                x=x, y=z, text=self.x_text, showarrow=False, font_size=40,
                textangle=None
            ))
        if self.z_text is not None:
            x, z = rotate(
                self.x, self.z,
                self.x - self.scale / 15, (2 * self.z + self.scale) / 2,
                self.rotation
            )
            annotations.append(go.layout.Annotation(
                x=x, y=z, text=self.z_text, showarrow=False, font_size=40,
                textangle=None
            ))
        return tuple(annotations)

    @property
    def traces(self):
        x_axis = Arrow(
            self.x + self.scale, self.z, tail_length=6, scale=self.scale,
            rotation=-np.pi / 2
        )
        z_axis = Arrow(
            self.x, self.z + self.scale, tail_length=6, scale=self.scale
        )
        x_axis_traces = x_axis.rotate_traces(self.x, self.z, self.rotation)
        z_axis_traces = z_axis.rotate_traces(self.x, self.z, self.rotation)
        return x_axis_traces + z_axis_traces


class Hatching(Rectangle):

    def __init__(
            self, x, z, a, b=None, angle=np.pi / 4, spacing=0.17,
            rectangle=False, **kwargs
    ):
        if not 0 < angle < np.pi:
            raise ValueError(
                '"angle" has to be a number from the interval (0, pi).'
            )
        super().__init__(x, z, a, b, **kwargs)
        self.angle = angle
        self.spacing = spacing * self.scale
        self.rectangle = rectangle

    @property
    def traces(self):
        x_offset, z_offset = self.a / 2, self.b / 2
        tan_angle = np.tan(self.angle)
        rect = super().traces[0] if self.rectangle else go.Scatter(x=[], y=[])
        x_list, z_list = [], []
        n = 1
        max_dim = max(self.a, self.b)
        while (d := np.sqrt((n * self.spacing) ** 2 / 2)) <= max_dim:
            if d * (1 + tan_angle) > self.b:
                x = -x_offset + d - (self.b - d) / tan_angle
                if x >= x_offset:
                    break
                x_list.append(x + self.x)
                z_list.append(z_offset + self.z)
            else:
                x_list.append(-x_offset + self.x)
                z_list.append(-z_offset + d * (1 + tan_angle) + self.z)

            if d * (1 + 1 / tan_angle) < self.a:
                x_list.append(-x_offset + d * (1 + 1 / tan_angle) + self.x)
                z_list.append(-z_offset + self.z)
            else:
                z = -z_offset + d - (self.a - d) * tan_angle
                x_list.append(x_offset + self.x)
                z_list.append(z + self.z)
            n += 1
        x, z = rotate(
            self.x, self.z,
            np.array(x_list), np.array(z_list),
            rotation=self.rotation
        )
        hatching = [
            go.Scatter(x=[x[i], x[i + 1]], y=[z[i], z[i + 1]],
                       **self.scatter_kwargs)
            for i in range(0, len(x_list), 2)
        ]
        return rect, *hatching
