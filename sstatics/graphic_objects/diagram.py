
import numpy as np
import plotly.graph_objs as go

from sstatics.graphic_objects import (
    rotate, transform, SingleGraphicObject, LineGraphic,
    IsoscelesTriangleGraphic, RectangleGraphic
)
from sstatics.graphic_objects.geometry import EllipseGraphic


class Arrow(SingleGraphicObject):

    def __init__(self, x, z, width, **kwargs):
        if width <= 0:
            raise ValueError('"width" has to be greater than zero.')
        super().__init__(x, z, **kwargs)
        self.width = width


class StraightArrow(Arrow):

    def __init__(
            self, x, z, length: float = 3, head_width: float = 0.4,
            offset: float = 0, **kwargs
    ):
        if length <= 0:
            raise ValueError('"length" has to be greater than zero.')
        super().__init__(x, z, head_width, **kwargs)
        self.length = length
        self.offset = offset

    @property
    def traces(self):
        head = IsoscelesTriangleGraphic.from_width(
            self.x, self.z - self.offset, self.width,
            scatter_options=SingleGraphicObject.scatter_options | {
                'fill': 'toself', 'fillcolor': 'black'
            },
            rotation=np.pi
        )
        head_traces = head.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        tail = LineGraphic.from_points(
            [(self.x, self.z - self.offset),
             (self.x, self.z - self.length - self.offset)],
            scatter_options=self.scatter_kwargs
        )
        tail_traces = tail.transform_traces(self.x, self.z, self.rotation)
        return *head_traces, *tail_traces


class CurvedArrow(Arrow):

    def __init__(
            self, x, z, radius: float = 1,
            angle_span: tuple[float, float] = (0, np.pi / 2),
            head_width: float = 0.4, **kwargs
    ):
        super().__init__(x, z, head_width, **kwargs)
        self.radius = radius
        self.angle_span = angle_span

    @property
    def head_position(self):
        end_angle = self.angle_span[1]
        x = self.x + self.radius * 2 * np.cos(end_angle)
        z = self.z + self.radius * 2 * np.sin(end_angle)
        return x, z

    @property
    def traces(self):
        head_angle_offset = (
            np.pi + np.pi / 12 if self.angle_span[0] < self.angle_span[1]
            else - np.pi / 12
        )
        head = IsoscelesTriangleGraphic.from_width(
            *self.head_position, self.width,
            scatter_options=SingleGraphicObject.scatter_options | {
                'fill': 'toself', 'fillcolor': 'black'
            },
            rotation=-self.angle_span[1] + head_angle_offset
        )
        head_traces = head.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )

        tail = EllipseGraphic(
            self.x, self.z, self.radius * 2, angle_range=self.angle_span,
            scatter_options=self.scatter_kwargs
        )
        tail_traces = tail.transform_traces(
            self.x, self.z, self.rotation, self.scale
        )
        return *head_traces, *tail_traces


class ScaledArrow(SingleGraphicObject):

    scatter_options = SingleGraphicObject.scatter_options | {
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
        arrow = IsoscelesTriangleGraphic.from_angle(
            self.x, self.z, scatter_options=self.scatter_kwargs,
            rotation=np.pi, scale=self.scale / (1 + self.tail_length)
        )
        arrow_traces = arrow.transform_traces(self.x, self.z, self.rotation)
        z_offset = np.cos(np.pi / 8) * self.scale / (1 + self.tail_length)
        tail = LineGraphic.from_points(
            [(self.x, self.z - z_offset), (self.x, self.z - self.scale)],
            scatter_options=self.scatter_kwargs
        )
        tail_traces = tail.transform_traces(self.x, self.z, self.rotation)
        return *arrow_traces, *tail_traces


class CoordinateSystem(SingleGraphicObject):

    annotation_options = SingleGraphicObject.annotation_options | {
        'font': {'size': 40, 'family': 'Times New Roman'},
    }

    def __init__(self, x, z, x_text=None, z_text=None, **kwargs):
        super().__init__(x, z, **kwargs)
        self.x_text = x_text
        self.z_text = z_text

    @property
    def _annotations(self):
        annotations = []
        if self.x_text is not None:
            x, z = rotate(
                self.x, self.z,
                (2 * self.x + self.scale) / 2, self.z - self.scale / 15,
                self.rotation
            )
            annotations.append((x, z, self.x_text),)
        if self.z_text is not None:
            x, z = rotate(
                self.x, self.z,
                self.x - self.scale / 15, (2 * self.z + self.scale) / 2,
                self.rotation
            )
            annotations.append((x, z, self.z_text),)
        return tuple(annotations)

    @property
    def traces(self):
        x_axis = ScaledArrow(
            self.x + self.scale, self.z, tail_length=6, rotation=np.pi / 2,
            scale=self.scale, scatter_options=self.scatter_kwargs
        )
        z_axis = ScaledArrow(
            self.x, self.z + self.scale, tail_length=6, scale=self.scale,
            scatter_options=self.scatter_kwargs
        )
        x_axis_traces = x_axis.transform_traces(
            self.x, self.z, self.rotation
        )
        z_axis_traces = z_axis.transform_traces(
            self.x, self.z, self.rotation
        )
        return *x_axis_traces, *z_axis_traces


class Hatching(RectangleGraphic):

    def __init__(
            self, x, z, a, b=None, angle=np.pi / 4, spacing=0.2,
            rectangle=False, **kwargs
    ):
        if not 0 < angle < np.pi:
            raise ValueError(
                '"angle" has to be a number from the interval (0, pi).'
            )
        super().__init__(x, z, a, b, **kwargs)
        self.angle = angle
        self.spacing = spacing
        self.rectangle = rectangle

    @property
    def traces(self):
        x_off, z_off = self.a / 2, self.b / 2
        tan_angle = np.tan(self.angle)
        rect = super().traces[0] if self.rectangle else go.Scatter(x=[], y=[])
        x_list, z_list = [], []
        n = 1 / 2
        max_dim = max(self.a, self.b)
        while (d := np.sqrt((n * self.spacing) ** 2 / 2)) <= max_dim:
            if d * (1 + tan_angle) > self.b:
                x = -x_off + d - (self.b - d) / tan_angle
                if x >= x_off:
                    break
                x_list.append(x + self.x)
                z_list.append(z_off + self.z)
            else:
                x_list.append(-x_off + self.x)
                z_list.append(-z_off + d * (1 + tan_angle) + self.z)

            if d * (1 + 1 / tan_angle) < self.a:
                x_list.append(-x_off + d * (1 + 1 / tan_angle) + self.x)
                z_list.append(-z_off + self.z)
            else:
                z = -z_off + d - (self.a - d) * tan_angle
                x_list.append(x_off + self.x)
                z_list.append(z + self.z)
            n += 1
        x, z = transform(
            self.x, self.z,
            np.array(x_list), np.array(z_list),
            self.rotation, self.scale
        )
        hatching = [
            go.Scatter(x=[x[i], x[i + 1]], y=[z[i], z[i + 1]],
                       **self.scatter_kwargs)
            for i in range(0, len(x_list), 2)
        ]
        return rect, *hatching
