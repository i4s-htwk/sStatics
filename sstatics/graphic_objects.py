
import abc

import numpy as np
import plotly.graph_objs as go


def rotate(ox, oz, x, z, rotation=0):
    x_rot = ox + np.cos(rotation) * (x - ox) - np.sin(rotation) * (z - oz)
    z_rot = oz + np.sin(rotation) * (x - ox) + np.cos(rotation) * (z - oz)
    return x_rot, z_rot


class Figure(go.Figure):

    def __init__(self, **kwargs):
        layout = go.Layout(
            template='simple_white', yaxis_autorange='reversed',
            yaxis_scaleanchor='x', yaxis_scaleratio=1
        )
        kwargs['layout'] = kwargs.get('layout', layout)
        super().__init__(**kwargs)


class GraphicObject(abc.ABC):

    # TODO: find a better solution to pass down options to customize an object
    # TODO: each instance should be customizable
    # TODO: default values should be the intended appearance
    scatter_options = {
        'mode': 'lines',
        'line_color': 'black',
        'showlegend': False,
        'hoverinfo': 'skip',
    }

    def __init__(self, x, z, rotation=0, scale=1, **scatter_kwargs):
        if scale <= 0:
            raise ValueError('"scale" has to be a number greater than zero.')
        self.x = x
        self.z = z
        self.rotation = rotation
        self.scale = scale
        self.scatter_kwargs = self.scatter_options | scatter_kwargs

    @property
    def annotations(self):
        return ()

    @property
    @abc.abstractmethod
    def traces(self):
        pass

    def rotate_traces(self, ox, oz, rotation=0):
        traces = []
        for trace in self.traces:
            x, z = rotate(
                ox, oz, np.array(trace.x), np.array(trace.y), rotation
            )
            traces.append(trace.update(x=x, y=z))
        return tuple(traces)

    def show(self, *args, **kwargs):
        fig = Figure(data=self.traces)
        for annotation in self.annotations:
            fig.add_annotation(annotation)
        fig.show(*args, **kwargs)


class Polygon(GraphicObject):

    def __init__(self, x, z, vertices, **kwargs):
        if len(vertices) < 3:
            raise ValueError(
                'a polygon needs at least three points'
            )
        if not all(
                isinstance(v, (tuple, list)) and len(v) == 2 for v in vertices
        ):
            raise TypeError(
                '"vertices" must be a list of (x, y) tuples or [x, y] lists'
            )
        super().__init__(x, z, **kwargs)
        self.vertices = np.array(vertices)

    @property
    def traces(self):
        center = np.mean(self.vertices, axis=0)
        self.vertices = self.vertices - center + np.array([self.x, self.z])
        x, z = self.vertices[:, 0], self.vertices[:, 1]
        x, z = np.append(x, x[0]), np.append(z, z[0])
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class Rectangle(GraphicObject):

    def __init__(self, x, z, a, b=None, **kwargs):
        if b is None:
            b = a
        if a <= 0 or b <= 0:
            raise ValueError(
                '"a" and "b" have to be a numbers greater than zero.'
            )
        super().__init__(x, z, **kwargs)
        self.a = a * self.scale
        self.b = b * self.scale

    @property
    def traces(self):
        x_offset, z_offset = self.a / 2, self.b / 2
        x = np.array([
            self.x - x_offset, self.x + x_offset,
            self.x + x_offset, self.x - x_offset,
            self.x - x_offset
        ])
        z = np.array([
            self.z - z_offset, self.z - z_offset,
            self.z + z_offset, self.z + z_offset,
            self.z - z_offset
        ])
        x, z = rotate(self.x, self.z, x, z, self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class IsoscelesTriangle(GraphicObject):

    def __init__(self, x, z, angle=np.pi / 4, **kwargs):
        if not 0 < angle < np.pi:
            raise ValueError(
                '"angle" has to be a number from the interval (0, pi).'
            )
        self.angle = angle
        super().__init__(x, z, **kwargs)

    @property
    def traces(self):
        x_offset = np.tan(self.angle / 2) * self.scale
        x = np.array([
            self.x - x_offset, self.x, self.x + x_offset, self.x - x_offset
        ])
        z = np.array([
            self.z + self.scale, self.z, self.z + self.scale,
            self.z + self.scale
        ])
        x, z = rotate(self.x, self.z, x, z, self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


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
        self.scale = self.scale / (1 + self.tail_length)

    @property
    def traces(self):
        arrow = IsoscelesTriangle(
            self.x, self.z, scale=self.scale, rotation=np.pi,
            **self.scatter_kwargs
        )
        arrow_traces = arrow.rotate_traces(
            self.x, self.z, rotation=self.rotation
        )
        x = np.array([self.x, self.x])
        z = np.array([
            self.z - self.scale, self.z - (1 + self.tail_length) * self.scale
        ])
        x, z = rotate(self.x, self.z, x, z, self.rotation)
        tail_traces = go.Scatter(x=x, y=z, **self.scatter_kwargs)
        return *arrow_traces, tail_traces


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
                textangle=np.rad2deg(self.rotation)
            ))
        if self.z_text is not None:
            x, z = rotate(
                self.x, self.z,
                self.x - self.scale / 15, (2 * self.z + self.scale) / 2,
                self.rotation
            )
            annotations.append(go.layout.Annotation(
                x=x, y=z, text=self.z_text, showarrow=False, font_size=40,
                textangle=np.rad2deg(self.rotation)
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


class Ellipse(GraphicObject):

    def __init__(
        self, x, z, a, b=None, angle_range=(0, 2 * np.pi), n_points=100,
        **kwargs
    ):
        if b is None:
            b = a
        if a <= 0 or b <= 0:
            raise ValueError(
                '"a" and "b" have to be a numbers greater than zero.'
            )
        if n_points <= 3:
            raise ValueError(
                '"n_points" has to be an integer greater or equal to 4.'
            )
        for angle in angle_range:
            if not 0 <= angle <= 2 * np.pi:
                raise ValueError(
                    'Both angles in "angle_range" have to be in the interval '
                    '[0, 2 * pi].'
                )
        if angle_range[0] >= angle_range[1]:
            raise ValueError(
                'The lower bound of "angle_range" has to be less than the '
                'upper bound.'
            )
        super().__init__(x, z, **kwargs)
        self.a = a
        self.b = b
        self.angle_range = angle_range
        self.n_points = n_points

    @property
    def traces(self):
        angles = np.linspace(
            self.angle_range[0], self.angle_range[1], self.n_points
        )
        x = self.x + self.a * np.cos(angles) * self.scale
        z = self.z + self.b * np.sin(angles) * self.scale
        x, z = rotate(self.x, self.z, x, z, rotation=self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class Hatching(Rectangle):

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


class Hinge(GraphicObject):

    def __init__(self, x, z, width, **kwargs):
        if width <= 0:
            raise ValueError('"width" has to be a number greater than zero.')
        super().__init__(x, z, **kwargs)
        self.width = width * self.scale


class ShearForceHinge(Hinge):

    def __init__(self, x, z, width=0.4, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        x, z = rotate(
            self.x, self.z,
            np.array([self.x - self.width / 2, self.x - self.width / 2]),
            np.array([self.z - self.scale / 2, self.z + self.scale / 2]),
            rotation=self.rotation
        )
        left_line = go.Scatter(x=x, y=z, **self.scatter_kwargs)
        x, z = rotate(
            self.x, self.z,
            np.array([self.x + self.width / 2, self.x + self.width / 2]),
            np.array([self.z - self.scale / 2, self.z + self.scale / 2]),
            rotation=self.rotation
        )
        right_line = go.Scatter(x=x, y=z, **self.scatter_kwargs)
        return left_line, right_line


class NormalForceHinge(Hinge):

    def __init__(self, x, z, width=0.8, **kwargs):
        super().__init__(x, z, width=width, **kwargs)

    @property
    def traces(self):
        x = np.array([
            self.x + self.scale, self.x, self.x, self.x + self.scale
        ])
        z = np.array([
            self.z - self.width / 2, self.z - self.width / 2,
            self.z + self.width / 2, self.z + self.width / 2
        ])
        x, z = rotate(self.x, self.z, x, z, rotation=self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class MomentHinge(Hinge, Ellipse):

    def __init__(self, x, z, width=0.8, **kwargs):
        super().__init__(x, z, a=width / 2, width=width, **kwargs)


class Support(GraphicObject):

    def __init__(self, x, z, width, **kwargs):
        if width <= 0:
            raise ValueError('"width" has to be a number greater than zero.')
        super().__init__(x, z, **kwargs)
        self.width = width * self.scale


class FreeSupport(Support, IsoscelesTriangle):

    def __init__(self, x, z, width=4/3, **kwargs):
        super().__init__(x, z, width=width, **kwargs)
        self.angle = 2 * np.arctan(self.width / (2 * self.scale))

    @property
    def traces(self):
        x, z = rotate(
            self.x, self.z,
            np.array([self.x - self.width / 2, self.x + self.width / 2]),
            np.array([self.z + 4/3 * self.scale, self.z + 4/3 * self.scale]),
            rotation=self.rotation
        )
        line = go.Scatter(x=x, y=z, **self.scatter_kwargs)
        return line, *super().traces


class FixedSupport(Support, IsoscelesTriangle):

    def __init__(self, x, z, width=4/3, **kwargs):
        super().__init__(x, z, width=width, **kwargs)
        self.angle = 2 * np.arctan(self.width / (2 * self.scale))

    @property
    def traces(self):
        hatching = Hatching(
            self.x, self.z + self.scale * 7/6,
            2 * self.width / (2 * self.scale), 1/3, scale=self.scale
        )
        hatching_traces = hatching.rotate_traces(self.x, self.z, self.rotation)
        return *hatching_traces, *super().traces
