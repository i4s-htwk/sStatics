
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
        x_offset = np.sin(self.angle / 2) * self.scale
        x = np.array([
            self.x - x_offset, self.x, self.x + x_offset, self.x - x_offset
        ])
        z = np.array([
            self.z - self.scale, self.z, self.z - self.scale,
            self.z - self.scale
        ])
        x, z = rotate(self.x, self.z, x, z, self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class Arrow(IsoscelesTriangle):

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
        x = np.array([self.x, self.x])
        z = np.array([
            self.z - self.scale, self.z - (1 + self.tail_length) * self.scale
        ])
        x, z = rotate(self.x, self.z, x, z, self.rotation)
        return super().traces + (go.Scatter(x=x, y=z, **self.scatter_kwargs),)


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
                self.x, self.z, (2 * self.x + self.scale) / 2, self.z,
                self.rotation
            )
            annotations.append(go.layout.Annotation(
                x=x, y=z, text=self.x_text, showarrow=False, font_size=12,
                textangle=np.rad2deg(self.rotation)
            ))
        # TODO: add z_text
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
        super().__init__(x, z, **kwargs)
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


class Hinge(GraphicObject):

    def __init__(self, x, z, width, **kwargs):
        super().__init__(x, z, **kwargs)
        if width <= 0:
            raise ValueError('"width" has to be a number greater than zero.')
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
            self.x - self.width / 2, self.x - self.width / 2,
            self.x + self.width / 2, self.x + self.width / 2
        ])
        z = np.array([
            self.z + self.scale, self.z, self.z, self.z + self.scale
        ])
        x, z = rotate(self.x, self.z, x, z, rotation=self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class MomentHinge(Hinge, Ellipse):

    def __init__(self, x, z, width=0.8, **kwargs):
        super().__init__(x, z, a=width / 2, width=width, **kwargs)
