
import numpy as np
import plotly.graph_objs as go
from functools import cached_property

from sstatics.graphic_objects import (
    transform, MultiGraphicObject, SingleGraphicObject
)


class Point(SingleGraphicObject):

    scatter_options = SingleGraphicObject.scatter_options | {
        'mode': 'markers',
        'marker': dict(size=2),
    }

    def __init__(self, x, z, **kwargs):
        super().__init__(x, z, **kwargs)

    @property
    def traces(self):
        x, z = transform(
            self.x, self.z, np.array([self.x]), np.array([self.z]),
            self.rotation, self.scale
        )
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


class Line(MultiGraphicObject):

    def __init__(self, x, z, points, **kwargs):
        super().__init__(points, **kwargs)
        self.x = x
        self.z = z

    @classmethod
    def from_center(cls, x, z, length=4/3, **kwargs):
        if length <= 0:
            raise ValueError('"length" must be greater than zero.')
        points = [(x + length / 2, z), (x - length / 2, z)]
        return cls(x, z, points, **kwargs)

    @classmethod
    def from_points(cls, points, **kwargs):
        if not (
            isinstance(points, list) and len(points) > 1 and
            all(
                isinstance(p, (list, tuple)) and len(p) == 2 for p in points
            )
        ):
            raise ValueError(
                '"points" has to be a list of (x, z) tuples or [x, z] lists.'
            )
        x, z = points[0]
        return cls(x, z, points, **kwargs)

    @property
    def traces(self):
        x, z = np.array(list(zip(*self.points)))
        x, z = transform(self.x, self.z, x, z, self.rotation, self.scale)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),

    # TODO: new names for parameters
    def stretching(self, start=0.0, end=0.0):
        if len(self.points) != 2:
            raise ValueError(
                '"stretching()" only can be used for lines with two points.'
            )
        p1, p2 = np.array(self.points[0]), np.array(self.points[1])
        direction = p2 - p1
        length = np.linalg.norm(direction)
        if length == 0:
            raise ValueError("Cannot stretch a line of zero length.")
        unit = direction / length
        new_p1 = p1 - start * unit
        new_p2 = p2 + end * unit
        self.points = [tuple(new_p1), tuple(new_p2)]
        return self


class Polygon(MultiGraphicObject):

    def __init__(self, points, **kwargs):
        if points[0] != points[-1]:
            points += [points[0]]
        super().__init__(points, **kwargs)

    @property
    def traces(self):
        return Line.from_points(
            self.points, rotation=self.rotation, scale=self.scale,
            **self.scatter_kwargs
        ).traces


class Rectangle(SingleGraphicObject):

    def __init__(self, x, z, a, b=None, **kwargs):
        if b is None:
            b = a
        if a <= 0 or b <= 0:
            raise ValueError(
                '"a" and "b" have to be a numbers greater than zero.'
            )
        super().__init__(x, z, **kwargs)
        self.a = a
        self.b = b

    @cached_property
    def corner_points(self):
        x_off, z_off = self.a / 2, self.b / 2
        return [
            (self.x - x_off, self.z - z_off), (self.x + x_off, self.z - z_off),
            (self.x + x_off, self.z + z_off), (self.x - x_off, self.z + z_off)
        ]

    @property
    def traces(self):
        polygon = Polygon(self.corner_points, **self.scatter_kwargs)
        return polygon.transform_traces(
            self.x, self.z, rotation=self.rotation, scale=self.scale
        )


class IsoscelesTriangle(GraphicObject):

    def __init__(self, x, z, angle=np.pi / 4, width=None, **kwargs):
        super().__init__(x, z, **kwargs)
        if width is None:
            if not 0 < angle < np.pi:
                raise ValueError(
                    '"angle" has to be a number from the interval [0, pi].'
                )
            self.angle = angle
            self.width = 2 * np.sin(self.angle / 2) * self.scale
        else:
            if width <= 0 or width >= 2 * self.scale:
                raise ValueError(
                    '"width" has to be a number from the interval (0, 2).'
                )
            self.width = width
            self.angle = 2 * np.arcsin(self.width / (2 * self.scale))

    @property
    def traces(self):
        x_offset = np.sin(self.angle / 2) * self.scale
        z_offset = np.cos(self.angle / 2) * self.scale
        x = np.array([
            self.x - x_offset, self.x, self.x + x_offset, self.x - x_offset
        ])
        z = np.array([
            self.z + z_offset, self.z, self.z + z_offset,
            self.z + z_offset
        ])
        x, z = rotate(self.x, self.z, x, z, self.rotation)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),


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


class Ellipse(SingleGraphicObject):

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
        x = self.x + self.a * np.cos(angles)
        z = self.z + self.b * np.sin(angles)
        x, z = transform(self.x, self.z, x, z, self.rotation, self.scale)
        return go.Scatter(x=x, y=z, **self.scatter_kwargs),
