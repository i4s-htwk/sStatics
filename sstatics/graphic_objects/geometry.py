
import numpy as np
import plotly.graph_objs as go
from sstatics.graphic_objects import rotate, GraphicObject


class Line(GraphicObject):

    def __init__(self, x, z, length=4/3, angle=0, **kwargs):
        if length <= 0:
            raise ValueError('"length" has to be a number greater than zero.')
        if not 0 <= angle <= np.pi:
            raise ValueError(
                '"angle" has to be a number from the interval (0, pi).'
            )
        super().__init__(x, z, **kwargs)
        self.length = length * self.scale
        self.angle = angle

    @property
    def traces(self):
        x, z = rotate(
            self.x, self.z,
            np.array([self.x + np.cos(self.angle) * self.length / 2,
                      self.x - np.cos(self.angle) * self.length / 2]),
            np.array([self.z + np.sin(self.angle) * self.length / 2,
                      self.z - np.sin(self.angle) * self.length / 2]),
            rotation=self.rotation
        )
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
