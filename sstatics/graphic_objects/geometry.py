
import numpy as np
import plotly.graph_objs as go
from functools import cached_property

from sstatics.core.preprocessing.geometry.objects import Polygon
from sstatics.graphic_objects.utils import (
    transform, MultiGraphicObject, SingleGraphicObject
)


def set_alpha(rgba_string: str, alpha: float) -> str:
    # Extrahiert rgba-Werte und ersetzt den letzten Wert durch das neue alpha
    try:
        rgba = rgba_string.strip().lower()
        if not rgba.startswith("rgba"):
            return rgba  # kein gültiger rgba-Wert
        parts = rgba[5:-1].split(",")
        if len(parts) != 4:
            return rgba
        r, g, b, _ = [p.strip() for p in parts]
        return f'rgba({r}, {g}, {b}, {alpha})'
    except Exception:
        return rgba_string


class PointGraphic(SingleGraphicObject):

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


class LineGraphic(MultiGraphicObject):

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


class PolygonGraphic(MultiGraphicObject):

    scatter_options = MultiGraphicObject.scatter_options | {
        'fill': 'toself',
        'fillcolor': 'rgba(0, 0, 0, 0)',
    }

    def __init__(
            self, polygon: Polygon, show_center_of_mass: bool = False, **kwargs
    ):
        super().__init__(polygon.points, **kwargs)
        self.holes = polygon.holes
        self.x = polygon.center_of_mass_y
        self.z = polygon.center_of_mass_z
        self.show_center_of_mass = show_center_of_mass

    @property
    def traces(self):
        ex, ez = np.array(list(zip(*self.points)))
        x, z = transform(self.x, self.z, ex, ez, self.rotation, self.scale)
        x, z = list(x), list(z)
        for hole in self.holes:
            ix, iz = np.array(list(zip(*hole)))
            ix, iz = transform(
                self.x, self.z, ix, iz, self.rotation, self.scale
            )
            x += [None] + list(ix)
            z += [None] + list(iz)

        fill_color = self.scatter_kwargs['fillcolor']
        point = (
            PointGraphic(
                self.x, self.z, marker=dict(size=10),
                line_color=",".join(fill_color.split(",")[:3]) + ", 1.0)"
            ).traces if self.show_center_of_mass else ()
        )
        return (go.Scatter(x=x, y=z, **self.scatter_kwargs),) + point


class RectangleGraphic(SingleGraphicObject):

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
    def _points(self):
        x_off, z_off = self.a / 2, self.b / 2
        return [
            (self.x - x_off, self.z - z_off), (self.x + x_off, self.z - z_off),
            (self.x + x_off, self.z + z_off), (self.x - x_off, self.z + z_off),
            (self.x - x_off, self.z - z_off)
        ]

    @property
    def traces(self):
        polygon = PolygonGraphic(Polygon(self._points), **self.scatter_kwargs)
        return polygon.transform_traces(
            self.x, self.z, rotation=self.rotation, scale=self.scale
        )


class IsoscelesTriangleGraphic(SingleGraphicObject):

    def __init__(self, x, z, angle, width, **kwargs):
        super().__init__(x, z, **kwargs)
        self.angle = angle
        self.width = width

    @classmethod
    def from_angle(cls, x, z, angle=np.pi / 4, **kwargs):
        if not 0 < angle < np.pi:
            raise ValueError('"angle" must be in the interval (0, π).')
        width = 2 * np.sin(angle / 2)
        return cls(x, z, angle, width, **kwargs)

    @classmethod
    def from_width(cls, x, z, width=11 / 10, **kwargs):
        if not 0 < width < 2:
            raise ValueError('"width" must be in the interval (0, 2).')
        angle = 2 * np.arcsin(width / 2)
        return cls(x, z, angle, width, **kwargs)

    @cached_property
    def _points(self):
        x_off = np.sin(self.angle / 2)
        z_off = np.cos(self.angle / 2)
        return [
            (self.x, self.z),
            (self.x + x_off, self.z + z_off), (self.x - x_off, self.z + z_off),
            (self.x, self.z)
        ]

    @property
    def traces(self):
        return PolygonGraphic(
            Polygon(self._points), rotation=self.rotation, scale=self.scale,
            **self.scatter_kwargs
        ).traces


class EllipseGraphic(SingleGraphicObject):

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


class CircularSector(EllipseGraphic):

    def __init__(
        self, x, z, a, b=None, angle_range=(0, 2 * np.pi), n_points=100,
        **kwargs
    ):
        super().__init__(x, z, a, b, angle_range, n_points, **kwargs)
        self.sector = super().traces[0]

    @property
    def start_points(self):
        return self.sector.x[0], self.sector.y[0]

    @property
    def end_points(self):
        return self.sector.x[-1], self.sector.y[-1]

    @property
    def traces(self):
        start_line = LineGraphic.from_points(
            [(self.x, self.z), self.start_points], **self.scatter_kwargs
        ).traces
        end_line = LineGraphic.from_points(
            [(self.x, self.z), self.end_points], **self.scatter_kwargs
        ).traces
        return self.sector, *start_line, *end_line
