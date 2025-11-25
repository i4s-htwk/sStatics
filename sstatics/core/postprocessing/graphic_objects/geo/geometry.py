
import abc
import math
from functools import cached_property
from types import NoneType
from typing import Any

import numpy as np

from sstatics.core.preprocessing.geometry.objects import Polygon
from .object_geo import ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_POLYGON, DEFAULT_CENTER_OF_MASS, DEFAULT_HATCH
)
from ..utils.uitls import is_clockwise


class PointGeo(ObjectGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            **kwargs
    ):
        super().__init__(origin=origin, **kwargs)

    @cached_property
    def graphic_elements(self):
        ox, oz = self._origin
        return [([ox], [oz], self._point_style)]

    @cached_property
    def text_elements(self):
        return [(*self._origin, self._text, self._text_style)]

    def __repr__(self):
        return (
            f'PointGeo('
            f'origin={self._origin}, '
            f'point_style={self._point_style}, '
            f'text={self._text if self._text else None}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )


class OpenCurveGeo(ObjectGeo):

    def __init__(
            self,
            x: list[float],
            z: list[float],
            origin: tuple[float, float] | None = None,
            **kwargs
    ):
        self._validate_coords(x, z)
        if origin is None:
            origin = (x[0], z[0])
        super().__init__(origin=origin, **kwargs)
        self._x = x
        self._z = z

    @cached_property
    def graphic_elements(self):
        return [(self._x, self._z, self._line_style)]

    @cached_property
    def text_elements(self):
        text = self._text
        if not text:
            return []
        n = len(text)
        n_points = len(self._x)
        if n == 1:
            if n_points == 2:
                (x0, x1) = self._x
                (z0, z1) = self._z
                return [((x0 + x1) / 2, (z0 + z1) / 2, text, self._text_style)]
            return [(*self._origin, text, self._text_style)]
        elif n == n_points:
            return [
                (self._x[i], self._z[i], [t], self._text_style)
                for i, t in enumerate(text)
            ]
        else:
            raise ValueError(
                f'Invalid number of text entries: expected either 1 or'
                f'{n_points}, but got {n}. The number of text labels must '
                f'match the number of curve points (len(x) and len(z)) or be '
                f'exactly one.'
            )

    @classmethod
    def from_center(cls, origin: tuple[float, float], length: float, **kwargs):
        cls._validate_center(origin, length)
        ox, oz = origin
        x = [ox - length / 2, ox + length / 2]
        z = [oz, oz]
        return cls(x, z, origin=origin, **kwargs)

    @classmethod
    def from_slop_intercept(
            cls, slope: float | None, intercept: float,
            boundaries: tuple[float, float, float, float] = (-5, 5, -5, 5),
            **kwargs
    ):
        cls._validate_slop_intercept(slope, intercept, boundaries)
        x_max, x_min, z_max, z_min = boundaries
        if slope is None:
            x, z = [intercept, intercept], [z_min, z_max]
        elif slope == 0:
            x, z = [x_min, x_max], [intercept, intercept]
        else:
            x = [x_min, x_max]
            z = [slope * x_min + intercept, slope * x_max + intercept]
        return cls(x, z, **kwargs)

    def stretch(self, start: float = 0.0, end: float = 0.0):
        self._validate_stretch(start, end)
        (x0, z0), (x1, z1) = (
            (self._x[0], self._z[0]), (self._x[-1], self._z[-1])
        )

        dx, dz = x1 - x0, z1 - z0
        length = math.hypot(dx, dz)
        if length == 0:
            raise ValueError('Cannot stretch a line of zero length.')

        ux, uz = dx / length, dz / length
        self._x = [x0 - start * ux, x1 + end * ux]
        self._z = [z0 - start * uz, z1 + end * uz]
        return self

    @staticmethod
    def _validate_center(origin, length):
        if not isinstance(origin, tuple) or len(origin) != 2:
            raise TypeError(
                f'origin must be a tuple of length 2, got '
                f'{type(origin).__name__}.'
            )

        if not all(isinstance(v, (int, float)) for v in origin):
            raise TypeError(f'origin must contain numbers, got {origin!r}.')

        if not isinstance(length, (int, float)):
            raise TypeError(
                f'length must be a number, got {type(length).__name__}'
            )

        if length <= 0:
            raise ValueError('length must be positive.')

    @staticmethod
    def _validate_coords(x, z):
        if not isinstance(x, list):
            raise TypeError(f'x must be a list, got {type(x).__name__}.')

        if not isinstance(z, list):
            raise TypeError(f'z must be a list, got {type(z).__name__}.')

        if not all(isinstance(v, (int, float)) for v in x):
            raise TypeError('All x values must be a numbers.')

        if not all(isinstance(v, (int, float)) for v in z):
            raise TypeError('All z values must be a numbers.')

        if len(x) != len(z):
            raise ValueError('x and z must have the same length.')

        if len(x) < 2:
            raise ValueError('A line requires at least two points.')

    @staticmethod
    def _validate_slop_intercept(slope, intercept, boundaries):
        if slope is not None and not isinstance(slope, (int, float)):
            raise TypeError(
                f'slope must be a number or None, got {type(slope).__name__}'
            )

        if not isinstance(intercept, (int, float)):
            raise TypeError(
                f'intercept must be a number or None, got '
                f'{type(intercept).__name__}'
            )

        if not isinstance(boundaries, tuple) or len(boundaries) != 4:
            raise TypeError(
                f'boundaries must be a tuple of length 4, got '
                f'{type(boundaries).__name__}.'
            )

        if not all(isinstance(v, (int, float)) for v in boundaries):
            raise TypeError(
                f'boundaries must contain numbers, got {boundaries!r}.'
            )

    def _validate_stretch(self, start, end):
        if not isinstance(start, (int, float)):
            raise TypeError(f'start must be a number, got {start!r}.')

        if not isinstance(end, (int, float)):
            raise TypeError(f'end must be a number, got {end!r}.')

        if len(self._x) != 2 or len(self._z) != 2:
            raise ValueError(
                '"stretch()" only can be used for lines with two points'
            )

    @property
    def x(self):
        return self._x

    @property
    def z(self):
        return self._z

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'x={self._x}, '
            f'z={self._z}, '
            f'line_style={self._line_style}, '
            f'text={self._text if self._text else None}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )


class ClosedCurveGeo(ObjectGeo, abc.ABC):
    DEFAULT_HATCH_STYLE = DEFAULT_HATCH

    def __init__(
            self,
            origin: tuple[float, float],
            show_center: bool = False,
            show_hatch: bool | None = None,
            hatch_style: dict[str, Any] | None = None,
            show_outline: bool = True,
            **kwargs
    ):
        self._validate_curve(
            show_center, show_hatch, hatch_style, show_outline
        )
        super().__init__(origin=origin, **kwargs)
        self._show_center = show_center
        self._hatch_style = self._set_hatch_style(show_hatch, hatch_style)
        self._show_outline = show_outline

        if not self._show_outline:
            self._line_style = self._style_without_outline(self._line_style)

    @cached_property
    def graphic_elements(self):
        elements = []
        elements.extend(self._point_coords)
        if self._show_center:
            elements.append(self._center_geo)
        if self._hatch_style:
            elements.append(self._hatch_geo)
        return elements

    @cached_property
    def text_elements(self):
        return []

    @cached_property
    @abc.abstractmethod
    def _point_coords(self):
        pass

    @cached_property
    def _center_geo(self):
        return PointGeo(
                self._origin, text=self._text, point_style=self._point_style
            )

    @cached_property
    def _hatch_geo(self):
        from .hatch import HatchGeo
        return HatchGeo(self, **self._hatch_style)

    @staticmethod
    def _validate_curve(show_center, show_hatch, hatch_style, show_outline):
        if not isinstance(show_center, bool):
            raise TypeError(
                f'"show_center" must be a boolean, got '
                f'{type(show_center).__name__}.'
            )

        if not isinstance(show_hatch, (bool, NoneType)):
            raise TypeError(
                f'"show_hatch" must be a boolean, got '
                f'{type(show_hatch).__name__}.'
            )

        if not isinstance(hatch_style, (dict, NoneType)):
            raise TypeError(
                f'"hatch_style" must be a dictionary or None, got '
                f'{type(hatch_style).__name__}'
            )

        if not isinstance(show_outline, bool):
            raise TypeError(
                f'"show_outline" must be a boolean, got '
                f'{type(show_outline).__name__}.'
            )

    def _set_hatch_style(self, show_hatch, user_hatch):
        if show_hatch is False:
            return None
        class_hatch = getattr(self, 'CLASS_HATCH_STYLE', {})
        hatch = show_hatch or user_hatch or class_hatch
        if hatch:
            return self._merge_style(
                self.DEFAULT_HATCH_STYLE,
                class_hatch,
                user_hatch or {}
            )
        return None

    @staticmethod
    def _style_without_outline(style: dict[str, Any]) -> dict[str, Any]:
        style = {**style, 'line': {**style.get('line', {}), 'width': 0}}
        if 'fill' not in style and 'fillcolor' in style:
            style['fill'] = 'toself'
        return style

    @property
    def show_center(self):
        return self._show_center

    @property
    def shape_coords(self):
        return [(x, z) for x, z, _ in self._point_coords]

    @property
    def hatch(self):
        return self._hatch_style is not None

    @property
    def hatch_style(self):
        return self._hatch_style

    @property
    def show_outline(self):
        return self._show_outline


class PolygonGeo(ClosedCurveGeo):
    CLASS_STYLES = {
        'line': DEFAULT_POLYGON,
        'point': DEFAULT_CENTER_OF_MASS
    }

    def __init__(
            self,
            polygon: Polygon,
            **kwargs
    ):
        self._validate_polygon(polygon)
        super().__init__(
            origin=(polygon.center_of_mass_y, polygon.center_of_mass_z),
            **kwargs
        )
        self._ex, self._ez = list(zip(*polygon.points))
        self._holes = polygon.holes

    @cached_property
    def _point_coords(self):
        x, z = list(self._ex), list(self._ez)
        if not is_clockwise(x, z):
            x.reverse()
            z.reverse()
        holes = [
            ([None] + ix[::-1], [None] + iz[::-1])
            if is_clockwise(ix, iz) else ([None] + ix, [None] + iz)
            for ix, iz in (map(list, zip(*h)) for h in self._holes)
        ]
        for hx, hz in holes:
            x += hx
            z += hz
        x += [None]
        z += [None]
        return [(x, z, self._line_style)]

    @staticmethod
    def _validate_polygon(polygon):
        if not isinstance(polygon, Polygon):
            raise TypeError(f'polygon must be a Polygon, got {polygon!r}')

    def __repr__(self):
        points_x = [x for x, _, _ in self._point_coords]
        points_z = [z for _, z, _ in self._point_coords]
        return (
            f'{self.__class__.__name__}('
            f'points_x={points_x}, points_z={points_z}, '
            f'line_style={self._line_style}, '
            f'point={self._center_geo}, '
            f'hatch_style={self._hatch_style}, '
            f'show_outline={self._show_outline}, '
            f'Transform={self._transform})'
        )


class EllipseGeo(ClosedCurveGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            width: float,
            height: float | None = None,
            angle_range: tuple[float, float] = (0, 2 * np.pi),
            n_points: int = 100,
            **kwargs
    ):
        if height is None:
            height = width
        self._validate_ellipse(width, height, angle_range, n_points)
        super().__init__(origin=origin, **kwargs)
        self._a = width / 2
        self._b = height / 2
        self._angle_range = angle_range
        self._n_points = n_points

    @cached_property
    def _point_coords(self):
        angles = np.linspace(
            self._angle_range[0], self._angle_range[1], self._n_points
        )
        x0, z0 = self._origin
        x = x0 + self._a * np.cos(angles)
        z = z0 + self._b * np.sin(angles)
        return [(x, z, self._line_style)]

    @staticmethod
    def _validate_ellipse(width, height, angle_range, n_points):
        if not isinstance(width, (int, float)):
            raise TypeError(f'"width" must be a number, got {width!r}')

        if not isinstance(height, (int, float)):
            raise TypeError(f'"height" must be a number, got {height!r}')

        if width <= 0 or height <= 0:
            raise ValueError(
                '"width" and "height" have to be a numbers greater than zero.'
            )

        if not all(isinstance(angle, (int, float)) for angle in angle_range):
            raise ValueError(
                f'Both angles in "angle_range" must be numbers, '
                f'got {angle_range!r}.'
            )

        if not isinstance(n_points, int):
            raise TypeError(f'"n_points" must be an integer, got {n_points!r}')

        if n_points < 4:
            raise ValueError(
                '"n_points" has to be an integer greater or equal to 4.'
            )

    @property
    def width(self):
        return self._a * 2

    @property
    def height(self):
        return self._b * 2

    @property
    def a(self):
        return self._a

    @property
    def b(self):
        return self._b

    @property
    def angle_range(self):
        return self._angle_range

    @property
    def n_points(self):
        return self._n_points

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'a={self._a}, '
            f'b={self._b}, '
            f'angle_range={self._angle_range}, '
            f'n_points={self._n_points}, '
            f'line_style={self._line_style}, '
            f'point={self._center_geo}, '
            f'hatch_style={self._hatch_style}, '
            f'show_outline={self._show_outline}, '
            f'Transform={self._transform})'
        )


class IsoscelesTriangleGeo(ClosedCurveGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            width: float,
            height: float | None = None,
            **kwargs
    ):
        if height is None:
            height = width / 2
        self._validate_triangle(width, height)
        super().__init__(origin=origin, **kwargs)
        self._width = width
        self._height = height

    @cached_property
    def _point_coords(self):
        x0, z0 = self._origin
        x_off = self._width / 2
        x = x0 - x_off, x0 + x_off, x0, x0 - x_off
        z = z0 + self._height, z0 + self._height, z0, z0 + self._height
        return [(x, z, self._line_style)]

    @staticmethod
    def _validate_triangle(width, height):
        if not isinstance(width, (int, float)):
            raise TypeError(f'"width" must be a number, got {width!r}')

        if not isinstance(height, (int, float)):
            raise TypeError(f'"height" must be a number, got {height!r}')

        if width < 0 or height < 0:
            raise ValueError(
                '"width" and "height" have to be a numbers greater than or '
                'equal to zero.'
            )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'width={self._width}, '
            f'height={self._height}, '
            f'line_style={self._line_style}, '
            f'point={self._center_geo}, '
            f'hatch_style={self._hatch_style}, '
            f'show_outline={self._show_outline}, '
            f'Transform={self._transform})'
        )


class RectangleGeo(ClosedCurveGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            width: float,
            height: float | None = None,
            **kwargs
    ):
        if height is None:
            height = width / 2
        self._validate_rectangle(width, height)
        super().__init__(origin=origin, **kwargs)
        self._width = width
        self._height = height

    @cached_property
    def _point_coords(self):
        x0, z0 = self._origin
        x_off, z_off = self._width / 2, self._height / 2
        x = x0 - x_off, x0 + x_off, x0 + x_off, x0 - x_off, x0 - x_off
        z = z0 - z_off, z0 - z_off, z0 + z_off, z0 + z_off, z0 - z_off
        return [(x, z, self._line_style)]

    @staticmethod
    def _validate_rectangle(width, height):
        if not isinstance(width, (int, float)):
            raise TypeError(f'"width" must be a number, got {width!r}')

        if not isinstance(height, (int, float)):
            raise TypeError(f'"height" must be a number, got {height!r}')

        if width <= 0 or height <= 0:
            raise ValueError(
                '"width" and "height" have to be a numbers greater than zero.'
            )

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'width={self._width}, '
            f'height={self._height}, '
            f'line_style={self._line_style}, '
            f'point={self._center_geo}, '
            f'hatch_style={self._hatch_style}, '
            f'show_outline={self._show_outline}, '
            f'Transform={self._transform})'
        )
