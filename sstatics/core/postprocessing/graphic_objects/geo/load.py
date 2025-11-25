from types import NoneType

import numpy as np
from sympy.core.cache import cached_property

from sstatics.core.preprocessing.loads import PointLoad, BarLineLoad
from sstatics.core.postprocessing.graphic_objects.geo.arrow import (
    StraightArrowGeo, CurvedArrowGeo, LineArrowGeo
)
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_POINT_FORCE, DEFAULT_LOAD_DISTANCE, DEFAULT_POINT_MOMENT,
    DEFAULT_ARROW_DISTANCE
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo


class PointLoadGeo(ObjectGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            load: PointLoad,
            distance: float | None = None,
            flip_moment: bool = False,
            **kwargs
    ):
        super().__init__(origin=origin, **kwargs)
        self._validate_point_load(load, distance, flip_moment)
        self._load = load
        self._distance = distance or DEFAULT_LOAD_DISTANCE
        self._flip_moment = flip_moment

    @cached_property
    def graphic_elements(self):
        elements = []
        if self._load.x:
            elements.append(self._create_force(np.pi / 2, self._load.x))
        if self._load.z:
            elements.append(self._create_force(0, self._load.z))
        if self._load.phi:
            elements.append(self._create_moment)
        return elements

    @cached_property
    def text_elements(self):
        return []

    def _create_force(self, base_rot: float, value: float):
        sign_rot = 0 if value > 0 else np.pi
        return StraightArrowGeo(
            self._origin, **DEFAULT_POINT_FORCE, distance=self._distance,
            text=abs((float(value))),
            rotation=base_rot + self._load.rotation + sign_rot,
            line_style=self._line_style
        )

    @property
    def _create_moment(self):
        flip_rot = np.pi if self._flip_moment else 0
        angle_span = DEFAULT_POINT_MOMENT['angle_span']
        if self._load.phi < 0:
            angle_span = angle_span[::-1]
        return CurvedArrowGeo(
            self._origin, **{**DEFAULT_POINT_MOMENT, 'angle_span': angle_span},
            text=abs(float(self._load.phi)),
            rotation=self._load.rotation + flip_rot,
            line_style=self._line_style
        )

    @staticmethod
    def _validate_point_load(load, distance, flip_moment):
        if not isinstance(load, PointLoad):
            raise TypeError(
                f'"load" must be NodePointLoad or BarPointLoad, got '
                f'{type(load).__name__!r}'
            )

        if not isinstance(distance, (int, float, NoneType)):
            raise TypeError(
                f'"distance" must be int, float or None, got '
                f'{type(distance).__name__!r}'
            )

        if not isinstance(flip_moment, bool):
            raise TypeError(
                f'"flip_moment" must be a bbolean, got '
                f'{type(flip_moment).__name__!r}'
            )

    @property
    def load(self):
        return self._load

    @property
    def distance(self):
        return self._distance

    @property
    def flip_moment(self):
        return self._flip_moment

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'load={self._load}, '
            f'distance={self._distance}, '
            f'flip_moment={self._flip_moment}, '
            f'line_style={self._line_style}, '
            f'Transform={self._transform})'
        )


class LineLoadGeo(ObjectGeo):

    def __init__(
            self,
            bar_coords: tuple[list[float], list[float]],
            load: BarLineLoad,
            distance_to_bar: float | None = None,
            distance_to_arrow: float | None = None,
            arrow_style: dict | None = None,
            **kwargs
    ):
        super().__init__(
            origin=((bar_coords[0][0] + bar_coords[0][1]) / 2,
                    (bar_coords[1][0] + bar_coords[1][1]) / 2),
            **kwargs
        )
        self._validate_line_load(load, distance_to_bar, distance_to_arrow)
        self._load = load
        self._bar_coords = bar_coords
        self._distance_to_bar = distance_to_bar or DEFAULT_LOAD_DISTANCE
        self._distance_to_arrow = distance_to_arrow or DEFAULT_ARROW_DISTANCE
        self._arrow_style = self._deep_style_merge(
            DEFAULT_POINT_FORCE, arrow_style or {}
        )

    @cached_property
    def graphic_elements(self):
        load = self._load
        (x0, x1), (z0, z1) = self._bar_coords
        load_angle = 0
        arrow_angle = 0

        if load.pi == 0 and load.pj == 0:
            return []

        if load.direction == 'z':
            if x0 == x1 and load.coord == 'system' and load.length == 'proj':
                raise ValueError(
                    'Combination of "z", "system" and "proj" does not work '
                    'on a vertical bar.'
                )
            load_values = self._handle_z_direction(
                load, x0, x1, z0, z1, load_angle, arrow_angle
            )
        else:
            load_values = self._handle_x_direction(
                load, x0, x1, z0, z1, load_angle, arrow_angle
            )

        return [LineArrowGeo(
            **load_values, load_distance=self._distance_to_bar,
            arrow_spacing=self._distance_to_arrow,
            start_scale=self._start_scale, end_scale=self._end_scale,
            **self._arrow_style, text=self._text_values,
            line_style=self._line_style, text_style=self._text_style
        )]

    @cached_property
    def text_elements(self):
        return []

    def _handle_z_direction(
            self, load, x0, x1, z0, z1, load_angle, arrow_angle
    ):
        if x0 != x1:
            if load.coord == 'system' and load.length == 'proj':
                z0 = z1 = min(z0, z1)
                x0, x1 = sorted((x0, x1))
            elif load.coord == 'bar':
                load_angle = self._inclination
        else:
            if load.coord == 'system' and load.length == 'exact':
                arrow_angle = np.pi / 2

        return dict(
            start=(x0, z0), end=(x1, z1), load_angle=load_angle,
            arrow_angle=arrow_angle
        )

    def _handle_x_direction(
            self, load, x0, x1, z0, z1, load_angle, arrow_angle
    ):
        if load.coord == 'bar':
            load_angle = 0 if x0 == x1 else self._inclination
            arrow_angle = np.pi / 2
        elif load.coord == 'system' and load.length == 'proj':
            x0 = x1 = min(x0, x1)
            z0, z1 = max(z0, z1), min(z0, z1)
        elif load.coord == 'system' and load.length == 'exact':
            if x0 == x1:
                z0, z1 = max(z0, z1), min(z0, z1)
            else:
                load_angle = np.pi / 2
        return dict(
            start=(x0, z0), end=(x1, z1), load_angle=load_angle,
            arrow_angle=arrow_angle
        )

    @cached_property
    def _inclination(self):
        (x0, x1), (z0, z1) = self._bar_coords
        return np.arctan2(-z1 + z0, x1 - x0)

    @property
    def _max_value(self):
        return max(abs(self._load.pi), abs(self._load.pj))

    @property
    def _start_scale(self):
        return self._load.pi / self._max_value

    @property
    def _end_scale(self):
        return self._load.pj / self._max_value

    @property
    def _text_values(self):
        pi, pj = self._load.pi, self._load.pj
        if pi == pj:
            return [abs(float(pi))]
        return [abs(float(pi)), abs(float(pj))]

    @staticmethod
    def _validate_line_load(load, distance_to_bar, distance_to_arrow):
        if not isinstance(load, BarLineLoad):
            raise TypeError(
                f'"load" must be BarLineLoad, got {type(load).__name__}'
            )

        if not isinstance(distance_to_bar, (int, float, NoneType)):
            raise TypeError(
                f'"distance_to_bar" must be int, float or None, got '
                f'{type(distance_to_bar).__name__}'
            )

        if not isinstance(distance_to_arrow, (int, float, NoneType)):
            raise TypeError(
                f'"distance_to_arrow" must be int, float or None, got '
                f'{type(distance_to_arrow).__name__}'
            )

    @property
    def bar_coords(self):
        return self._bar_coords

    @property
    def load(self):
        return self._load

    @property
    def distance_to_bar(self):
        return self._distance_to_bar

    @property
    def distance_to_arrow(self):
        return self._distance_to_arrow

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'bar_coords={self._bar_coords}, '
            f'load={self._load}, '
            f'distance_to_bar={self._distance_to_bar}, '
            f'distance_to_arrow={self._distance_to_arrow}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
