import abc
from types import NoneType

import numpy as np
from functools import cached_property

from sstatics.core.preprocessing.dof import NodeDisplacement
from sstatics.core.preprocessing.loads import PointLoad, BarLineLoad
from sstatics.core.preprocessing.temperature import BarTemp

from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_POINT_FORCE, DEFAULT_LOAD_DISTANCE, DEFAULT_POINT_MOMENT,
    DEFAULT_ARROW_DISTANCE, DEFAULT_DISPLACEMENT
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.arrow import (
    StraightArrowGeo, CurvedArrowGeo, LineArrowGeo
)


class PointEffectGeo(ObjectGeo):

    _DEFAULT_FORCE = DEFAULT_POINT_FORCE
    _DEFAULT_MOMENT = DEFAULT_POINT_MOMENT

    def __init__(
            self,
            origin: tuple[float, float],
            effect: NodeDisplacement | PointLoad,
            distance: float | None = None,
            show_text: bool = True,
            **kwargs
    ):
        super().__init__(origin=origin, **kwargs)
        self._validate_point_effect(effect, distance, show_text)
        self._effect = effect
        self._distance = distance or DEFAULT_LOAD_DISTANCE
        self._show_text = show_text

    @cached_property
    def graphic_elements(self):
        effect = self._effect
        effect_rot = self._effect_rotation
        flip_rot = self._flip_moment_rotation
        post_trans_z = self._effect_transformation_z
        elements = []
        if effect.x:
            elements.append(self._create_force(
                np.pi / 2 + effect_rot, (0.0, 0.0), effect.x
            ))
        if effect.z:
            elements.append(self._create_force(
                effect_rot, (0.0, post_trans_z), effect.z)
            )
        if effect.phi:
            elements.append(self._create_moment(effect_rot + flip_rot))
        return elements

    @cached_property
    def text_elements(self):
        return []

    def _create_force(
            self, rotation: float, post_trans: tuple[float, float],
            value: float
    ):
        sign_rot = 0 if value > 0 else np.pi
        return StraightArrowGeo(
            self._origin, **self._DEFAULT_FORCE, distance=self._distance,
            text=self._text_value(value, is_phi=False),
            rotation=rotation + sign_rot, post_translation=post_trans,
            line_style=self._line_style, text_style=self._text_style
        )

    def _create_moment(self, rotation: float):
        angle_span = self._DEFAULT_MOMENT['angle_span']
        if self._effect.phi < 0:
            angle_span = angle_span[::-1]
        return CurvedArrowGeo(
            self._origin, **{**self._DEFAULT_MOMENT, 'angle_span': angle_span},
            text=self._text_value(self._effect.phi, is_phi=True),
            rotation=rotation,
            line_style=self._line_style, text_style=self._text_style
        )

    @abc.abstractmethod
    def _text_value(self, value: int | float, is_phi: bool):
        pass

    @property
    def _effect_rotation(self):
        return 0

    @property
    def _flip_moment_rotation(self):
        return 0

    @property
    def _effect_transformation_z(self):
        return 0

    @staticmethod
    def _validate_point_effect(effect, distance, show_text):
        if not all(hasattr(effect, a) for a in ('x', 'z', 'phi')):
            raise TypeError('"effect" must have numeric attributes x, z, phi')

        if not isinstance(distance, (int, float, NoneType)):
            raise TypeError(
                f'"distance" must be int, float or None, got '
                f'{type(distance).__name__!r}'
            )

        if not isinstance(show_text, bool):
            raise TypeError(
                f'"show_text" must be a boolean, got '
                f'{type(show_text).__name__!r}'
            )

    @property
    def distance(self):
        return self._distance

    @property
    def show_text(self):
        return self._show_text


class DisplacementGeo(PointEffectGeo):

    _DEFAULT_MOMENT = DEFAULT_DISPLACEMENT

    def __init__(
            self,
            origin: tuple[float, float],
            displacement: NodeDisplacement,
            **kwargs
    ):
        self._validate_displacement(displacement)
        super().__init__(origin=origin, effect=displacement, **kwargs)

    def _text_value(self, value: int | float, is_phi: bool):
        start = '\u03C6 = ' if is_phi else '\u03B4 = '
        end = ' rad' if is_phi else ' LE'
        text = start + str(abs(float(value))) + end if self._show_text else ''
        return text

    @property
    def _effect_transformation_z(self):
        return (
                self._DEFAULT_FORCE['length_head']
                + self._DEFAULT_FORCE['length_tail']
                + self._distance * 2
        ) if self._effect.z > 0 else 0

    @staticmethod
    def _validate_displacement(displacement):
        if not isinstance(displacement, NodeDisplacement):
            raise TypeError(
                f'"displacement" must be NodeDisplacement, got '
                f'{type(displacement).__name__!r}'
            )

    @property
    def displacement(self):
        return self._effect

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'displacement={self._effect}, '
            f'distance={self._distance}, '
            f'show_text={self._show_text}, '
            f'line_style={self._line_style}, '
            f'Transform={self._transform})'
        )


class PointLoadGeo(PointEffectGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            load: PointLoad,
            rotate_moment: float = 0.0,
            **kwargs
    ):
        self._validate_point_load(load, rotate_moment)
        super().__init__(origin=origin, effect=load, **kwargs)
        self._rotate_moment = rotate_moment

    def _text_value(self, value: int | float, is_phi: bool):
        return abs(float(value)) if self._show_text else ''

    @property
    def _effect_rotation(self):
        return self._effect.rotation

    @property
    def _flip_moment_rotation(self):
        return self._rotate_moment

    @staticmethod
    def _validate_point_load(load, rotate_moment):
        if not isinstance(load, PointLoad):
            raise TypeError(
                f'"load" must be NodePointLoad or BarPointLoad, got '
                f'{type(load).__name__!r}'
            )

        if not isinstance(rotate_moment, (float, int)):
            raise TypeError(
                f'"rotate_moment" must be a boolean, got '
                f'{type(rotate_moment).__name__!r}'
            )

    @property
    def load(self):
        return self._effect

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'load={self._effect}, '
            f'distance={self._distance}, '
            f'show_text={self._show_text}, '
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
            show_text: bool = True,
            arrow_style: dict | None = None,
            **kwargs
    ):
        self._validate_line_load(
            bar_coords, load, distance_to_bar, distance_to_arrow, show_text
        )
        super().__init__(
            origin=((bar_coords[0][0] + bar_coords[0][1]) / 2,
                    (bar_coords[1][0] + bar_coords[1][1]) / 2),
            **kwargs
        )
        self._bar_coords = bar_coords
        self._load = load
        self._distance_to_bar = distance_to_bar or DEFAULT_LOAD_DISTANCE
        self._distance_to_arrow = distance_to_arrow or DEFAULT_ARROW_DISTANCE
        self._show_text = show_text
        self._arrow_style = self._deep_style_merge(
            DEFAULT_POINT_FORCE, arrow_style or {}
        )

    @cached_property
    def graphic_elements(self):
        load = self._load
        (x0, x1), (z0, z1) = self._bar_coords
        load_angle = 0
        arrow_angle = 0

        if self._no_load(load, x0, x1, z0, z1):
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

    @staticmethod
    def _no_load(load, x0, x1, z0, z1):
        return any([
            load.pi == 0 and load.pj == 0,
            z0 == z1 and load.direction == 'x' and load.coord == 'system',
            x0 == x1 and load.direction == 'z' and load.coord == 'system',
        ])

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
    def _is_special_case(self):
        load = self._load
        _, (z0, z1) = self._bar_coords
        if load.direction == 'x' and load.coord == 'system' and z0 < z1:
            return True
        return False

    @property
    def _start_scale(self):
        if self._is_special_case:
            return self._load.pj / self._max_value
        return self._load.pi / self._max_value

    @property
    def _end_scale(self):
        if self._is_special_case:
            return self._load.pi / self._max_value
        return self._load.pj / self._max_value

    @property
    def _text_values(self):
        if not self._show_text:
            return ''
        if self._is_special_case:
            pi, pj = self._load.pj, self._load.pi
        else:
            pi, pj = self._load.pi, self._load.pj
        if pi == pj:
            return [abs(float(pi))]
        return [abs(float(pi)), abs(float(pj))]

    @staticmethod
    def _validate_line_load(
            bar_coords, load, distance_to_bar, distance_to_arrow, show_text
    ):
        if (
            not isinstance(bar_coords, (tuple, list))
            or len(bar_coords) != 2
            or not all(isinstance(v, list) and len(v) == 2 for v in bar_coords)
            or not all(
                isinstance(x, (int, float)) for lst in bar_coords for x in lst
            )
        ):
            raise TypeError(
                "bar_coords must be ((x0,x1),(z0,z1)) with numeric values.")

        if not isinstance(load, BarLineLoad):
            raise TypeError(
                f'"load" must be BarLineLoad, got {type(load).__name__!r}'
            )

        if not isinstance(distance_to_bar, (int, float, NoneType)):
            raise TypeError(
                f'"distance_to_bar" must be int, float or None, got '
                f'{type(distance_to_bar).__name__!r}'
            )

        if not isinstance(distance_to_arrow, (int, float, NoneType)):
            raise TypeError(
                f'"distance_to_arrow" must be int, float or None, got '
                f'{type(distance_to_arrow).__name__!r}'
            )

        if not isinstance(show_text, bool):
            raise TypeError(
                f'"show_text" must be a boolean, got '
                f'{type(show_text).__name__!r}'
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

    @property
    def show_text(self):
        return self._show_text

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'bar_coords={self._bar_coords}, '
            f'load={self._load}, '
            f'distance_to_bar={self._distance_to_bar}, '
            f'distance_to_arrow={self._distance_to_arrow}, '
            f'show_text={self._show_text}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )


class TempGeo(ObjectGeo):

    def __init__(
            self,
            bar_coords: tuple[list[float], list[float]],
            temp: BarTemp,
            **kwargs
    ):
        self._validate_temp(bar_coords, temp)
        super().__init__(
            origin=((bar_coords[0][0] + bar_coords[0][1]) / 2,
                    (bar_coords[1][0] + bar_coords[1][1]) / 2),
            **kwargs
        )
        self._bar_coords = bar_coords
        self._temp = temp
        self._rotation = self._transform.rotation

    @cached_property
    def graphic_elements(self):
        return []

    @cached_property
    def text_elements(self):
        to, tu = self._temp.temp_o, self._temp.temp_u

        if to == 0.0 and tu == 0.0:
            return []

        return [
            (
                *self._origin, [f'To = {to:g} K'], self._text_style, '0,0!',
                self._rotation
            ),
            (
                *self._origin, [f'Tu = {tu:g} K'], self._text_style, '0,2!',
                self._rotation
            )
        ]

    @staticmethod
    def _validate_temp(bar_coords, temp):
        if (
            not isinstance(bar_coords, (tuple, list))
            or len(bar_coords) != 2
            or not all(isinstance(v, list) and len(v) == 2 for v in bar_coords)
            or not all(
                isinstance(x, (int, float)) for lst in bar_coords for x in lst
            )
        ):
            raise TypeError(
                "bar_coords must be ((x0,x1),(z0,z1)) with numeric values.")

        if not isinstance(temp, BarTemp):
            raise TypeError(
                f'"temp" must be BarTemp, got {type(temp).__name__!r}'
            )

    @property
    def bar_coords(self):
        return self._bar_coords

    @property
    def temp(self):
        return self._temp

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'bar_coords={self._bar_coords}, '
            f'temp={self._temp}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
