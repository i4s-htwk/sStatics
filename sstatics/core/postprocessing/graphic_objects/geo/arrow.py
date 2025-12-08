
import abc
import math
from functools import cached_property

import numpy as np

from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    IsoscelesTriangleGeo, OpenCurveGeo, EllipseGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_ARROW, DEFAULT_ARROW_HEAD
)
from sstatics.core.postprocessing.graphic_objects.utils.transform import \
    Transform


class ArrowGeo(ObjectGeo, abc.ABC):
    CLASS_STYLES = {
        'line': DEFAULT_ARROW
    }

    def __init__(
            self,
            origin: tuple[float, float],
            width_head: float,
            length_head: float,
            **kwargs
    ):
        self._validate_arrow(width_head, length_head)
        super().__init__(origin=origin, **kwargs)
        self._width_head = width_head
        self._length_head = length_head
        self._x, self._z = self._origin

    @cached_property
    @abc.abstractmethod
    def graphic_elements(self):
        pass

    @cached_property
    @abc.abstractmethod
    def text_elements(self):
        pass

    @property
    def _head_style(self):
        return self._merge_style(DEFAULT_ARROW_HEAD, self._line_style)

    @staticmethod
    def _validate_arrow(width_head, length_head):
        if not isinstance(width_head, (int, float)):
            raise TypeError(
                f'"width_head" must be a number, got {width_head!r}'
            )

        if width_head < 0:
            raise ValueError(
                '"width_head" has to be a numbers greater than or equal to '
                'zero.'
            )

        if not isinstance(length_head, (int, float)):
            raise TypeError(
                f'"length_head" must be a number, got {length_head!r}'
            )

        if length_head < 0:
            raise ValueError(
                '"length_head" has to be a numbers greater than or equal to '
                'zero.'
            )

    @property
    def width_head(self):
        return self._width_head

    @property
    def length_head(self):
        return self._length_head


class StraightArrowGeo(ArrowGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            width_head: float,
            length_head: float,
            length_tail: float,
            distance: float = 0.0,
            **kwargs
    ):
        self._validate_straight_arrow(length_tail, distance)
        super().__init__(
            origin=origin, width_head=width_head, length_head=length_head,
            **kwargs
        )
        self._length_tail = length_tail
        self._distance = distance

    @cached_property
    def graphic_elements(self):
        head = IsoscelesTriangleGeo(
            (self._x, self._z), width=self._width_head,
            height=self._length_head, line_style=self._head_style,
            rotation=np.pi, post_translation=(0, -self._distance)
        )
        x = [self._x, self._x]
        z = [
            self._z - self._length_head,
            self._z - self._length_head - self._length_tail
        ]
        tail = OpenCurveGeo(
            x, z, post_translation=(0, -self._distance),
            line_style=self._line_style
        )
        return [head, tail]

    @cached_property
    def text_elements(self):
        return [(
            self._x,
            self._z - self._distance - self._length_head - self._length_tail,
            self._text, self._text_style
        )]

    @staticmethod
    def _validate_straight_arrow(length_tail, distance):
        if not isinstance(length_tail, (int, float)):
            raise TypeError(
                f'"length_tail" must be a number, got {length_tail!r}'
            )

        if length_tail < 0:
            raise ValueError(
                '"length_tail" has to be a numbers greater than or equal to '
                'zero.'
            )

        if not isinstance(distance, (int, float)):
            raise TypeError(
                f'"distance" must be a number, got {type(distance.__name__)!r}'
            )

    @property
    def length_tail(self):
        return self._length_tail

    @property
    def distance(self):
        return self._distance

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'width_head={self._width_head}, '
            f'length_head={self._length_head}, '
            f'length_tail={self._length_tail}, '
            f'distance={self._distance}, '
            f'line_style={self._line_style}, '
            f'Transform={self._transform})'
        )


class CurvedArrowGeo(ArrowGeo):

    def __init__(
            self,
            origin: tuple[float, float],
            width_head: float,
            length_head: float,
            radius: float,
            angle_span: tuple[float, float],
            **kwargs
    ):
        self._validate_curved_arrow(radius, angle_span)
        super().__init__(
            origin=origin, width_head=width_head, length_head=length_head,
            **kwargs
        )
        self._radius = radius
        self._angle_span = angle_span

    @cached_property
    def graphic_elements(self):
        tail = EllipseGeo(
            self._origin, width=2 * self._radius, angle_range=self._angle_span,
            line_style=self._line_style
        )
        return [self._head, tail]

    @cached_property
    def text_elements(self):
        _, end_angle = self._angle_span
        angle = end_angle + np.pi / 15
        x = self._x + self._radius * np.cos(angle)
        z = self._z + self._radius * np.sin(angle)
        return [(x, z, self._text, self._text_style)]

    @property
    def _head(self):
        angle_start, angle_end = self._angle_span
        direction = np.sign(angle_end - angle_start)
        angle_head = angle_end + direction * self._length_head / self._radius

        x = self._x + self._radius * np.cos(angle_head)
        z = self._z + self._radius * np.sin(angle_head)

        off = np.arctan(
            (self._width_head / 2) / (self._radius + self._length_head / 2)
        )
        rot = (
            (np.pi - angle_end - off) if direction == 1.0 else -angle_end + off
        )
        return IsoscelesTriangleGeo(
            (x, z), self._width_head, self._length_head,
            line_style=self._head_style, rotation=rot
        )

    @staticmethod
    def _validate_curved_arrow(radius, angle_span):
        if not isinstance(radius, (int, float)):
            raise TypeError(f'"radius" must be a number, got {radius!r}')

        if radius <= 0:
            raise ValueError('"radius" has to be a numbers greater than zero.')

        if not all(isinstance(angle, (int, float)) for angle in angle_span):
            raise ValueError(
                f'Both angles in "angle_span" must be numbers, '
                f'got {angle_span!r}.'
            )

    @property
    def radius(self):
        return self._radius

    @property
    def angle_span(self):
        return self._angle_span

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'width_head={self._width_head}, '
            f'length_head={self._length_head}, '
            f'radius={self._radius}, '
            f'angle_span={self._angle_span}, '
            f'line_style={self._line_style}, '
            f'Transform={self._transform})'
        )


class LineArrowGeo(ArrowGeo):

    def __init__(
            self,
            start: tuple[float, float],
            end: tuple[float, float],
            load_distance: float,
            load_angle: float,
            arrow_angle: float,
            arrow_spacing: float,
            start_scale: float,
            end_scale: float,
            width_head: float,
            length_head: float,
            length_tail: float,
            **kwargs
    ):
        super().__init__(
            origin=start, width_head=width_head, length_head=length_head,
            **kwargs
        )
        self._validate_line_arrow(
            end, load_distance, load_angle, arrow_angle, arrow_spacing,
            start_scale, end_scale, length_tail
        )
        self._start = start
        self._end = end
        self._load_distance = load_distance
        self._load_angle = self._extra_load_rotation(load_angle)
        self._arrow_angle = arrow_angle
        self._arrow_spacing = arrow_spacing
        self._start_scale = start_scale
        self._end_scale = end_scale
        self._length_tail = length_tail

    @cached_property
    def graphic_elements(self):
        return [*self._create_lines, *self._create_arrows]

    @cached_property
    def text_elements(self):
        return []

    @cached_property
    def _sin_a(self):
        return np.sin(self._load_angle)

    @cached_property
    def _cos_a(self):
        cos_a = np.cos(self._load_angle)
        return 1.0 if np.isclose(cos_a, 0) else cos_a

    @cached_property
    def _arrow_length(self):
        return self._length_head + self._length_tail

    @cached_property
    def _offset(self):
        neg_scale = min(self._start_scale, self._end_scale)
        return self._arrow_length * neg_scale if neg_scale < 0 else 0

    @cached_property
    def _bottom_dist(self):
        offset = 0 if self._arrow_angle else self._offset
        return self._load_distance * self._cos_a - offset

    @cached_property
    def _top_dist_start(self):
        return (
                self._load_distance * self._cos_a
                + self._arrow_length * self._start_scale - self._offset
        )

    @cached_property
    def _top_dist_end(self):
        return (
                self._load_distance * self._cos_a
                + self._arrow_length * self._end_scale - self._offset
        )

    @cached_property
    def _eff_spacing(self):
        angle = abs(self._arrow_angle)
        return (
            self._arrow_spacing + self._length_head * np.sin(angle)
            if self._arrow_angle else self._arrow_spacing
        )

    def _pre_translation(self, dist: float):
        (x0, z0), (x1, z1) = self._start, self._end
        if x0 == x1:
            sign = 1 if z0 < z1 else -1
            return sign * dist, 0.0
        sign = -1 if x0 < x1 else 1
        return 0.0, sign * dist

    def _extra_load_rotation(self, load_angle: float):
        (x0, _), (x1, _) = self._start, self._end
        if x0 > x1 and load_angle != 0:
            return load_angle + np.pi
        return load_angle

    @property
    def _extra_arrow_rotation(self):
        (x0, z0), (x1, z1) = self._start, self._end
        if x0 == x1:
            if z0 < z1:
                return -np.pi / 2
            else:
                return np.pi / 2
        if x0 < x1:
            return 0
        else:
            return np.pi

    def _transform_point(self, point, dist: float):
        x, z = point
        t = Transform(
            origin=(x, z), pre_translation=self._pre_translation(dist),
            rotation=self._load_angle
        )
        return t.apply(x, z)

    def _make_line(self, distances, has_text):
        xz = [
            self._transform_point(p, d)
            for p, d in zip((self._start, self._end), distances)
        ]
        x, z = zip(*xz)
        text = self._text if has_text else ''
        return OpenCurveGeo(
            x=list(x), z=list(z), line_style=self._line_style, text=text,
            text_style=self._text_style
        )

    @property
    def _create_lines(self):
        lines = [self._make_line(
                distances=(self._bottom_dist, self._bottom_dist),
                has_text=(True if self._arrow_angle else False)
            )
        ]
        if not self._arrow_angle:
            lines.append(self._make_line(
                distances=(self._top_dist_start, self._top_dist_end),
                has_text=True
            ))
        return lines

    @property
    def _create_arrows(self):
        elements = []
        x, z = self._create_lines[0].x, self._create_lines[0].z
        bottom_start = np.array([x[0], z[0]])
        bottom_end = np.array([x[1], z[1]])
        vec = bottom_end - bottom_start
        line_length = np.linalg.norm(vec)
        if line_length == 0:
            return elements

        n_arrows = max(int(math.ceil(line_length / self._eff_spacing)) + 1, 2)
        ts = np.linspace(0.0, 1.0, n_arrows)
        for t in ts:
            pos = bottom_start + t * vec
            scale = (
                    self._start_scale
                    + t * (self._end_scale - self._start_scale)
            )
            length = abs(self._arrow_length * scale)
            flip = self._arrow_length * scale < 0

            if abs(self._arrow_angle) == np.pi / 2:
                if (flip and t == 1.0) or (not flip and t == 0.0):
                    continue

            if self._arrow_angle:
                width_head = self._width_head * abs(scale)
                length_head = self._length_head * abs(scale)
            else:
                width_head = (
                    0 if length < self._length_head else self._width_head
                )
                length_head = (
                    0 if length < self._length_head else self._length_head
                )
            if length < self._length_head:
                length_tail = length * np.cos(self._arrow_angle)
            else:
                length_tail = (
                    (length - self._length_head) * np.cos(self._arrow_angle)
                )
            rotation = (
                self._load_angle + self._arrow_angle + (np.pi if flip else 0)
                + self._extra_arrow_rotation
            )

            elements.append(StraightArrowGeo(
                (float(pos[0]), float(pos[1])), width_head=width_head,
                length_head=length_head, length_tail=length_tail,
                rotation=rotation, line_style=self._line_style
            ))
        return elements

    @staticmethod
    def _validate_line_arrow(
            end, load_distance, load_angle, arrow_angle, arrow_spacing,
            start_scale, end_scale, length_tail
    ):
        if not isinstance(end, tuple) or len(end) != 2:
            raise TypeError(f'end must be a tuple of length 2, got {end!r}.')

        if not all(isinstance(v, (int, float)) for v in end):
            raise TypeError(f'end must contain numbers, got {end!r}.')

        if not isinstance(load_distance, (int, float)):
            raise TypeError(
                f'"load_distance" must be a number, got '
                f'{type(load_distance.__name__)!r}'
            )

        if not isinstance(load_angle, (int, float)):
            raise TypeError(
                f'"load_angle" must be a number, got '
                f'{type(load_angle.__name__)!r}'
            )

        if not isinstance(arrow_angle, (int, float)):
            raise TypeError(
                f'"arrow_angle" must be a number, got '
                f'{type(arrow_angle.__name__)!r}'
            )

        if not isinstance(arrow_spacing, (int, float)):
            raise TypeError(
                f'"arrow_spacing" must be a number, got {arrow_spacing!r}'
            )

        if arrow_spacing <= 0:
            raise ValueError(
                '"arrow_spacing" has to be a numbers greater than zero.'
            )

        if not isinstance(length_tail, (int, float)):
            raise TypeError(
                f'"length_tail" must be a number, got {length_tail!r}'
            )

        if length_tail <= 0:
            raise ValueError(
                '"length_tail" has to be a numbers greater than zero.'
            )

        if not isinstance(start_scale, (int, float)):
            raise TypeError(
                f'"start_scale" must be a number, got '
                f'{type(start_scale.__name__)!r}'
            )

        if not isinstance(end_scale, (int, float)):
            raise TypeError(
                f'"end_scale" must be a number, got '
                f'{type(end_scale.__name__)!r}'
            )

    @property
    def start(self):
        return self._start

    @property
    def end(self):
        return self._end

    @property
    def load_distance(self):
        return self._load_distance

    @property
    def load_angle(self):
        return self._load_angle

    @property
    def arrow_angle(self):
        return self._arrow_angle

    @property
    def arrow_spacing(self):
        return self._arrow_spacing

    @property
    def start_scale(self):
        return self._start_scale

    @property
    def end_scale(self):
        return self._end_scale

    @property
    def length_tail(self):
        return self._length_tail

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'start={self._start}, '
            f'end={self._end}, '
            f'load_distance={self._load_distance}, '
            f'load_angle={self._load_angle}, '
            f'arrow_angle={self._arrow_angle}, '
            f'arrow_spacing={self._arrow_spacing}, '
            f'start_scale={self._start_scale}, '
            f'end_scale={self._end_scale}, '
            f'width_head={self._width_head}, '
            f'length_head={self._length_head}, '
            f'length_tail={self._length_tail}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
