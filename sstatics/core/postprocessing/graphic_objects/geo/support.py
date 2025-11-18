
import abc
from functools import cached_property

import numpy as np

from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    EllipseGeo, IsoscelesTriangleGeo, OpenCurveGeo, RectangleGeo, PointGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_SUPPORT, DEFAULT_ROLLER_SUPPORT, DEFAULT_PINNED_SUPPORT,
    DEFAULT_FIXED_SUPPORT_UW, DEFAULT_FIXED_SUPPORT_WPHI,
    DEFAULT_CHAMPED_SUPPORT, DEFAULT_FIXED_SUPPORT_UPHI, DEFAULT_SUPPORT_HATCH,
    DEFAULT_CHAMPED_SUPPORT_HATCH, DEFAULT_SPRING_W, DEFAULT_FILL_WHITE,
    DEFAULT_SPRING_PHI
)


class SupportGeo(ObjectGeo, abc.ABC):
    CLASS_STYLES = {
        'line': DEFAULT_SUPPORT
    }
    CLASS_DIMENSIONS: dict[str, float]

    def __init_subclass__(cls):
        super().__init_subclass__()
        if not hasattr(cls, 'CLASS_DIMENSIONS'):
            raise TypeError(
                f'Class "{cls.__name__}" must define a CLASS_DIMENSIONS '
                f'attribute.'
            )

    def __init__(
            self,
            origin: tuple[float, float],
            width: float | None = None,
            height: float | None = None,
            **kwargs
    ):
        width = width if width is not None else self.CLASS_DIMENSIONS['width']
        height = (
            height if height is not None else self.CLASS_DIMENSIONS['height']
        )
        self._validate_support(width, height)
        super().__init__(origin=origin, **kwargs)
        self._width = width
        self._height = height
        self._x0, self._z0 = self._origin

    @cached_property
    @abc.abstractmethod
    def graphic_elements(self):
        pass

    @cached_property
    def text_elements(self):
        return [(*self._origin, self._text, self._text_style)]

    @staticmethod
    def _validate_support(width, height):
        if not isinstance(width, (int, float)):
            raise TypeError(f'"width" must be a number, got {width!r}')

        if not isinstance(height, (int, float)):
            raise TypeError(f'"height" must be a number, got {height!r}')

        if width <= 0 or height <= 0:
            raise ValueError(
                '"width" and "height" have to be a numbers greater than zero.'
            )

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'width={self._width}, '
            f'height={self._height}, '
            f'text={self._text}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )


class LineHatchGeo(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_CHAMPED_SUPPORT

    @cached_property
    def graphic_elements(self):
        line = OpenCurveGeo.from_center(
            self._origin, length=self._height, rotation=np.pi / 2,
            line_style=self._line_style
        )
        hatch_style = DEFAULT_CHAMPED_SUPPORT_HATCH.copy()
        hatch_style['line_style'] = self._line_style
        hatch = RectangleGeo(
            (self._x0 - self._width / 2, self._z0), width=self._width,
            height=self._height, hatch_style=hatch_style, show_outline=False,
            line_style=self._line_style,
        )
        return [line, hatch]


class DoubleLineHatchGeo(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_FIXED_SUPPORT_UPHI

    @cached_property
    def graphic_elements(self):
        line = OpenCurveGeo.from_center(
            self._origin, length=self._height, rotation=np.pi / 2,
            line_style=self._line_style
        )
        line_hatch = LineHatchGeo(
            (self._x0 - self._width / 2, self._z0),
            width=self._width / 2, height=self._height,
            line_style=self._line_style
        )
        return [line, line_hatch]


class RollerSupportGeo(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_ROLLER_SUPPORT

    @cached_property
    def graphic_elements(self):
        triangle = IsoscelesTriangleGeo(
            self._origin, width=self._width, height=3 / 4 * self._height,
            line_style=self._line_style
        )
        line = OpenCurveGeo.from_center(
            (self._x0, self._z0 + self._height), length=self._width,
            line_style=self._line_style
        )
        return [triangle, line]


class PinnedSupportGeo(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_PINNED_SUPPORT

    @cached_property
    def graphic_elements(self):
        top_line = OpenCurveGeo.from_center(
            (self._x0 + 3 / 14 * self._width,
             self._z0 - 3 / 8 * self._height), length=5 / 7 * self._width,
            line_style=self._line_style
        )
        bottom_line = OpenCurveGeo.from_center(
            (self._x0 + 3 / 14 * self._width,
             self._z0 + 3 / 8 * self._height), length=5 / 7 * self._width,
            line_style=self._line_style
        )
        double_line_hatch = DoubleLineHatchGeo(
            (self._x0 - self._width / 7, self._z0),
            width=2 / 7 * self._width, height=self._height,
            line_style=self._line_style
        )
        point = PointGeo(self._origin, point_style=self._point_style)
        return [top_line, bottom_line, double_line_hatch, point]


class FixedSupportUWGeo(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_FIXED_SUPPORT_UW

    @cached_property
    def graphic_elements(self):
        triangle = IsoscelesTriangleGeo(
            self._origin, self._width, 3 / 4 * self._height,
            line_style=self._line_style
        )
        hatch_style = DEFAULT_SUPPORT_HATCH.copy()
        hatch_style['line_style'] = self._line_style
        hatch = RectangleGeo(
            (self._x0, self._z0 + 7 / 8 * self._height),
            width=self._width, height=self._height / 4,
            hatch_style=hatch_style, show_outline=False,
            line_style=self._line_style,
        )
        return [triangle, hatch]


FixedSupportUPhiGeo = DoubleLineHatchGeo
""" Alias of :py:class:`DoubleLineHatchGeo` to make the use case of this class
more clear. """


class FixedSupportWPhiGeo(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_FIXED_SUPPORT_WPHI

    @cached_property
    def graphic_elements(self):
        top_line = OpenCurveGeo.from_center(
            (self._x0 + self._width / 4,
             self._z0 - 3 / 8 * self._height), length=5 / 6 * self._width,
            line_style=self._line_style
        )
        bottom_line = OpenCurveGeo.from_center(
            (self._x0 + self._width / 4,
             self._z0 + 3 / 8 * self._height), length=5 / 6 * self._width,
            line_style=self._line_style
        )
        line_hatch = LineHatchGeo(
            (self._x0 - self._width / 6, self._z0),
            width=self._width / 6, height=self._height,
            line_style=self._line_style
        )
        point = PointGeo(self._origin, point_style=self._point_style)
        return [top_line, bottom_line, line_hatch, point]


ChampedSupportGeo = LineHatchGeo
""" Alias of :py:class:`LineHatchGeo` to make the use case of this class more
clear. """


class SpringW(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_SPRING_W

    @cached_property
    def graphic_elements(self):

        vertical_line_top = OpenCurveGeo(
            [self._x0, self._x0],
            [self._z0, self._z0 + 2 / 11 * self._height],
            line_style=self._line_style
        )

        vertical_line_bottom = OpenCurveGeo(
            [self._x0, self._x0],
            [self._z0 + 9 / 11 * self._height, self._z0 + self._height],
            line_style=self._line_style
        )

        bottom_line = OpenCurveGeo(
            [self._x0 - self._width / 2, self._x0 + self._width / 2],
            [self._z0 + self._height, self._z0 + self._height],
            line_style=self._line_style
        )

        x_diagonal = [
            self._x0, self._x0 - self._width / 4, self._x0 + self._width / 4,
            self._x0 - self._width / 4, self._x0 + self._width / 4,
            self._x0 - self._width / 4, self._x0 + self._width / 4,
            self._x0
        ]

        z_diagonal = [
            self._z0 + 2 / 11 * self._height, self._z0 + 3 / 11 * self._height,
            self._z0 + 4 / 11 * self._height, self._z0 + 5 / 11 * self._height,
            self._z0 + 6 / 11 * self._height, self._z0 + 7 / 11 * self._height,
            self._z0 + 8 / 11 * self._height, self._z0 + 9 / 11 * self._height,
        ]

        diagonal_lines = OpenCurveGeo(
            x_diagonal, z_diagonal, line_style=self._line_style)

        circle = EllipseGeo((self._x0, self._z0 + self.height), self._width/8,
                            self._height/11, line_style=self._line_style)

        background = EllipseGeo(
            (self._x0, self._z0 + self.height), self._width/8,
            self._height/11, line_style=DEFAULT_FILL_WHITE, show_outline=False)

        return [vertical_line_top, vertical_line_bottom, diagonal_lines,
                bottom_line, circle, background]


class SpringPhi(SupportGeo):
    CLASS_DIMENSIONS = DEFAULT_SPRING_PHI

    @cached_property
    def graphic_elements(self):
        x_line = [
            self._x0 - self._width / 2, self._x0 - self._width / 2
        ]

        z_line = [
            self._z0 + 3 / 10 * self._height, self._z0 + 11 / 20 * self._height
        ]

        vertical_line = OpenCurveGeo(
            x_line, z_line, line_style=self._line_style)

        ellipse = EllipseGeo(
            (self._x0 - self._width / 2, self._z0), self._width,
            self._height * 7 / 8, (np.pi / 2, 2 * np.pi))

        return [vertical_line, ellipse]
