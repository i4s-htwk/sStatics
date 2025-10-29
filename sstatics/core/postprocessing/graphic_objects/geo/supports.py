
import abc
from functools import cached_property

import numpy as np

from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    IsoscelesTriangleGeo, OpenCurveGeo, RectangleGeo, PointGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_SUPPORT, DEFAULT_ROLLER_SUPPORT, DEFAULT_PINNED_SUPPORT,
    DEFAULT_FIXED_SUPPORT_UW, DEFAULT_FIXED_SUPPORT_WPHI,
    DEFAULT_CHAMPED_SUPPORT, DEFAULT_FIXED_SUPPORT_UPHI, DEFAULT_SUPPORT_HATCH,
    DEFAULT_CHAMPED_SUPPORT_HATCH
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
                f"Class '{cls.__name__}' must define a CLASS_DIMENSIONS "
                f"attribute."
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


class FixedSupportUW(SupportGeo):
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


FixedSupportUPhi = LineHatchGeo
""" Alias of :py:class:`LineHatchGeo` to make the use case of this class more
clear. """


class FixedSupportWPhi(SupportGeo):
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


ChampedSupport = DoubleLineHatchGeo
""" Alias of :py:class:`DoubleLineHatchGeo` to make the use case of this
class more clear. """
