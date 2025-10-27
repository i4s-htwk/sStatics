
import abc

import numpy as np
from sympy.core.cache import cached_property

from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    OpenCurveGeo, EllipseGeo, RectangleGeo
)
from ..geo.object_geo import ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_HINGE, DEFAULT_NORMAL_HINGE, DEFAULT_SHEAR_HINGE,
    DEFAULT_MOMENT_HINGE, DEFAULT_FILL_WHITE
)


class HingeGeo(ObjectGeo, abc.ABC):
    CLASS_STYLES = {
        'line': DEFAULT_HINGE
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
        self._validate_hinge(width, height)
        super().__init__(origin=origin, **kwargs)
        self._width = width
        self._height = height

    @cached_property
    def graphic_elements(self):
        elements = []
        if hasattr(self, '_background') and self._background:
            elements.append(self._background)
        elements.extend(self._curve)
        return elements

    @cached_property
    def text_elements(self):
        x0, z0 = self._origin
        return [(x0, z0, self._text, self._text_style)]

    @cached_property
    @abc.abstractmethod
    def _curve(self):
        pass

    @cached_property
    @abc.abstractmethod
    def _background(self):
        pass

    @staticmethod
    def _validate_hinge(width, height):
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
            f"{self.__class__.__name__}("
            f"origin={self._origin}, "
            f"width={self._width}, "
            f"height={self._height}, "
            f"line_style={self._line_style}, "
            f"Transform={self._transform})"
        )


class NormalHingeGeo(HingeGeo):
    CLASS_DIMENSIONS = DEFAULT_NORMAL_HINGE

    @cached_property
    def _curve(self):
        x0, z0 = self._origin
        x = [
            x0 + 3 / 4 * self._width, x0 - 1 / 4 * self._width,
            x0 - 1 / 4 * self._width, x0 + 3 / 4 * self._width
        ]
        z = [
            z0 - 1 / 2 * self._height, z0 - 1 / 2 * self._height,
            z0 + 1 / 2 * self._height, z0 + 1 / 2 * self._height
        ]
        return [OpenCurveGeo(x, z, line_style=self._line_style)]

    @cached_property
    def _background(self):
        x0, z0 = self._origin
        return RectangleGeo(
            (x0 - 1 / 8 * self._width, z0), self._width / 4,
            self._height, show_outline=False, line_style=DEFAULT_FILL_WHITE
        )


class ShearHingeGeo(HingeGeo):
    CLASS_DIMENSIONS = DEFAULT_SHEAR_HINGE

    @cached_property
    def _curve(self):
        x0, z0 = self._origin
        offsets = (-1 / 2 * self._width, 1 / 2 * self._width)
        return [
            OpenCurveGeo.from_center(
                (x0 + dx, z0), self._height, rotation=np.pi / 2,
                line_style=self._line_style
            ) for dx in offsets
        ]

    @cached_property
    def _background(self):
        return RectangleGeo(
            self._origin, self._width, self._height,
            show_outline=False, line_style=DEFAULT_FILL_WHITE
        )


class MomentHingeGeo(HingeGeo):
    CLASS_DIMENSIONS = DEFAULT_MOMENT_HINGE

    @cached_property
    def _curve(self):
        return [EllipseGeo(
            self._origin, self._width, self._height,
            line_style=self._line_style
        )]

    @cached_property
    def _background(self):
        return EllipseGeo(
            self._origin, self._width, self._height,
            show_outline=False, line_style=DEFAULT_FILL_WHITE
        )
