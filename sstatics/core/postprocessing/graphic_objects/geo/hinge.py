
import abc
from typing import Type

import numpy as np
from functools import cached_property

from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    OpenCurveGeo, EllipseGeo, RectangleGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_HINGE, DEFAULT_NORMAL_HINGE, DEFAULT_SHEAR_HINGE,
    DEFAULT_MOMENT_HINGE, DEFAULT_FILL_WHITE, DEFAULT_FULL_MOMENT_HINGE
)


class HingeGeo(ObjectGeo, abc.ABC):
    """
    Abstract base class for graphical hinge representations.

    This class defines the interface for hinge symbols in 2D plots,
    including geometric dimensions, line styles, and optional background
    shapes.

    Notes
    -----
    Subclasses must implement :py:meth:`_curve` and :py:meth:`_background`
    to define the visual appearance of the hinge.

    Parameters
    ----------
    origin : tuple[float, float]
        Pivot point of the hinge symbol.
    width : float, optional
        Symbol width. Defaults to :py:attr:`CLASS_DIMENSIONS['width']`.
    height : float, optional
        Symbol height. Defaults to :py:attr:`CLASS_DIMENSIONS['height']`.
    **kwargs
        Additional keyword arguments forwarded to :py:class:`ObjectGeo`.

    Raises
    ------
    TypeError
        If width or height is not numeric.
    ValueError
        If width or height is less than or equal to zero.
    """

    CLASS_STYLES = {
        'line': DEFAULT_HINGE
    }
    CLASS_DIMENSIONS: dict[str, float]

    def __init_subclass__(cls):
        """
        Validate that subclasses define CLASS_DIMENSIONS.

        Raises
        ------
        TypeError
            If :py:attr:`CLASS_DIMENSIONS` is missing.
        """
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
        self._validate_hinge(width, height)
        super().__init__(origin=origin, **kwargs)
        self._width = width
        self._height = height
        self._x0, self._z0 = self._origin

    @cached_property
    def graphic_elements(self):
        """
        Return the graphic elements of the hinge, including background and
        the hinge curve.

        Returns
        -------
        list
            A list of graphic primitives for rendering.
        """
        elements = []
        if hasattr(self, '_background') and self._background:
            elements.append(self._background)
        elements.extend(self._curve)
        return elements

    @cached_property
    def text_elements(self):
        """
        Return the text element of the hinge, positioned at the origin.

        Returns
        -------
        list[tuple[float, float, str, dict]]
            List containing a single text element.
        """
        x0, z0 = self._origin
        return [(x0, z0, self._text, self._text_style)]

    @cached_property
    @abc.abstractmethod
    def _curve(self):
        """Return the geometric primitives representing the hinge symbol.

        Must be implemented by subclasses.
        """
        pass

    @cached_property
    @abc.abstractmethod
    def _background(self):
        """Return the background shape of the hinge, e.g., a filled rectangle
        or ellipse.

        Must be implemented by subclasses.
        """
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
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'width={self._width}, '
            f'height={self._height}, '
            f'line_style={self._line_style}, '
            f'Transform={self._transform})'
        )


class NormalHingeGeo(HingeGeo):
    """
    Geometric representation of a normal hinge.

    The hinge consists of a rectangular curve with background rectangle for
    visualization.
    """

    CLASS_DIMENSIONS = DEFAULT_NORMAL_HINGE

    @cached_property
    def _curve(self):
        x = [
            self._x0 + 3 / 4 * self._width, self._x0 - 1 / 4 * self._width,
            self._x0 - 1 / 4 * self._width, self._x0 + 3 / 4 * self._width
        ]
        z = [
            self._z0 - 1 / 2 * self._height, self._z0 - 1 / 2 * self._height,
            self._z0 + 1 / 2 * self._height, self._z0 + 1 / 2 * self._height
        ]
        return [OpenCurveGeo(x, z, line_style=self._line_style)]

    @cached_property
    def _background(self):
        return RectangleGeo(
            (self._x0 - 1 / 8 * self._width, self._z0), self._width / 4,
            self._height, show_outline=False, line_style=DEFAULT_FILL_WHITE
        )


class ShearHingeGeo(HingeGeo):
    """
    Geometric representation of a shear hinge.

    The hinge is represented by two vertical lines with background
    rectangle for visualization.
    """

    CLASS_DIMENSIONS = DEFAULT_SHEAR_HINGE

    @cached_property
    def _curve(self):
        offsets = (-1 / 2 * self._width, 1 / 2 * self._width)
        return [
            OpenCurveGeo.from_center(
                (self._x0 + dx, self._z0), self._height,
                rotation=np.pi / 2, line_style=self._line_style
            ) for dx in offsets
        ]

    @cached_property
    def _background(self):
        return RectangleGeo(
            self._origin, self._width, self._height,
            show_outline=False, line_style=DEFAULT_FILL_WHITE
        )


class MomentHingeGeo(HingeGeo):
    """
    Geometric representation of a moment hinge.

    The hinge is represented by a circle with background circle.
    """

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


class FullMomentHingeGeo(MomentHingeGeo):
    """
    Geometric representation of a full moment hinge.

    Extends :py:class:`MomentHingeGeo` with class-specific dimensions.
    """

    CLASS_DIMENSIONS = DEFAULT_FULL_MOMENT_HINGE


class CombiHingeGeo(ObjectGeo):
    """
    Combine multiple hinge symbols into a single graphic object.

    Parameters
    ----------
    origin : tuple[float, float]
        Reference point for the composite hinge object.
    *hinges : HingeGeo instances or classes
        Hinge objects or subclasses to include in the composite symbol.
    **kwargs
        Additional keyword arguments forwarded to :py:class:`ObjectGeo`.

    Notes
    -----
    The composite object automatically positions the hinges next to each
    other and aggregates their graphical elements.
    """

    def __init__(
            self,
            origin: tuple[float, float],
            *hinges: tuple[HingeGeo | Type[HingeGeo], ...],
            **kwargs
    ):
        self._validate_init(hinges)
        super().__init__(origin=origin, **kwargs)
        self._hinges = [
            h((0, 0)) if isinstance(h, type) else h for h in hinges
        ]

    @cached_property
    def graphic_elements(self):
        """
        Return the graphic elements of the composite hinge object.

        This method positions each hinge horizontally relative to the previous
        one and collects their individual graphical elements into a single
        list.

        Returns
        -------
        list
            A list of :py:class:`ObjectGeo` instances representing all hinges
            in the composite object.

        Notes
        -----
        The positions of the hinges are automatically computed based on their
        width and type (normal or other hinge) to ensure proper spacing.
        """
        elements = []
        x, z = self._origin
        for i, hinge in enumerate(self._hinges):
            x += self._x_off(i)
            hinge_cls = type(hinge)
            elements.append(hinge_cls(
                (x, z), hinge.width, hinge.height, line_style=hinge.line_style
            ))
        return elements

    @cached_property
    def text_elements(self):
        """
        Return the text elements for the composite hinge object.

        The text is positioned at the origin of the composite hinge.

        Returns
        -------
        list[tuple[float, float, str, dict]]
            List containing a single text element for rendering.
        """
        x0, z0 = self._origin
        return [(x0, z0, self._text, self._text_style)]

    def _x_off(self, i):
        """
        Compute horizontal offset for the i-th hinge in a composite object.
        """
        if i == 0:
            return self._width_factor(self._hinges[i], is_prev=False)
        prev, curr = self._hinges[i - 1], self._hinges[i]
        return (
            self._width_factor(prev, is_prev=True) +
            self._width_factor(curr, is_prev=False)
        )

    @staticmethod
    def _width_factor(hinge, is_prev: bool):
        """
        Return the width factor of a hinge depending on its position in the
        sequence.
        """
        factor = (3 / 4 if is_prev else 1 / 4) \
            if isinstance(hinge, NormalHingeGeo) else 1 / 2
        return factor * hinge.width

    @staticmethod
    def _validate_init(hinges):
        if not all(
            (isinstance(h, type) and issubclass(h, HingeGeo))
            or issubclass(type(h), HingeGeo)
            for h in hinges
        ):
            raise ValueError(
                'each hinge must be an instance or a subclass of Hinge'
            )

    @property
    def hinges(self):
        return self._hinges

    @property
    def last_hinge(self):
        """
        Return the last hinge in the composite object.
        """
        return self._hinges[-1]

    @property
    def total_width(self):
        """
        Return the sum of the widths of all hinges in the composite object.
        """
        return sum(h.width for h in self._hinges)

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'hinges={self._hinges}, '
            f'line_style={self._line_style}, '
            f'Transform={self._transform})'
        )
