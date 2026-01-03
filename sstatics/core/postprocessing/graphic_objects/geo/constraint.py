
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
    DEFAULT_CHAMPED_SUPPORT_HATCH, DEFAULT_TRANSLATIONAL_SPRING,
    DEFAULT_FILL_WHITE, DEFAULT_TORSIONAL_SPRING, DEFAULT_FREE_NODE
)


class ConstraintGeo(ObjectGeo, abc.ABC):
    """
    Abstract base class for geometric representations of supports and springs.

    This class serves as a common base for all graphical constraint symbols
    used in statical system plots, such as fixed supports, pinned supports,
    roller supports, or free nodes.

    It extends :py:class:`ObjectGeo` by introducing a standardized width and
    height concept, which defines the characteristic dimensions of the
    support symbol.

    Subclasses must define a class-level :py:attr:`CLASS_DIMENSIONS`
    dictionary specifying default values for ``width`` and ``height``.

    Parameters
    ----------
    origin : tuple[float, float]
        Reference point of the constraint symbol, usually coinciding with
        the associated node position.
    width : float, optional
        Width of the constraint symbol. If not provided, the default value
        from :py:attr:`CLASS_DIMENSIONS` is used.
    height : float, optional
        Height of the constraint symbol. If not provided, the default value
        from :py:attr:`CLASS_DIMENSIONS` is used.
    **kwargs
        Additional keyword arguments forwarded to :py:class:`ObjectGeo`.

    Raises
    ------
    TypeError
        If ``width`` or ``height`` is not numeric.
    ValueError
        If ``width`` or ``height`` is negative.
    """

    CLASS_STYLES = {
        'line': DEFAULT_SUPPORT
    }
    CLASS_DIMENSIONS: dict[str, float]

    def __init_subclass__(cls):
        """Validate subclass definitions.

        Ensures that every subclass defines a :py:attr:`CLASS_DIMENSIONS`
        attribute. This enforces a consistent geometric interface for all
        constraint symbols.

        Raises
        ------
        TypeError
            If the subclass does not define :py:attr:`CLASS_DIMENSIONS`.
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
        self._validate_support(width, height)
        super().__init__(origin=origin, **kwargs)
        self._width = width
        self._height = height
        self._x0, self._z0 = self._origin

    @cached_property
    @abc.abstractmethod
    def graphic_elements(self):
        """
        Return the geometric primitives representing the constraint symbol.

        Returns
        -------
        list
            A list of graphic elements used to render the constraint.

        Notes
        -----
        Subclasses must implement this property to define the visual
        appearance of the constraint symbol.
        """
        pass

    @cached_property
    def text_elements(self):
        """
        Return the text elements associated with the constraint.

        The text is rendered at the origin of the constraint symbol.

        Returns
        -------
        list[tuple[float, float, str, dict]]
            List containing a single text element.
        """
        return [(*self._origin, self._text, self._text_style)]

    @staticmethod
    def _validate_support(width, height):
        if not isinstance(width, (int, float)):
            raise TypeError(f'"width" must be a number, got {width!r}')

        if not isinstance(height, (int, float)):
            raise TypeError(f'"height" must be a number, got {height!r}')

        if width < 0 or height < 0:
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


class LineHatchGeo(ConstraintGeo):
    """
    Geometric representation of a clamped support using a line hatch pattern.

    This symbol is typically used to visualize fully fixed constraints
    involving rotational degrees of freedom.

    This class is also used by :py:class:`DoubleLineHatchGeo` and
    :py:class:`FixedSupportWPhiGeo`.
    """

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


class DoubleLineHatchGeo(ConstraintGeo):
    """
    Geometric representation of a double-hatched clamped support.

    Used for visualizing fixed constraints acting on multiple degrees of
    freedom.

    This class is also used by :py:class:`FixedSupportUPhiGeo` and
    :py:class:`PinnedSupportGeo`.
    """

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


class FreeNodeGeo(ConstraintGeo):
    """
    Geometric representation of a free node.

    This symbol typically consists of a single point without any constraint
    indicators.
    """

    CLASS_DIMENSIONS = DEFAULT_FREE_NODE

    @cached_property
    def graphic_elements(self):
        return [PointGeo(origin=self._origin, point_style=self._point_style)]


class RollerSupportGeo(ConstraintGeo):
    """
    Geometric representation of a roller support.

    Allows rotation and translation in one direction while restricting motion
    in the vertical direction.
    """

    CLASS_DIMENSIONS = DEFAULT_ROLLER_SUPPORT

    @cached_property
    def graphic_elements(self):
        triangle = IsoscelesTriangleGeo(
            origin=self._origin, width=self._width,
            height=3 / 4 * self._height, line_style=self._line_style
        )
        line = OpenCurveGeo.from_center(
            origin=(self._x0, self._z0 + self._height), length=self._width,
            line_style=self._line_style
        )
        return [triangle, line]


class PinnedSupportGeo(ConstraintGeo):
    """
    Geometric representation of a pinned (hinged) support.

    Restricts translational degrees of freedom while allowing rotation.
    """

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


class FixedSupportUWGeo(ConstraintGeo):
    """
    Geometric representation of a fixed support restraining translations
    in both coordinate directions.
    """

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


class FixedSupportWPhiGeo(ConstraintGeo):
    """
    Geometric representation of a fixed support restraining vertical
    translation and rotation.
    """

    CLASS_DIMENSIONS = DEFAULT_FIXED_SUPPORT_WPHI

    @cached_property
    def graphic_elements(self):
        top_line = OpenCurveGeo.from_center(
            (self._x0 + self._width / 4, self._z0 - 3 / 8 * self._height),
            length=5 / 6 * self._width, line_style=self._line_style
        )
        bottom_line = OpenCurveGeo.from_center(
            (self._x0 + self._width / 4, self._z0 + 3 / 8 * self._height),
            length=5 / 6 * self._width, line_style=self._line_style
        )
        line_hatch = LineHatchGeo(
            (self._x0 - self._width / 6, self._z0),
            width=self._width / 6, height=self._height,
            line_style=self._line_style
        )
        point = PointGeo(self._origin, point_style=self._point_style)
        return [top_line, bottom_line, line_hatch, point]


ClampedSupportGeo = LineHatchGeo
""" Alias of :py:class:`LineHatchGeo` to make the use case of this class more
clear. """


class TranslationalSpringGeo(ConstraintGeo):
    """
    Graphic representation of a translational spring.

    This object depicts a vertical spring with a zigzag line, a top horizontal
    line, and a small circle at the upper end.

    The spring's width and height can be customized, otherwise the defaults
    from :py:attr:`CLASS_DIMENSIONS` are used.
    """
    CLASS_DIMENSIONS = DEFAULT_TRANSLATIONAL_SPRING

    @cached_property
    def graphic_elements(self):
        x = [
            self._x0, self._x0,
            *[self._x0 - self._width / 4, self._x0 + self._width / 4] * 3,
            self._x0, self._x0
        ]
        z = [
            self._z0,
            *[self._z0 + i / 11 * self._height for i in range(2, 10)],
            self._z0 + self._height
        ]
        zigzag = OpenCurveGeo(x, z, line_style=self._line_style)

        bottom_line = OpenCurveGeo(
            [self._x0 - self._width / 2, self._x0 + self._width / 2],
            [self._z0 + self._height, self._z0 + self._height],
            line_style=self._line_style
        )

        circle = EllipseGeo(
            origin=(self._x0, self._z0 + self.height), width=self._width / 8,
            height=self._height / 11, line_style=self._line_style
        )

        background = EllipseGeo(
            (self._x0, self._z0 + self.height), self._width/8,
            self._height/11, line_style=DEFAULT_FILL_WHITE, show_outline=False
        )

        return [zigzag, bottom_line, background, circle]


class TorsionalSpringGeo(ConstraintGeo):
    """
    Graphic representation of a torsional spring.

    This object depicts a spring that rotates about a point, represented by
    a vertical line and a partial circle to indicate the spring's curvature.

    The spring's width and height can be customized, otherwise the defaults
    from :py:attr:`CLASS_DIMENSIONS` are used.

    Notes
    -----
    The circle spans from π/2 to 2π radians, visually indicating the spring's
    torsional effect.
    """
    CLASS_DIMENSIONS = DEFAULT_TORSIONAL_SPRING

    @cached_property
    def graphic_elements(self):
        line = OpenCurveGeo.from_center(
            (self._x0 - self._width / 2, self._z0 + self._height * 7 / 16),
            length=self._z0 + self._height / 4,
            line_style=self._line_style, rotation=np.pi / 2
        )

        ellipse = EllipseGeo(
            origin=(self._x0 - self._width / 2, self._z0), width=self._width,
            height=self._height * 7 / 8, angle_range=(np.pi / 2, 2 * np.pi),
            line_style=self._line_style
        )
        return [line, ellipse]
