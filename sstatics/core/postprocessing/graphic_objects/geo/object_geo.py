
from __future__ import annotations
import abc
from functools import cached_property
from types import NoneType
from typing import Any

from ..utils.defaults import (
    DEFAULT_LINE, DEFAULT_TEXT, DEFAULT_POINT
)
from ..utils.transform import Transform


class ObjectGeo(abc.ABC):
    """
    Abstract base class for all geometric graphic objects in 2D space.

    This class defines the common interface and shared functionality for all
    objects that can be rendered in a 2D statical system plot. It provides a
    unified transformation pipeline (translation, rotation, scaling), style
    handling, and recursive composition of graphic elements.

    Concrete subclasses represent geometric entities such as bars, nodes,
    loads, support symbols, or result visualizations (e.g. internal force
    diagrams).

    Parameters
    ----------
    origin : tuple[float, float], default=(0.0, 0.0)
        Local reference point used as pivot for rotation and scaling.
    rotation : float, default=0.0
        Rotation angle in radians, applied counterclockwise.
    scaling : float, default=1.0
        Uniform scaling factor applied relative to :py:attr:`origin`.
    pre_translation : tuple[float, float], default=(0.0, 0.0)
        Translation applied before rotation and scaling.
    post_translation : tuple[float, float], default=(0.0, 0.0)
        Translation applied after rotation and scaling.
    text : str or list[str], default=''
        Text labels or annotations associated with the object.
    line_style : dict, optional
        Style parameters for line-based graphic elements.
        Overrides default and class-level styles.
    point_style : dict, optional
        Style parameters for point-based graphic elements.
    text_style : dict, optional
        Style parameters for text rendering.
    global_scale : float, optional
        Global scaling factor used to normalize size-dependent visual
        elements. If provided, it overrides the automatically computed base
        scale.

    Raises
    ------
    TypeError
        If `text` is not a valid scalar or sequence type.
    TypeError
        If any of the style arguments is not a dictionary.
    TypeError
        If `global_scale` is not ``float``, ``int`` or ``None``.

    Notes
    -----
    This class is purely geometric and rendering-oriented. It does **not**
    perform any statical computations.

    Subclasses are expected to implement at least
    :py:meth:`graphic_elements` and :py:meth:`text_elements`.
    """
    _DEFAULT_STYLES = {
        'line': DEFAULT_LINE,
        'point': DEFAULT_POINT,
        'text': DEFAULT_TEXT
    }
    _DEFAULT_ELEMENT_STYLES = {}

    def __init__(
            self,
            origin: tuple[float, float] = (0.0, 0.0),
            rotation: float = 0.0,
            scaling: float = 1.0,
            pre_translation: tuple[float, float] = (0.0, 0.0),
            post_translation: tuple[float, float] = (0.0, 0.0),
            text: str | list[str] = '',
            line_style: dict[str, Any] | None = None,
            point_style: dict[str, Any] | None = None,
            text_style: dict[str, Any] | None = None,
            global_scale: float | None = None
    ):
        line_style = line_style or {}
        point_style = point_style or {}
        text_style = text_style or {}
        self._validate_base(
            text, line_style, text_style, point_style, global_scale
        )
        user_styles = {
            'line': line_style or {},
            'point': point_style or {},
            'text': text_style or {}
        }
        class_styles = getattr(self, 'CLASS_STYLES', {})
        self._transform = Transform(
            origin=origin, rotation=rotation, scaling=scaling,
            pre_translation=pre_translation, post_translation=post_translation
        )
        self._origin = self._transform.origin
        self._text = text if isinstance(text, (list, tuple)) else [str(text)]

        self._line_style = self._merge_style(
            self._DEFAULT_STYLES['line'],
            class_styles.get('line'),
            user_styles['line']
        )
        self._point_style = self._merge_style(
            self._DEFAULT_STYLES['point'],
            class_styles.get('point'),
            user_styles['point']
        )
        self._text_style = self._merge_style(
            self._DEFAULT_STYLES['text'],
            class_styles.get('text'),
            user_styles['text']
        )
        self._global_scale = global_scale

    @cached_property
    @abc.abstractmethod
    def graphic_elements(self) -> list[
        ObjectGeo | tuple[list[float], list[float], dict]
    ]:
        """
        Return the geometric point or line elements representing the object.

        Each element is either
        - another :py:class:`ObjectGeo` instance (composite object), or
        - a tuple ``(x, z, style)`` describing a polyline in local coordinates.

        The returned elements are interpreted recursively and transformed
        automatically by the object's transformation pipeline.

        Returns
        -------
        list[ObjectGeo | tuple[list[float], list[float], dict]]
            A list of graphic elements used for rendering.

        Notes
        -----
        Subclasses must override this property.

        An empty list may be returned if the object has no visible point or
        line representation.
        """
        return []

    @cached_property
    @abc.abstractmethod
    def text_elements(self) -> list[tuple[float, float, str, dict]]:
        """
        Return the text elements associated with the object.

        Each text element is defined by its position, content and style.

        Returns
        -------
        list[tuple[float, float, str, dict]]
            Tuples of the form ``(x, z, text, style)``.

        Notes
        -----
        Subclasses must override this property.

        An empty list may be returned if the object has no text annotations.
        """
        return []

    @property
    def _raw_graphic_elements(self):
        """Return the unprocessed graphic elements of the object.

        This method provides direct access to the elements returned by
        :py:meth:`graphic_elements` without applying any recursive resolution
        or transformation.

        Returns
        -------
        list
            Raw graphic elements as defined by :py:meth:`graphic_elements`.

        Notes
        -----
        This method exists mainly to support recursive traversal of composite
        :py:class:`ObjectGeo` instances. It can also be overridden by
        subclasses.
        """
        return self.graphic_elements

    @cached_property
    def _boundaries(self) -> tuple[float, float, float, float]:
        """Return the geometric bounding box of the object.

        The boundaries are computed from all recursively resolved graphic
        elements returned by :py:meth:`graphic_elements`, after applying all
        transformations.

        Returns
        -------
        tuple[float, float, float, float]
            Tuple ``(x_min, x_max, z_min, z_max)`` defining the enclosing
            rectangle.

        Notes
        -----
        If the object contains no geometric elements, all values are returned
        as ``0.0``.
        """
        x_coords, z_coords = [], []

        # Rekursive Iteration über alle Grafik-Elemente
        for x_list, z_list, _ in self._iter_raw_graphic_elements(self):
            x_coords.extend(x for x in x_list if x is not None)
            z_coords.extend(z for z in z_list if z is not None)

        if not x_coords or not z_coords:
            return 0.0, 0.0, 0.0, 0.0

        return min(x_coords), max(x_coords), min(z_coords), max(z_coords)

    def _iter_raw_graphic_elements(self, obj=None):
        """Iterate recursively over all graphic elements of an object.

        This generator resolves nested :py:class:`ObjectGeo` instances and
        yields fully transformed geometric primitives.

        Parameters
        ----------
        obj : :py:class:`ObjectGeo`, optional
            Object whose graphic elements are to be traversed. If omitted,
            the current instance is used.

        Yields
        ------
        tuple[list[float], list[float], dict]
            Transformed x- and z-coordinate lists together with the associated
            style dictionary.

        Notes
        -----
        Transformations are applied hierarchically, such that child objects
        inherit the transformation of their parent objects.
        """
        for element in obj.raw_graphic_elements:
            if self._is_geo_object(element):
                for x, z, style in self._iter_raw_graphic_elements(element):
                    x, z = obj.transform(x, z)
                    yield x, z, style
            else:
                x, z, style = element
                x, z = obj.transform(x, z)
                yield x, z, style

    @staticmethod
    def _is_geo_object(obj):
        """Check whether an object behaves like a geometric graphic object.

        Parameters
        ----------
        obj : any
            Object to be tested.

        Returns
        -------
        bool
            ``True`` if the object exposes a callable
            :py:meth:`raw_graphic_elements` method, ``False`` otherwise.

        Notes
        -----
        This method is used internally to distinguish between composite
        :py:class:`ObjectGeo` instances and primitive graphic elements.
        """
        return (
                hasattr(obj, "_raw_graphic_elements")
                and callable(obj.raw_graphic_elements)
        )

    @cached_property
    def _max_dimensions(self) -> tuple[float, float]:
        """Return the horizontal and vertical extents of the object.

         Based on the geometric _boundaries computed by :py:meth:`_boundaries`,
         this property determines the overall width and height of the object
         in local coordinates.

         Returns
         -------
         tuple[float, float]
             A tuple ``(dx, dz)`` representing the horizontal and vertical
             extents of the object.

         Notes
         -----
         This property is used internally by :py:meth:`_base_scale` to provide
         consistent scaling behavior for visual elements.
         """
        x_min, x_max, z_min, z_max = self._boundaries
        return x_max - x_min, z_max - z_min

    @property
    def _base_scale(self) -> float:
        """Return a normalized scaling factor based on the object size.

        The base scale is used to ensure that visual elements maintain
        consistent proportions across objects of different sizes. It depends
        on the maximum geometric extent of the object.

        Returns
        -------
        float
            Scaling factor computed as ``0.04 * max(dx, dz) + 0.01``, where
            ``dx`` and ``dz`` are the horizontal and vertical dimensions of
            the object.

        Notes
        -----
        If :py:attr:`global_scale` is set, it is returned directly.
        """
        if self._global_scale is not None:
            return self._global_scale
        dx, dz = self._max_dimensions
        return 0.04 * max(dx, 2 * dz) + 0.01

    @staticmethod
    def _merge_style(
            default_style: dict,
            class_style: dict | None = None,
            user_style: dict | None = None
    ) -> dict:
        """Merge default, class-level and user-defined styles.

        Styles are merged in increasing order of priority:
        default > class-level > user-defined

        Parameters
        ----------
        default_style : dict
            Base style dictionary.
        class_style : dict, optional
            Style dictionary defined at class level.
        user_style : dict, optional
            Style dictionary provided by the user.

        Returns
        -------
        dict
            Merged style dictionary.

        Notes
        -----
        Nested dictionaries are merged recursively using
        :py:meth:`_deep_style_merge`.
        """
        result = dict(default_style)
        for layer in (class_style, user_style):
            if layer:
                result = ObjectGeo._deep_style_merge(result, layer)
        return result

    @staticmethod
    def _deep_style_merge(
            default: dict,
            override: dict[str, Any]
    ) -> dict:
        """
        Recursively merge two dictionaries without modifying the originals.

        Parameters
        ----------
        default : dict
            Base dictionary providing default values.
        override : dict
            Dictionary with values to override the defaults.

        Returns
        -------
        dict
            A new dictionary containing merged values.

        Notes
        -----
        If both `default` and `override` have a key whose value is a dict, the
        merge is applied recursively.
        """
        result = dict(default)
        for k, v in override.items():
            if (
                k in result and isinstance(result[k], dict)
                and isinstance(v, dict)
            ):
                result[k] = ObjectGeo._deep_style_merge(result[k], v)
            else:
                result[k] = v
        return result

    def _resolve_style(self, element, default_style, user_style):
        """Resolve the effective style for a specific graphic element.

        The resolution order is:
        1. Explicit style for the given element
        2. Style matching the element's class name
        3. Default style entry
        4. Fallback to the provided default style

        Parameters
        ----------
        element : any
            Graphic element or identifier.
        default_style : dict
            Base style dictionary.
        user_style : dict
            User-defined style configuration.

        Returns
        -------
        dict
            Resolved style dictionary.

        Notes
        -----
        This method enables fine-grained styling of individual elements while
        maintaining sensible defaults.
        """
        if not user_style:
            return default_style

        if element in user_style:
            return self._merge_style(default_style, user_style[element])

        elem_type = type(element).__name__.lower()
        if elem_type in user_style:
            return self._merge_style(default_style, user_style[elem_type])

        if "default" in user_style:
            return self._merge_style(default_style, user_style["default"])

        return default_style

    @staticmethod
    def _validate_base(
            text, line_style, text_style, point_style, global_scale
    ):
        if not isinstance(text, (str, list, tuple, int, float)):
            raise TypeError(
                f'"text" must be string, list, tuple, int or float, got '
                f'{type(text).__name__}'
            )
        if isinstance(text, (list, tuple)):
            if not all(isinstance(t, (str, int, float)) for t in text):
                raise TypeError(
                    'all text elements must be strings, int or float'
                )

        if not isinstance(line_style, dict):
            raise TypeError(
                f'"line_style" must be a dictionary, got '
                f'{type(line_style).__name__}'
            )

        if not isinstance(point_style, dict):
            raise TypeError(
                f'"point_style" must be a dictionary, got '
                f'{type(point_style).__name__}'
            )

        if not isinstance(text_style, dict):
            raise TypeError(
                f'"text_style" must be a dictionary, got '
                f'{type(text_style).__name__}'
            )

        if not isinstance(global_scale, (int, float, NoneType)):
            raise TypeError(
                f'"global_scal" must be int, float or NoneType, got '
                f'{type(global_scale).__name__}'
            )

    @property
    def transform(self):
        return self._transform

    @property
    def origin(self):
        return self._origin

    @property
    def text(self):
        return self._text

    @property
    def line_style(self):
        return self._line_style

    @property
    def point_style(self):
        return self._point_style

    @property
    def text_style(self):
        return self._text_style

    @property
    def global_scale(self):
        return self._base_scale

    @property
    def raw_graphic_elements(self):
        return self._raw_graphic_elements
