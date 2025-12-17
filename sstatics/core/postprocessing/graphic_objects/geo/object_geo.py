
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
    r"""
    Abstract base class for all graphic objects in 2D space.

    This class defines the interface and common functionality for objects that
    can be represented graphically with lines and text. It handles
    transformation (post_translation, scaling, rotation) and style management.

    Parameters
    ----------
    origin : tuple[float, float], default=(0.0, 0.0)
        Pivot point for rotation and scaling.
    rotation : float, default=0.0
        Rotation angle in radians, counterclockwise.
    scaling : float, default=1.0
        Uniform scaling factor applied relative to the origin.
    pre_translation : tuple[float, float], default=(0.0, 0.0)
        Translation vector applied before rotation and scaling.
    post_translation : tuple[float, float], default=(0.0, 0.0)
        Translation vector applied after rotation and scaling.
    text : str, default=''
        Content of the label or annotation.
    line_style : dict, optional
        Dictionary specifying line style attributes (color, width, etc.).
        Overrides default line style.
    text_style : dict, optional
        Dictionary specifying text style attributes (font size, color, etc.).
        Overrides default text style.

    Raises
    ------
    TypeError
        If `line_style` or `text_style` is not a dictionary.
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
        Return a list of polylines representing the object.

        Each polyline is a sequence of (x, z) tuples.

        Returns
        -------
        list[list[tuple[float, float]]]
            List of polylines for rendering.

        Notes
        -----
        Subclasses must override this method. Can return an empty list if the
        object has no line representation.
        """
        return []

    @cached_property
    @abc.abstractmethod
    def text_elements(self) -> list[tuple[float, float, str, dict]]:
        """
        Return a list of text objects associated with the graphic object.

        Each text object could include position, content, and style.

        Returns
        -------
        list[dict]
            List of text objects for rendering.

        Notes
        -----
        Subclasses must override this method. Can return an empty list if the
        object has no text.
        """
        return []

    def _raw_graphic_elements(self):
        return self.graphic_elements

    @cached_property
    def _boundaries(self) -> tuple[float, float, float, float]:
        """Return the outer geometric boundaries of the object.

        This method computes the minimum and maximum coordinates of all
        polylines returned by :py:meth:`graphic_elements`. The result defines
        the rectangular bounding box that encloses the entire object.

        Returns
        -------
        tuple[float, float, float, float]
            A tuple ``(x_min, x_max, z_min, z_max)`` representing the extreme
            coordinates of the object.

        Notes
        -----
        If the object has no polylines, all values are returned as ``0.0``.
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
        for element in obj._raw_graphic_elements():
            if self._is_geo_object(element):
                # Rekursion in Kindobjekte
                for x, z, style in self._iter_raw_graphic_elements(element):
                    x, z = obj.transform(x, z)
                    yield x, z, style
            else:
                x, z, style = element
                x, z = obj.transform(x, z)
                yield x, z, style

    def _is_geo_object(self, obj):
        return hasattr(obj, "_raw_graphic_elements") and callable(
            obj._raw_graphic_elements)

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

        The base scale is used to ensure that visual elements such as text or
        markers maintain consistent proportions across objects of different
        sizes. It depends on the maximum geometric extent of the object.

        Returns
        -------
        float
            Scaling factor computed as ``0.04 * max(dx, dz) + 0.01``, where
            ``dx`` and ``dz`` are the horizontal and vertical dimensions of
            the object.

        Notes
        -----
        This property is primarily used for adjusting text or symbol positions
        relative to the object size.
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
        """
        Validate text, line_style and text_style parameters.

        Parameters
        ----------
        text : str
            Label or annotation.
        line_style : dict
            Style dictionary for lines.
        text_style : dict
            Style dictionary for text.

        Raises
        ------
        TypeError
            If `text` is not String, list, tuple, int or float.
            If `line_style` or `text_style` is not a dictionary.

        Notes
        -----
        This method is called by the constructor to ensure the object receives
        valid style dictionaries.
        """
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
