from functools import cached_property

import numpy as np

from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    OpenCurveGeo
)
from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_LINE, DEFAULT_TEXT
)


class DimensioningGeo(ObjectGeo):
    """
    Geometrische Darstellung einer Bemaßung zwischen zwei Punkten.
    Für Renderer ohne spezielle Unterstützung (mpl, plotly) wird eine einfache, versetzte Linie mit Maßtext gezeichnet. 
    Der TikzRenderer erkennt die Markierung im line_style und ersetzt die Zeichnung durch einen \\dimensioning-Befehl aus der structuralanalysis-Bibliothek.
    """

    CLASS_STYLES = {
        'line': DEFAULT_LINE,
        'text': DEFAULT_TEXT
    }

    def __init__(
            self,
            point_i: tuple[float, float],
            point_j: tuple[float, float],
            distance: float,
            measure: str | None = None,
            dim_type: int | None = None,
            **kwargs
    ):
        self._validate_dimensioning(point_i, point_j, distance, dim_type)
        super().__init__(origin=point_i, **kwargs)
        self._point_i = point_i
        self._point_j = point_j
        self._distance = distance
        self._dim_type = dim_type
        self._measure = (
            measure if measure is not None else self._auto_measure()
        )

    def _auto_measure(self):
        xi, zi = self._point_i
        xj, zj = self._point_j
        length = np.hypot(xj - xi, zj - zi)
        return f'{length:.2f}'

    @cached_property
    def graphic_elements(self):
        xi, zi = self._point_i
        xj, zj = self._point_j

        line_style = {
            **self._line_style,
            'element_type': 'dimensioning',      # Markierung für TikzRenderer
            'dim_point_i': self._point_i,
            'dim_point_j': self._point_j,
            'dim_distance': self._distance,
            'dim_measure': self._measure,
            'dim_type': self._dim_type,
        }
        text_style = {**self._text_style, 'element_type': 'dimensioning'}       # damit der TikzRenderer die Textstile korrekt erkennt
        # Fallback-Darstellung für mpl/plotly: einfache, versetzte Linie
        angle = np.arctan2(zj - zi, xj - xi) + np.pi / 2
        off_x = np.cos(angle) * self._distance
        off_z = np.sin(angle) * self._distance

        return [OpenCurveGeo(
            [xi, xj], [zi, zj],
            text=[self._measure],
            preferred_text_pos='0,0!',
            line_style=line_style,
            text_style=text_style,
            post_translation=(off_x, off_z)
        )]

    @cached_property
    def text_elements(self):
        return []

    @staticmethod
    def _validate_dimensioning(point_i, point_j, distance, dim_type):
        for name, p in (('point_i', point_i), ('point_j', point_j)):
            if (
                    not isinstance(p, tuple) or len(p) != 2
                    or not all(isinstance(v, (int, float)) for v in p)
            ):
                raise TypeError(
                    f'"{name}" must be a tuple of two numbers, got {p!r}'
                )

        if not isinstance(distance, (int, float)):
            raise TypeError(
                f'"distance" must be a number, got {type(distance).__name__!r}'
            )

        if dim_type is not None and dim_type not in (1, 2):
            raise ValueError('"dim_type" must be 1, 2 or None')

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'point_i={self._point_i}, point_j={self._point_j}, '
            f'distance={self._distance}, measure={self._measure!r}, '
            f'dim_type={self._dim_type})'
        )