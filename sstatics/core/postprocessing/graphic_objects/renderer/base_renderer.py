
from abc import ABC, abstractmethod

import numpy as np

from .convert import convert_style
from ..geo.object_geo import ObjectGeo
from ..utils.defaults import PLOTLY, MPL


class AbstractRenderer(ABC):

    _show_grid: bool
    _show_axis: bool

    def __init__(self, mode: str):
        self._validate(mode)
        self._mode = mode
        self._all_graphic_elements = []

    @abstractmethod
    def _layout(self):
        pass

    def add_objects(self, *obj: ObjectGeo):
        self._all_graphic_elements.clear()
        for o in obj:
            for x, z, style in self._iter_graphic_elements(o):
                self._all_graphic_elements.append((x, z, style))
                style = convert_style(style, self._mode)
                self.add_graphic(x, z, **style)

            for x, z, text, style in self._iter_text_elements(o):
                style = convert_style(style, self._mode)
                if self._mode == 'mpl':
                    self.add_text(x, z, text, **style, ha='center')
                else:
                    self.add_text(x, z, text, **style)

    @abstractmethod
    def add_graphic(self, x, z, **style):
        pass

    @abstractmethod
    def add_text(self, x, z, text, **style):
        pass

    @abstractmethod
    def figure(self):
        pass

    @abstractmethod
    def show(self):
        pass

    def _iter_graphic_elements(self, obj, transform_fn=None):
        if transform_fn is None:
            def transform_fn(x, z):
                return obj.transform(x, z)

        else:
            def transform_fn(x, z, prev=transform_fn, current=obj.transform):
                x, z = prev(x, z)
                return current(x, z)

        for element in obj.graphic_elements:
            if hasattr(element, "graphic_elements"):
                yield from self._iter_graphic_elements(element, transform_fn)
            else:
                x, z, style = element
                x, z = transform_fn(x, z)
                yield x, z, style

    def _iter_text_elements(self, obj, transform_fn=None):
        if transform_fn is None:
            def transform_fn(x, z):
                return obj.transform(x, z)
        else:
            def transform_fn(x, z, prev=transform_fn, current=obj.transform):
                x, z = prev(x, z)
                return current(x, z)

        for element in getattr(obj, "text_elements", []):
            if isinstance(element, tuple) and len(element) == 4:
                x, z, text, style = element
                x, z = transform_fn(x, z)
                x_opt, z_opt = self._find_optimal_text_position(x, z, obj)
                yield x_opt, z_opt, text, style

        # Rekursion für Unterobjekte (auch verschachtelt)
        for sub in getattr(obj, "graphic_elements", []):
            if (
                    hasattr(sub, "text_elements")
                    or hasattr(sub, "graphic_elements")
            ):
                yield from self._iter_text_elements(sub, transform_fn)

    def _find_optimal_text_position(self, x, z, obj):
        """Berechnet die optimale Textposition nach Transformation."""
        offset = self._base_text_offset(obj)
        positions = [
            (x, z + offset),
            (x - offset, z),
            (x, z - offset),
            (x + offset, z)
        ]
        for x_try, z_try in positions:
            if not self._text_collision(x_try, z_try):
                return x_try, z_try
        return positions[0]

    def _base_text_offset(self, obj):
        try:
            dx, dz = obj._max_dimensions
            return 0.08 * max(dx, dz) + 0.02
        except Exception:
            return 0.05

    def _text_collision(self, x, z, margin=0.01):
        for px_list, pz_list, _ in self._all_graphic_elements:
            coords = [(xi, zi) for xi, zi in zip(px_list, pz_list)
                      if xi is not None and zi is not None]
            if len(coords) < 2:
                continue
            for (x0, z0), (x1, z1) in zip(coords[:-1], coords[1:]):
                dist = self._point_to_segment_distance(x, z, x0, z0, x1, z1)
                if dist < margin:
                    return True
        return False

    @staticmethod
    def _point_to_segment_distance(px, pz, x0, z0, x1, z1):
        """Kürzester Abstand zwischen Punkt und Liniensegment."""
        dx, dz = px - x0, pz - z0
        sx, sz = x1 - x0, z1 - z0
        seg_len_sq = sx ** 2 + sz ** 2
        if seg_len_sq == 0:
            return np.hypot(dx, dz)
        t = max(0, min(1, (dx * sx + dz * sz) / seg_len_sq))
        closest_x = x0 + t * sx
        closest_z = z0 + t * sz
        return np.hypot(px - closest_x, pz - closest_z)

    @staticmethod
    def _validate(mode):
        if not isinstance(mode, str):
            raise TypeError(
                f'mode must be a string, got {type(mode).__name__!r}'
            )
        if mode not in (PLOTLY, MPL):
            raise ValueError(
                f"Invalid mode {mode!r}. Expected one of: {PLOTLY!r}, {MPL!r}."
            )

    @property
    def mode(self):
        return self._mode

    @property
    def all_graphic_elements(self):
        return self._all_graphic_elements

    @property
    def show_grid(self):
        return self._show_grid

    @property
    def show_axis(self):
        return self._show_axis
