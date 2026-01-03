from types import NoneType

import numpy as np
from functools import cached_property

from sstatics.core.preprocessing.bar import Bar

from sstatics.core.postprocessing.graphic_objects.utils.defaults import (
    DEFAULT_CIRCLE_TEXT, DEFAULT_TEXT, DEFAULT_TENSILE_ZONE, DEFAULT_BAR,
    DEFAULT_LINE, DEFAULT_TENSILE_ZONE_DISTANCE, DEFAULT_LOAD_DISTANCE,
    DEFAULT_ARROW_DISTANCE, DEFAULT_POINT_FORCE
)
from sstatics.core.postprocessing.graphic_objects.geo.object_geo import \
    ObjectGeo
from sstatics.core.postprocessing.graphic_objects.geo.effect import (
    PointLoadGeo, LineLoadGeo, TempGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.geometry import (
    OpenCurveGeo
)
from sstatics.core.postprocessing.graphic_objects.geo.hinge import (
    CombiHingeGeo, MomentHingeGeo, NormalHingeGeo, ShearHingeGeo
)


class BarGeo(ObjectGeo):

    def __init__(
            self,
            bar: Bar,
            load_distances: dict | None = None,
            global_max_line_load: float | None = None,
            show_load: bool = True,
            show_point_load_text: bool = True,
            show_line_load_texts: list[tuple[bool, bool]] | None = None,
            show_tensile_zone: bool = True,
            show_full_hinges: tuple[bool, bool] = (True, True),
            decimals: int = 2,
            sig_digits: int | None = None,
            **kwargs
    ):
        self._validate_bar(
            bar, load_distances, global_max_line_load, show_load,
            show_point_load_text, show_line_load_texts, show_tensile_zone,
            show_full_hinges, decimals, sig_digits
        )
        super().__init__(origin=(bar.node_i.x, bar.node_i.z), **kwargs)
        self._bar = bar
        self._load_distances = load_distances
        self._global_max_line_load = global_max_line_load
        self._show_load = show_load
        self._show_point_load_text = show_point_load_text
        self._show_line_load_texts = show_line_load_texts or [
            (True, True) for _ in bar.line_loads
        ]
        self._show_tensile_zone = show_tensile_zone
        self._show_full_hinges = show_full_hinges
        self._decimals = decimals
        self._sig_digits = sig_digits

    @cached_property
    def graphic_elements(self):
        return [
            *self._bar_element,
            *self._tensile_zone_element,
            *self._hinge_elements,
            *self._point_load_elements,
            *self._line_load_elements,
            *self._temp_elements
        ]

    @cached_property
    def text_elements(self):
        return []

    @property
    def _raw_graphic_elements(self):
        return [self._single_bar]

    @cached_property
    def _bar_coords(self):
        xi, zi = self._bar.node_i.x, self._bar.node_i.z
        xj, zj = self._bar.node_j.x, self._bar.node_j.z
        return [xi, xj], [zi, zj]

    @property
    def _free_bar(self):
        def shift(u, w, phi):
            return -0.25 * self._base_scale if (u and w and phi) else 0.0

        bar = self._bar
        return (
            shift(bar.hinge_u_i, bar.hinge_w_i, bar.hinge_phi_i),
            shift(bar.hinge_u_j, bar.hinge_w_j, bar.hinge_phi_j)
        )

    @property
    def _single_bar(self):
        line_style = self._resolve_style(
            self._bar, DEFAULT_BAR, self._line_style
        )
        text_style = self._resolve_style(
            self._bar, DEFAULT_CIRCLE_TEXT, self._text_style
        )
        return OpenCurveGeo(
            *self._bar_coords, text=self._text, preferred_text_pos='0,2',
            line_style=line_style, text_style=text_style
        )

    @property
    def _bar_element(self):
        return [self._single_bar.stretch(*self._free_bar)]

    @cached_property
    def _tensile_zone_translation(self):
        angle = self._bar.inclination
        s = DEFAULT_TENSILE_ZONE_DISTANCE * self._base_scale
        return np.sin(angle) * s, np.cos(angle) * s

    @property
    def _tensile_zone_element(self):
        if not self._show_tensile_zone:
            return []

        line_style = self._resolve_style(
            self._bar, DEFAULT_TENSILE_ZONE, self._line_style
        )
        return [
            OpenCurveGeo(
                *self._bar_coords,
                line_style=line_style,
                post_translation=self._tensile_zone_translation
            ).stretch(*self._free_bar)
        ]

    def _hinge_rotation(self, idx):
        return self._bar.inclination + (np.pi if idx == 1 else 0)

    @property
    def _hinge_elements(self):
        bar = self._bar
        positions = [
            (bar.node_i.x, bar.node_i.z),
            (bar.node_j.x, bar.node_j.z),
        ]
        elements = []
        hinges = [
            [(bar.hinge_w_i, ShearHingeGeo),
             (bar.hinge_phi_i, MomentHingeGeo),
             (bar.hinge_u_i, NormalHingeGeo)],
            [(bar.hinge_w_j, ShearHingeGeo),
             (bar.hinge_phi_j, MomentHingeGeo),
             (bar.hinge_u_j, NormalHingeGeo)]
        ]

        for idx, hinge_list in enumerate(hinges):
            if self._show_full_hinges[idx]:
                continue
            combi_elements = [cls for val, cls in hinge_list if val]
            if len(combi_elements) != 3:
                elements.append(
                    CombiHingeGeo(
                        positions[idx], *combi_elements,
                        rotation=self._hinge_rotation(idx),
                        scaling=self._base_scale
                    )
                )
        return elements

    def _point_load_pos(self, pos):
        [xi, xj], [zi, zj] = self._bar_coords
        return xi + (xj - xi) * pos, zi + (zj - zi) * pos

    @property
    def _point_load_elements(self):
        if not self._show_load:
            return []

        inclination = self._bar.inclination
        scale = self._base_scale
        return [
            PointLoadGeo(
                self._point_load_pos(load.position), load=load,
                distance=(
                        self._load_distances[load] / scale if
                        self._load_distances and load in self._load_distances
                        else None
                ),
                rotate_moment=(
                        inclination + (np.pi if load.position > 0.5 else 0)
                ),
                show_text=self._show_point_load_text,
                decimals=self._decimals, sig_digits=self._sig_digits,
                line_style=self._resolve_style(
                    load, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    load, DEFAULT_TEXT, self._text_style
                ),
                scaling=scale
            ) for load in self._bar.point_loads
        ]

    @property
    def _max_line_load_value(self):
        return (
            self._global_max_line_load if self._global_max_line_load else
            max(
                max((abs(load.pi), abs(load.pj)))
                for load in self._bar.line_loads
            )
        )

    @property
    def _line_load_elements(self):
        if not self._show_load:
            return []

        scale = self._base_scale
        return [
            LineLoadGeo(
                self._bar_coords, load=load,
                distance_to_bar=(
                        self._load_distances[load] if
                        self._load_distances and load in self._load_distances
                        else DEFAULT_LOAD_DISTANCE * scale
                ),
                distance_to_arrow=DEFAULT_ARROW_DISTANCE * scale,
                global_max_value=self._max_line_load_value,
                show_texts=self._show_line_load_texts[i],
                decimals=self._decimals, sig_digits=self._sig_digits,
                arrow_style={
                    k: v * scale for k, v in DEFAULT_POINT_FORCE.items()
                },
                line_style=self._resolve_style(
                    load, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    load, DEFAULT_TEXT, self._text_style
                )
            ) for i, load in enumerate(self._bar.line_loads)
        ]

    @property
    def _temp_elements(self):
        return [TempGeo(
            bar_coords=self._bar_coords, temp=self._bar.temp,
            decimals=self._decimals, sig_digits=self._sig_digits,
            rotation=self._bar.inclination
        )] if self._show_load else []

    @staticmethod
    def _validate_bar(
            bar, load_distances, global_max_line_load, show_load,
            show_point_load_text, show_line_load_texts, show_tensile_zone,
            show_full_hinges, decimals, sig_digits
    ):
        if not isinstance(bar, Bar):
            raise TypeError(f'"bar" must be a Bar, got {type(bar).__name__!r}')

        if not isinstance(load_distances, (dict, NoneType)):
            raise TypeError(
                f'"load_distances" must be dict or None, '
                f'got {type(load_distances).__name__!r}'
            )

        if isinstance(load_distances, dict) and not all(
                isinstance(v, (int, float)) for v in load_distances.values()
        ):
            raise TypeError(
                'all values of load_distances must be int or float'
            )

        if not isinstance(global_max_line_load, (int, float, NoneType)):
            raise TypeError(
                f'"global_max_line_load" must be int, float or None, got '
                f'{type(global_max_line_load).__name__!r}'
            )

        if not isinstance(show_load, bool):
            raise TypeError(
                f'"show_load" must be a boolean, got '
                f'{type(show_load).__name__!r}'
            )

        if not isinstance(show_point_load_text, bool):
            raise TypeError(
                f'"show_point_load_text" must be a boolean, got '
                f'{type(show_point_load_text).__name__!r}'
            )

        if not isinstance(show_line_load_texts, (list, NoneType)):
            raise TypeError(
                f'"show_line_load_texts" must be list or None, got '
                f'{type(show_line_load_texts).__name__!r}'
            )

        if (
                isinstance(show_line_load_texts, list) and
                not all(isinstance(v, tuple) for v in show_line_load_texts)
        ):
            raise TypeError('values of show_line_load_texts must be tuple')

        if isinstance(show_line_load_texts, list) and not all(
                isinstance(v, bool) for value in show_line_load_texts
                for v in value
        ):
            raise TypeError('values of show_line_load_texts must be boolean')

        if not isinstance(show_tensile_zone, bool):
            raise TypeError(
                f'"show_tensile_zone" must be a boolean, got '
                f'{type(show_tensile_zone).__name__!r}'
            )

        if not isinstance(show_full_hinges, tuple):
            raise TypeError(
                f'"show_full_hinges" must be a tuple, got '
                f'{type(show_full_hinges).__name__!r}'
            )

        if (
                isinstance(show_full_hinges, tuple)
                and not all(isinstance(v, bool) for v in show_full_hinges)
        ):
            raise TypeError(
                'every element of "show_full_hinges" must be a bool'
            )

        if not isinstance(decimals, int):
            raise TypeError(
                f'"decimals" must be int or None, '
                f'got {type(decimals).__name__!r}'
            )

        if not isinstance(sig_digits, (int, NoneType)):
            raise TypeError(
                f'"sig_digits" must be int or None, '
                f'got {type(sig_digits).__name__!r}'
            )

        if sig_digits is not None and sig_digits <= 0:
            raise ValueError('"sig_digits" has to be greater than zero.')

    @property
    def bar(self):
        return self._bar

    @property
    def load_distances(self):
        return self._load_distances

    @property
    def global_max_line_load(self):
        return self._global_max_line_load

    @property
    def show_load(self):
        return self._show_load

    @property
    def show_point_load_text(self):
        return self._show_point_load_text

    @property
    def show_line_load_texts(self):
        return self._show_line_load_texts

    @property
    def show_tensile_zone(self):
        return self._show_tensile_zone

    @property
    def show_full_hinges(self):
        return self._show_full_hinges

    @property
    def decimals(self):
        return self._decimals

    @property
    def sig_digits(self):
        return self._sig_digits

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'bar={self._bar}, '
            f'global_max_line_load={self._global_max_line_load}, '
            f'show_load={self._show_load}, '
            f'show_point_load_texts={self._show_point_load_text}, '
            f'show_line_load_texts={self._show_line_load_texts}, '
            f'show_tensile_zone={self._show_tensile_zone}, '
            f'show_full_hinges={self._show_full_hinges}, '
            f'decimals={self._decimals}, '
            f'sig_digits={self._sig_digits}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
