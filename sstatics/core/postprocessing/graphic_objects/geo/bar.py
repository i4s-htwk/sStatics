
import numpy as np
from sympy.core.cache import cached_property

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
            show_load: bool = True,
            show_load_text: bool = True,
            show_tensile_zone: bool = True,
            **kwargs
    ):
        self._validate_bar(bar, show_load, show_load_text, show_tensile_zone)
        super().__init__(origin=(bar.node_i.x, bar.node_i.z), **kwargs)
        self._bar = bar
        self._show_load = show_load
        self._show_load_text = show_load_text
        self._show_tensile_zone = show_tensile_zone

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
            *self._bar_coords, text=self._text, preferred_text_pos='1,0',
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
            combi_elements = [cls for val, cls in hinge_list if val]
            if len(combi_elements) != 3:
                rotation = bar.inclination + (np.pi if idx == 1 else 0)
                elements.append(
                    CombiHingeGeo(
                        positions[idx], *combi_elements,
                        rotation=rotation, scaling=self._base_scale
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
        return [
            PointLoadGeo(
                self._point_load_pos(load.position), load=load,
                rotate_moment=(
                        inclination + (np.pi if load.position > 0.5 else 0)
                ),
                show_text=self._show_load_text,
                line_style=self._resolve_style(
                    load, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    load, DEFAULT_TEXT, self._text_style
                ),
                scaling=self._base_scale
            ) for load in self._bar.point_loads
        ]

    @property
    def _line_load_elements(self):
        if not self._show_load:
            return []

        scale = self._base_scale
        return [
            LineLoadGeo(
                self._bar_coords, load=load,
                distance_to_bar=DEFAULT_LOAD_DISTANCE * scale,
                distance_to_arrow=DEFAULT_ARROW_DISTANCE * scale,
                show_text=self._show_load_text,
                arrow_style={
                    k: v * scale for k, v in DEFAULT_POINT_FORCE.items()
                },
                line_style=self._resolve_style(
                    load, DEFAULT_LINE, self._line_style
                ),
                text_style=self._resolve_style(
                    load, DEFAULT_TEXT, self._text_style
                )
            ) for load in self._bar.line_loads
        ]

    @property
    def _temp_elements(self):
        return [TempGeo(
            bar_coords=self._bar_coords, temp=self._bar.temp,
            rotation=self._bar.inclination
        )]

    @staticmethod
    def _validate_bar(bar, show_load, show_load_text, show_tensile_zone):
        if not isinstance(bar, Bar):
            raise TypeError(f'"bar" must be a Bar, got {type(bar).__name__!r}')

        if not isinstance(show_load, bool):
            raise TypeError(
                f'"show_load" must be a boolean, got '
                f'{type(show_load).__name__!r}'
            )

        if not isinstance(show_load_text, bool):
            raise TypeError(
                f'"show_load_text" must be a boolean, got '
                f'{type(show_load_text).__name__!r}'
            )

        if not isinstance(show_tensile_zone, bool):
            raise TypeError(
                f'"show_tensile_zone" must be a boolean, got '
                f'{type(show_tensile_zone).__name__!r}'
            )

    @property
    def bar(self):
        return self._bar

    @property
    def show_load(self):
        return self._show_load

    @property
    def show_load_text(self):
        return self._show_load_text

    @property
    def show_tensile_zone(self):
        return self._show_tensile_zone

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'origin={self._origin}, '
            f'bar={self._bar}, '
            f'show_load={self._show_load}, '
            f'show_load_text={self._show_load_text}, '
            f'show_tensile_zone={self._show_tensile_zone}, '
            f'line_style={self._line_style}, '
            f'text_style={self._text_style}, '
            f'Transform={self._transform})'
        )
