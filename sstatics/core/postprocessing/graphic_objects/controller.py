
from functools import cached_property

from sstatics.core.preprocessing.geometry import Polygon
from .geo.geometry import PolygonGeo
from .renderer import ObjectRenderer
from .utils.defaults import PLOTLY, MPL


class ControllerGraphic:

    def __init__(
            self,
            *objects,
    ):
        self._validate(objects)
        self._objects = list(objects)
        self._extra_geos = []

    def figure(self, show_axis=True, show_grid=False):
        return self._render.figure(show_axis, show_grid)

    def show(self, show_axis=True, show_grid=False):
        self._render.show(show_axis, show_grid)

    def add_geos(self, *geos):
        self._extra_geos.append(*geos)

    @cached_property
    def _render(self):
        return ObjectRenderer(*self.geos)

    @property
    def _build_geos(self):
        geos = []
        for obj in self._objects:
            if isinstance(obj, (list, tuple)):
                geos.append([self._core_to_geo(o) for o in obj])
            elif isinstance(obj, str):
                geos.append(obj)
            else:
                geos.append(self._core_to_geo(obj))
        return geos

    # TODO: better name for core
    @staticmethod
    def _core_to_geo(o):
        if isinstance(o, Polygon):
            return PolygonGeo(o)

    @staticmethod
    def _validate(objects):

        def _validate_core_objects(obj):
            if not isinstance(obj, VALID_CORE_TO_GEO_OBJECTS):
                raise ValueError(
                    f'Unsupported or invalid input: {obj!r}. '
                    f'Expected a core object, a geo object, a list/tuple of '
                    f'them, or a mode string (PLOTLY/MPL).'
                )

        for obj in objects:
            if isinstance(obj, str) and obj not in (PLOTLY, MPL):
                raise ValueError(
                    f"Invalid mode {obj!r}. Expected one of: {PLOTLY!r}, "
                    f"{MPL!r}."
                )

            if isinstance(obj, (list, tuple)):
                for o in obj:
                    _validate_core_objects(o)
            else:
                _validate_core_objects(obj)

    @property
    def geos(self):
        return self._build_geos + self._extra_geos


VALID_CORE_TO_GEO_OBJECTS = (
    str, Polygon
)
